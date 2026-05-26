"""Operators enabled by external modules."""

import math

import tvm
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

from mlc_llm.support import logging

from . import extern as _extern

logger = logging.getLogger(__name__)


WARN_FLASHINFER_GROUP_SIZE = False
WARN_FLASHINFER_HEAD_DIM = False


def attention(  # pylint: disable=invalid-name,too-many-locals,too-many-statements,too-many-arguments, unused-argument
    q: nn.Tensor,
    k: nn.Tensor,
    v: nn.Tensor,
    casual_mask: nn.Tensor,
    attn_score_scaling_factor: float = 1.0,
    qk_dtype: str = None,
) -> nn.Tensor:
    """Attention with casual mask.

    --- Variables ---
    s: sequence length of the current query
    t: total sequence length
    d: head dimension
    h, h_q: number of heads in query
    h_kv: number of heads in key and value
    b: batch size = 1

    --- Shapes ---
    q: [b, s, h_q, d]
    k: [t, h_kv, d]
    v: [t, h_kv, d]
    o: [1, s, hidden = h_q * d]

    --- Computation ---

    .. code-block:: python

        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=1)
            v = v.repeat(h_q // h_kv, axis=1)
        q -> [b, h, s, d]
        k, v -> [b, h, t, d]
        attn = q @ k^T / sqrt(d) * attn_score_scaling_factor  # [b, h, s, t]
        attn = softmax_with_mask(attn, casual_mask, axis=-1)
        o = attn @ v  # [b, h, s, d]
        o -> [b, s, h * d]

    --- Other params ---
    qk_dtype: if set, `matmul(Q, K, out_dtype=qk_dtype)`, (otherwise use `q.dtype` as `out_dtype`).
        For FlashInfer, if "float32", sets `allow_fp16_qk_reduction` to False; otherwise no effect.
    """
    assert q.ndim == 4 and k.ndim in [3, 4] and v.ndim in [3, 4]
    b, s, h_q, d = q.shape
    t, h_kv, _ = k.shape[-3:]
    group_size = h_q // h_kv

    def _fallback():
        from tvm.relax.frontend.nn.llm.kv_cache import (  # pylint: disable=import-outside-toplevel
            _attention_sequence_prefill,
        )

        nonlocal q, k, v, qk_dtype
        if k.ndim == 3:
            k = op.reshape(k, [b, t, h_kv, d])
        if v.ndim == 3:
            v = op.reshape(v, [b, t, h_kv, d])
        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)

        target = tvm.target.Target("cuda")
        attn_output, _ = op.tensor_ir_op(
            _attention_sequence_prefill(  # pylint: disable=no-value-for-parameter
                h_kv=h_kv,
                h_q=h_q,
                d=d,
                dtype=q.dtype,
                target=target,
                sm_scale=attn_score_scaling_factor / (d**0.5),
            ),
            "sequence_prefill",
            [q, k, v],
            [
                Tensor.placeholder([b, s, h_q, d], q.dtype),
                Tensor.placeholder([b, s, h_q], q.dtype),
            ],
        )

        output = op.reshape(attn_output, shape=(b, s, h_q * d))
        return output

    # FlashInfer Implementation
    if (
        _extern.get_store().flashinfer
        and attn_score_scaling_factor == 1.0
        and q.dtype == "float16"
        and k.dtype == "float16"
        and v.dtype == "float16"
    ):
        if group_size not in [1, 4, 6, 8]:
            global WARN_FLASHINFER_GROUP_SIZE  # pylint: disable=global-statement
            if not WARN_FLASHINFER_GROUP_SIZE:
                WARN_FLASHINFER_GROUP_SIZE = True
                logger.warning(
                    "FlashInfer only supports group size in [1, 4, 6, 8], but got %d. Skip and "
                    "fallback to default implementation.",
                    group_size,
                )
            return _fallback()
        if d not in [128]:
            global WARN_FLASHINFER_HEAD_DIM  # pylint: disable=global-statement
            if not WARN_FLASHINFER_HEAD_DIM:
                WARN_FLASHINFER_HEAD_DIM = True
                logger.warning(
                    "FlashInfer only supports head_dim in [128], but got %d. Skip and fallback to "
                    "default implementation.",
                    d,
                )
            return _fallback()
        rope_theta = 0.0
        rope_scale = 1.0
        qkv_layout = 0  # "NHD", N for seq_len, H for num_heads, D for head_dim
        rotary_mode = 0  # "kNone"
        casual = 1  # True
        fp16_qk = 1  # True
        if qk_dtype == "float32":
            fp16_qk = 0  # False

        # 32MB scratchpad
        scratch = op.empty([8192 * 1024], dtype="float32")  # pylint: disable=no-member

        def _decode():
            return op.extern(
                name="flashinfer.single_decode",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    qkv_layout,
                    rotary_mode,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        def _prefill():
            return op.extern(
                name="flashinfer.single_prefill",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    casual,
                    qkv_layout,
                    rotary_mode,
                    fp16_qk,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        if isinstance(s, int) and s == 1:
            func = "decode"
        else:
            func = "prefill"
        return {
            "decode": _decode,
            "prefill": _prefill,
        }[func]()

    # Fallback Implementation
    return _fallback()


def get_gated_delta_net_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    state_in_layer: Tensor,
    dtype: str,
):
    """Creates a TIR function for the GatedDeltaNet recurrent computation.

    Thread-per-column design: each thread owns one column of the state matrix.
    State S is (key_head_dim x value_head_dim) per head, accumulated in fp32.

    Supports arbitrary sequence length via an inner `for t in range(seq_len)` loop,
    matching RWKV6's approach. During prefill (seq_len > 1), the recurrence accumulates
    state across all tokens sequentially. During decode (seq_len = 1), it's a single step.

    For GVA (num_value_heads > num_key_heads), Q/K are expanded via repeat.
    The kernel operates on value_heads (the larger dimension).
    """

    b, s, n_kh, K = q.shape
    _, _, n_vh, V = v.shape

    heads_per_group = n_vh // n_kh  # 1 for 0.8B, 2 for 4B
    num_key_heads = n_kh
    num_value_heads = n_vh
    TX = V // 4
    TY = 4

    @T.prim_func
    def gdn_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )
        state_out_buf = T.match_buffer(
            state_out_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(TX, thread="threadIdx.x"):
                    for col_v in T.vectorized(4):
                        kh = h_idx // heads_per_group
                        with T.sblock("gdn_state"):
                            out_buf_local = T.alloc_buffer(
                                (TY, TX * 4), dtype="float32", scope="local"
                            )
                            out_buf_shared = T.alloc_buffer(
                                (TY, TX * 4), dtype="float32", scope="shared"
                            )
                            state_out_buf_local = T.alloc_buffer(
                                (K, V), dtype="float32", scope="local"
                            )
                            for row in T.unroll(K // TY):
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("init_state"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vr = T.axis.opaque(K, row * TY + ty)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        state_out_buf_local[vr, vc] = state_in_buf[vb, vh, vr, vc]

                            for t in range(seq_len):
                                # 1. Decay state: S = gate * S
                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("decay"):
                                            vb = T.axis.spatial(batch_size, b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            state_out_buf_local[vr, vc] = (
                                                state_out_buf_local[vr, vc] * gate_buf[vb, vt, vh]
                                            )

                                # 2. dot(S[:, col], k[:])
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_init"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_local[vy, vc] = T.float32(0)

                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("dot_sk"):
                                            vb = T.axis.spatial(batch_size, b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            vy = T.axis.opaque(TY, ty)
                                            out_buf_local[vy, vc] = out_buf_local[
                                                vy, vc
                                            ] + state_out_buf_local[vr, vc] * T.cast(
                                                k_buf[vb, vt, kh, vr], "float32"
                                            )
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_shared[vy, vc] = out_buf_local[vy, vc]
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        # T.where(ty<1)
                                        out_buf_shared[0, vc] = (
                                            out_buf_shared[0, vc]
                                            + out_buf_shared[1, vc]
                                            + out_buf_shared[2, vc]
                                            + out_buf_shared[3, vc]
                                        )
                                # 3. Delta rule: S += k * beta * (v - dot_sk)
                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("delta"):
                                            vb = T.axis.spatial(batch_size, b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            state_out_buf_local[vr, vc] = state_out_buf_local[
                                                vr, vc
                                            ] + T.cast(k_buf[vb, vt, kh, vr], "float32") * beta_buf[
                                                vb, vt, vh
                                            ] * (
                                                T.cast(v_buf[vb, vt, vh, vc], "float32")
                                                - out_buf_shared[0, vc]
                                            )
                                # 4. Output: o[t, col] = dot(S_updated[:, col], q[t, :]) / √K
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sq_init"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_local[vy, vc] = T.float32(0)

                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("dot_sq"):
                                            vb = T.axis.spatial(batch_size, b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            vy = T.axis.opaque(TY, ty)
                                            out_buf_local[vy, vc] = out_buf_local[
                                                vy, vc
                                            ] + state_out_buf_local[vr, vc] * T.cast(
                                                q_buf[vb, vt, kh, vr], "float32"
                                            )

                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_shared[vy, vc] = out_buf_local[vy, vc]

                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        # T.where(ty<1)
                                        out_buf_shared[0, vc] = (
                                            out_buf_shared[0, vc]
                                            + out_buf_shared[1, vc]
                                            + out_buf_shared[2, vc]
                                            + out_buf_shared[3, vc]
                                        )

                                # 5. Scale
                                with T.sblock("scale"):
                                    vb = T.axis.spatial(batch_size, b_idx)
                                    vt = T.axis.opaque(seq_len, t)
                                    vh = T.axis.spatial(num_value_heads, h_idx)
                                    vc = T.axis.spatial(V, col * 4 + col_v)
                                    out_buf[vb, vt, vh, vc] = out_buf_shared[0, vc] * T.float32(
                                        1.0 / math.sqrt(K)
                                    )
                            for row in T.unroll(K // TY):
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("init_state"):
                                        vb = T.axis.spatial(batch_size, b_idx)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vr = T.axis.opaque(K, row * TY + ty)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        state_out_buf[vb, vh, vr, vc] = state_out_buf_local[vr, vc]

    @T.prim_func
    def gdn_func_with_single_batch(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        seq_len = T.int64()
        q_buf = T.match_buffer(q_handle, (T.int64(1), seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (T.int64(1), seq_len, num_key_heads, K), dtype=dtype)
        v_buf = T.match_buffer(v_handle, (T.int64(1), seq_len, num_value_heads, V), dtype=dtype)
        gate_buf = T.match_buffer(
            gate_handle, (T.int64(1), seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (T.int64(1), seq_len, num_value_heads), dtype="float32"
        )
        state_in_buf = T.match_buffer(
            state_in_handle, (T.int64(1), num_value_heads, K, V), dtype="float32"
        )
        out_buf = T.match_buffer(
            out_handle, (T.int64(1), seq_len, num_value_heads, V), dtype="float32"
        )
        state_out_buf = T.match_buffer(
            state_out_handle, (T.int64(1), num_value_heads, K, V), dtype="float32"
        )

        for b_idx in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(TX, thread="threadIdx.x"):
                    for col_v in T.vectorized(4):
                        kh = h_idx // heads_per_group
                        with T.sblock("gdn_state"):
                            out_buf_local = T.alloc_buffer(
                                (TY, TX * 4), dtype="float32", scope="local"
                            )
                            out_buf_shared = T.alloc_buffer(
                                (TY, TX * 4), dtype="float32", scope="shared"
                            )
                            state_out_buf_local = T.alloc_buffer(
                                (K, V), dtype="float32", scope="local"
                            )
                            for row in T.unroll(K // TY):
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("init_state"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vr = T.axis.opaque(K, row * TY + ty)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        state_out_buf_local[vr, vc] = state_in_buf[vb, vh, vr, vc]

                            for t in range(seq_len):
                                # 1. Decay state: S = gate * S
                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("decay"):
                                            vb = T.axis.spatial(T.int64(1), b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            state_out_buf_local[vr, vc] = (
                                                state_out_buf_local[vr, vc] * gate_buf[vb, vt, vh]
                                            )

                                # 2. dot(S[:, col], k[:])
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_init"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_local[vy, vc] = T.float32(0)

                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("dot_sk"):
                                            vb = T.axis.spatial(T.int64(1), b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            vy = T.axis.opaque(TY, ty)
                                            out_buf_local[vy, vc] = out_buf_local[
                                                vy, vc
                                            ] + state_out_buf_local[vr, vc] * T.cast(
                                                k_buf[vb, vt, kh, vr], "float32"
                                            )
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_shared[vy, vc] = out_buf_local[vy, vc]
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        # T.where(ty<1)
                                        out_buf_shared[0, vc] = (
                                            out_buf_shared[0, vc]
                                            + out_buf_shared[1, vc]
                                            + out_buf_shared[2, vc]
                                            + out_buf_shared[3, vc]
                                        )
                                # 3. Delta rule: S += k * beta * (v - dot_sk)
                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("delta"):
                                            vb = T.axis.spatial(T.int64(1), b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            state_out_buf_local[vr, vc] = state_out_buf_local[
                                                vr, vc
                                            ] + T.cast(k_buf[vb, vt, kh, vr], "float32") * beta_buf[
                                                vb, vt, vh
                                            ] * (
                                                T.cast(v_buf[vb, vt, vh, vc], "float32")
                                                - out_buf_shared[0, vc]
                                            )
                                # 4. Output: o[t, col] = dot(S_updated[:, col], q[t, :]) / √K
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sq_init"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_local[vy, vc] = T.float32(0)

                                for row in T.unroll(K // TY):
                                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                        with T.sblock("dot_sq"):
                                            vb = T.axis.spatial(T.int64(1), b_idx)
                                            vt = T.axis.opaque(seq_len, t)
                                            vr = T.axis.opaque(K, row * TY + ty)
                                            vh = T.axis.spatial(num_value_heads, h_idx)
                                            vc = T.axis.spatial(V, col * 4 + col_v)
                                            vy = T.axis.opaque(TY, ty)
                                            out_buf_local[vy, vc] = out_buf_local[
                                                vy, vc
                                            ] + state_out_buf_local[vr, vc] * T.cast(
                                                q_buf[vb, vt, kh, vr], "float32"
                                            )

                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        vy = T.axis.opaque(TY, ty)
                                        out_buf_shared[vy, vc] = out_buf_local[vy, vc]

                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("dot_sk_shared_store"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vt = T.axis.opaque(seq_len, t)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        # T.where(ty<1)
                                        out_buf_shared[0, vc] = (
                                            out_buf_shared[0, vc]
                                            + out_buf_shared[1, vc]
                                            + out_buf_shared[2, vc]
                                            + out_buf_shared[3, vc]
                                        )

                                # 5. Scale
                                with T.sblock("scale"):
                                    vb = T.axis.spatial(T.int64(1), b_idx)
                                    vt = T.axis.opaque(seq_len, t)
                                    vh = T.axis.spatial(num_value_heads, h_idx)
                                    vc = T.axis.spatial(V, col * 4 + col_v)
                                    out_buf[vb, vt, vh, vc] = out_buf_shared[0, vc] * T.float32(
                                        1.0 / math.sqrt(K)
                                    )
                            for row in T.unroll(K // TY):
                                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                                    with T.sblock("init_state"):
                                        vb = T.axis.spatial(T.int64(1), b_idx)
                                        vh = T.axis.spatial(num_value_heads, h_idx)
                                        vr = T.axis.opaque(K, row * TY + ty)
                                        vc = T.axis.spatial(V, col * 4 + col_v)
                                        state_out_buf[vb, vh, vr, vc] = state_out_buf_local[vr, vc]

    return op.tensor_ir_op(
        gdn_func_with_single_batch if b == 1 else gdn_func,
        "gated_delta_net",
        [q, k, v, gate, beta, state_in_layer],
        [
            Tensor.placeholder([b, s, n_vh, V], "float32"),
            Tensor.placeholder([b, n_vh, K, V], "float32"),
        ],
    )
