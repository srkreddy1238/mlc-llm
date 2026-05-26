import tvm
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

from mlc_llm.support import logging


# Depthwise conv
def get_depthwise_conv_with_state_op(state: Tensor, qkv_in: Tensor, weight: Tensor):
    ks_m1 = state.shape[1]
    dim = qkv_in.shape[2]
    seq = qkv_in.shape[1]
    kernel_size = weight.shape[2]

    @T.prim_func
    def _depthwise_conv1d(
        var_lv2189: T.handle,
        var_matmul993: T.handle,
        model_layers_0_linear_attn_conv1d_weight4: T.Buffer(
            (dim, T.int64(1), kernel_size), "float16"
        ),
        var_depthwise_conv1d: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size = T.int64()
        seq_len = T.int64()
        lv2189 = T.match_buffer(var_lv2189, (batch_size, ks_m1, dim), "float16")
        matmul993 = T.match_buffer(var_matmul993, (batch_size, seq_len, dim), "float16")
        depthwise_conv1d = T.match_buffer(
            var_depthwise_conv1d, (batch_size, seq_len, dim), "float16"
        )
        if T.tvm_thread_invariant(seq_len <= 2):
            with T.sblock("root_low_seq"):
                for ax1 in T.thread_binding(batch_size, thread="blockIdx.z"):
                    for ax0 in T.thread_binding((dim + 511) // 512, thread="blockIdx.x"):
                        for ax0_0 in T.thread_binding(128, thread="threadIdx.x"):
                            for ax2_0 in range(seq_len):
                                for vec in T.vectorized(4):
                                    with T.sblock("depthwise_conv1d_init"):
                                        v0 = T.axis.spatial(batch_size, ax1)
                                        v1 = T.axis.spatial(seq_len, ax2_0)
                                        v2 = T.axis.spatial(dim, ax0 * 512 + ax0_0 * 4 + vec)
                                        T.reads()
                                        T.writes(depthwise_conv1d[v0, v1, v2])
                                        depthwise_conv1d[v0, v1, v2] = T.float16(0.0)
                            for ax2_0 in range(seq_len):
                                for kk in T.unroll(kernel_size):
                                    for vec in T.vectorized(4):
                                        with T.sblock("depthwise_conv1d"):
                                            v_bi = T.axis.spatial(batch_size, ax1)
                                            v_si = T.axis.spatial(seq_len, ax2_0)
                                            v_di = T.axis.spatial(dim, ax0 * 512 + ax0_0 * 4 + vec)
                                            v_kk = T.axis.reduce(kernel_size, kk)
                                            T.reads(
                                                lv2189[v_bi, v_si + v_kk, v_di],
                                                matmul993[v_bi, v_si + v_kk - ks_m1, v_di],
                                                model_layers_0_linear_attn_conv1d_weight4[
                                                    v_di, T.int64(0), v_kk
                                                ],
                                            )
                                            T.writes(depthwise_conv1d[v_bi, v_si, v_di])
                                            depthwise_conv1d[v_bi, v_si, v_di] = (
                                                depthwise_conv1d[v_bi, v_si, v_di]
                                                + T.if_then_else(
                                                    v_si + v_kk < ks_m1,
                                                    lv2189[v_bi, v_si + v_kk, v_di],
                                                    matmul993[v_bi, v_si + v_kk - ks_m1, v_di],
                                                )
                                                * model_layers_0_linear_attn_conv1d_weight4[
                                                    v_di, T.int64(0), v_kk
                                                ]
                                            )
        else:
            with T.sblock("root_seq"):
                depthwise_conv1d_local = T.alloc_buffer(
                    (batch_size, (seq_len + 15) // 16 * 16, dim), "float16"
                )
                for ax1 in T.thread_binding(batch_size, thread="blockIdx.z"):
                    for ax2 in T.thread_binding((seq_len + 15) // 16, thread="blockIdx.y"):
                        for ax0 in T.thread_binding((dim + 255) // 256, thread="blockIdx.x"):
                            for ax0_0 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax2_0 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax2_0_1 in T.unroll(4):
                                        for vec in T.vectorized(4):
                                            with T.sblock("depthwise_conv1d_init1"):
                                                v0 = T.axis.spatial(batch_size, ax1)
                                                v1 = T.axis.spatial(
                                                    (seq_len + 15) // 16 * 16,
                                                    ax2 * 16 + ax2_0 * 4 + ax2_0_1,
                                                )
                                                v2 = T.axis.spatial(
                                                    dim, ax0 * 256 + ax0_0 * 4 + vec
                                                )
                                                T.reads()
                                                T.writes(depthwise_conv1d_local[v0, v1, v2])
                                                depthwise_conv1d_local[v0, v1, v2] = T.float16(0.0)
                                for ax2_0 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax2_0_1 in T.unroll(4):
                                        for kk in T.unroll(kernel_size):
                                            for vec in T.vectorized(4):
                                                with T.sblock("depthwise_conv1d1"):
                                                    v_bi = T.axis.spatial(batch_size, ax1)
                                                    v_si = T.axis.spatial(
                                                        (seq_len + 15) // 16 * 16,
                                                        ax2 * 16 + ax2_0 * 4 + ax2_0_1,
                                                    )
                                                    v_di = T.axis.spatial(
                                                        dim, ax0 * 256 + ax0_0 * 4 + vec
                                                    )
                                                    v_kk = T.axis.reduce(kernel_size, kk)
                                                    T.reads(
                                                        lv2189[v_bi, v_si + v_kk, v_di],
                                                        matmul993[v_bi, v_si + v_kk - ks_m1, v_di],
                                                        model_layers_0_linear_attn_conv1d_weight4[
                                                            v_di, T.int64(0), v_kk
                                                        ],
                                                    )
                                                    T.writes(depthwise_conv1d[v_bi, v_si, v_di])
                                                    depthwise_conv1d_local[v_bi, v_si, v_di] = (
                                                        depthwise_conv1d_local[v_bi, v_si, v_di]
                                                        + T.if_then_else(
                                                            v_si + v_kk < ks_m1,
                                                            lv2189[v_bi, v_si + v_kk, v_di],
                                                            matmul993[
                                                                v_bi, v_si + v_kk - ks_m1, v_di
                                                            ],
                                                        )
                                                        * model_layers_0_linear_attn_conv1d_weight4[
                                                            v_di, T.int64(0), v_kk
                                                        ]
                                                    )
                                for ax2_0 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax2_0_1 in T.unroll(4):
                                        for vec in T.vectorized(4):
                                            with T.sblock("Store"):
                                                v_bi = T.axis.spatial(batch_size, ax1)
                                                v_si = T.axis.spatial(
                                                    (seq_len + 15) // 16 * 16,
                                                    ax2 * 16 + ax2_0 * 4 + ax2_0_1,
                                                )
                                                v_di = T.axis.spatial(
                                                    dim, ax0 * 256 + ax0_0 * 4 + vec
                                                )
                                                T.where((ax2 * 16 + ax2_0 * 4 + ax2_0_1) < seq_len)
                                                T.reads(depthwise_conv1d_local[v_bi, v_si, v_di])
                                                T.writes(depthwise_conv1d[v_bi, v_si, v_di])
                                                depthwise_conv1d[
                                                    v_bi, v_si, v_di
                                                ] = depthwise_conv1d_local[v_bi, v_si, v_di]

    return op.tensor_ir_op(
        _depthwise_conv1d,
        "depthwise_conv1d",
        args=[state, qkv_in, weight],
        out=Tensor.placeholder(qkv_in.shape, qkv_in.dtype),
    )
