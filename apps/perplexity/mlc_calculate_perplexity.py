# pylint: disable=too-many-arguments
import json
import random
import pdb
import torch
import numpy as np
import tvm

from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime import Device, Module, Object, ShapeTuple
from tvm.runtime.relax_vm import VirtualMachine
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.interface.help import HELP
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import engine_utils
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green, red
from mlc_llm.tokenizers import Tokenizer


def load_inputs_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines


def calculate_perplexity(logits, targets):
    shift_logits = logits[:, :-1, :]
    shift_labels = targets[:, 1:]
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)
    loss = loss_fct(
        torch.tensor(shift_logits, dtype=torch.float32),
        torch.tensor(shift_labels, dtype=torch.long),
    )

    return loss


def _extract_metadata(mod: Module):
    return json.loads(VirtualMachine(mod, tvm.runtime.device("cpu"))["_metadata"]())


def _load_params(
    model_weight_path: str, device: Device, model_metadata: Dict[str, Any]
) -> List[tvm.nd.NDArray]:
    params, meta = tvmjs.load_ndarray_cache(model_weight_path, device)
    param_names = [param["name"] for param in model_metadata["params"]]
    assert len(param_names) == meta["ParamSize"]

    plist = []
    for param_name in param_names:
        plist.append(params[param_name])
    return plist


def _get_tvm_module(model_weight_path: str, lib_path: str, device: Device):
    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, device)
    # vm.set_instrument(instrument)
    metadata = _extract_metadata(ex)
    params = _load_params(model_weight_path, device, metadata)
    return vm.module, params, metadata


class Perplexity:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(  # pylint: disable=too-many-arguments
        self, model: str, model_lib: str, device: Optional[str] = "auto"
    ):
        """_summary_

        Parameters
        ----------
        model: str
            The model folder after compiling with MLC-LLM build process. The parameter
            can either be the model name with its quantization scheme
            (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
            folder. In the former case, we will use the provided name to search
            for the model folder over possible paths.

        model_lib : str
            The full path to the model library file to use (e.g. a ``.so`` file).

        device : Optional[str]
            The description of the device to run on. User should provide a string in the
            form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
            'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
            local device), and 'device_id' is the device id to run on. If no 'device_id'
            is provided, it will be set to 0 by default.

        chat_config : Optional[ChatConfig]
            A ``ChatConfig`` instance partially filled. Will be used to override the
            ``mlc-chat-config.json``.

        """
        self.device = detect_device(device)
        self.mod, self.params, self.metadata = _get_tvm_module(model, model_lib, self.device)
        self.model_path = Path(model)
        self.config_file_path = self.model_path / "mlc-chat-config.json"
        with open(self.config_file_path, mode="rt", encoding="utf-8") as file:
            self.chat_config = MLCChatConfig.model_validate_json(file.read())

        conv_template = self.chat_config.conv_template

        self.conversation = (
            ConvTemplateRegistry.get_conv_template(conv_template)
            if isinstance(conv_template, str)
            else conv_template
        )
        self.tokenizer = Tokenizer(str(self.model_path))

        self.add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self.begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self.end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")
        self.nd_view_func = tvm.get_global_func("vm.builtin.reshape")
        self.sample_topp_from_prob_func = tvm.get_global_func("vm.builtin.sample_top_p_from_prob")

        try:
            self.embed_func = self.mod["embed"]
        except AttributeError as exc:
            raise RuntimeError("DebugChat only supports separate embedding layer") from exc

        self.prefill_func = self.mod["prefill"]
        self.create_kv_cache_func = None
        if self.mod.implements_function("create_flashinfer_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_flashinfer_paged_kv_cache"]
        elif self.mod.implements_function("create_tir_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_tir_paged_kv_cache"]
        else:
            # TODO: Support RNN KVState # pylint: disable=fixme
            raise RuntimeError("DebugChat cannot find create KV cache function")

        self.appeared_token_freq: Dict[int, int] = {}

    def _tokenize(self, prompt: str) -> tvm.nd.array:
        print("======================= Starts Tokenization & Embedding =======================")
        # Step 0. Generate prompt string using conversation template
        self.conversation.messages.append(("user", prompt))
        self.conversation.messages.append(("assistant", None))
        with open(self.config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        parsed_prompt = self.conversation.as_prompt(config)
        print(
            "Parsed prompt using conversation template "
            f"{green(self.conversation.name)}: {parsed_prompt}"
        )
        tokens = engine_utils.process_prompts(parsed_prompt, self.tokenizer.encode)  # type: ignore

        # TODO: Handle ImageData in DebugChat # pylint: disable=fixme
        assert len(tokens) == 1, "DebugChat will only handle TextData for now"
        if self.conversation.system_prefix_token_ids is not None:
            tokens[0] = self.conversation.system_prefix_token_ids + tokens[0]

        tokens = tvm.nd.array(np.array(tokens[0]).astype("int32"), device=self.device)
        return tokens

    def _embed(self, tokens: tvm.nd.array) -> Tuple[tvm.nd.NDArray, int]:
        input_len = tokens.shape[0]
        embedding = self.embed_func(tokens, self.params)
        embedding = self.nd_view_func(embedding, ShapeTuple([1, input_len, embedding.shape[1]]))
        return embedding, input_len

    def _prefill(self, embedding: tvm.nd.NDArray, input_len: int):
        print("======================= Starts Prefill =======================")
        seq_len_shape = ShapeTuple([input_len])
        max_num_sequence = 1
        page_size = 16
        sliding_window_size = (
            self.chat_config.sliding_window_size
            if self.chat_config.sliding_window_size
            else self.metadata["sliding_window_size"]
        )
        context_window_size = (
            self.chat_config.context_window_size
            if self.chat_config.context_window_size
            else self.metadata["context_window_size"]
        )
        prefill_chunk_size = (
            self.chat_config.prefill_chunk_size
            if self.chat_config.prefill_chunk_size
            else self.metadata["prefill_chunk_size"]
        )
        max_total_sequence_length = (
            sliding_window_size if context_window_size == -1 else context_window_size
        )
        support_sliding_window = int(sliding_window_size != -1)

        kv_caches = self.create_kv_cache_func(
            ShapeTuple([max_num_sequence]),
            ShapeTuple([max_total_sequence_length]),
            ShapeTuple([prefill_chunk_size]),
            ShapeTuple([page_size]),
            ShapeTuple([support_sliding_window]),
        )
        self.add_sequence_func(kv_caches, 0)
        self.begin_forward_func(kv_caches, ShapeTuple([0]), seq_len_shape)
        logits, kv_caches = self.prefill_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits, kv_caches

    def generate(self, prompt):
        input_tokens = prompt
        # print(f"{green('Input tokens')}: {input_tokens.numpy()}")
        embedding, input_len = self._embed(input_tokens)
        logits, kv_caches = self._prefill(embedding, input_len)
        return logits


def main():
    """The main function to start a DebugChat CLI"""

    parser = ArgumentParser("MLC LLM Perplexity Tool")
    parser.add_argument(
        "--model",
        type=str,
        help="An MLC model directory that contains `mlc-chat-config.json`",
        required=True,
    )
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to dumped input text file",
        required=True,
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        help="The full path to the model library file to use (e.g. a ``.so`` file).",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_compile"] + ' (default: "%(default)s")',
    )
    parser.add_argument("--prompt", type=str, help="Prompt to calculate perplexity", required=False)
    parsed = parser.parse_args()
    dc = Perplexity(model=parsed.model, model_lib=parsed.model_lib, device=parsed.device)
    perp_score_list = []

    if parsed.prompt is not None:
        input_tokens = dc._tokenize(parsed.prompt)
        prefill_logits = dc.generate(input_tokens)
        perp_score = calculate_perplexity(
            (prefill_logits.numpy()), np.reshape(input_tokens.numpy(), (1, input_tokens.shape[0]))
        )
        perp_score_list.append(perp_score)
    else:
        """Define the path here after generating dataset from create_input_file_wikitext.py"""
        input_tokens = load_inputs_from_file(parsed.input_path)

        for perp_vectors in input_tokens:
            values = list(map(int, perp_vectors.strip().split()))
            np_array = np.array(values, dtype="int32")
            tvm_nd_array = tvm.nd.array(np_array, tvm.cuda(0))
            prefill_logits = dc.generate(tvm_nd_array)
            perp_score = calculate_perplexity(
                (prefill_logits.numpy()), np.reshape(np_array, (1, np_array.shape[0]))
            )
            print(perp_score)
            perp_score_list.append(perp_score)

    return perp_score_list


if __name__ == "__main__":
    perp_score_list = main()
    ppl_mlc = torch.exp(torch.stack(perp_score_list).mean())
    print("Perplexity score: ", ppl_mlc)

"""
python3 calculate_perplexity.py --model MLC_Converted_weights --model-lib MLC_generated_lib --device cuda
                                --input-path PATH_OF_INPUT_TEXT_FILE

To run for some particular prompt other than wikitext

--prompt "Enter your text"
"""
