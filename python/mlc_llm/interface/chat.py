"""Python entrypoint of chat."""

import dataclasses
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from prompt_toolkit import prompt as get_prompt  # pylint: disable=import-error
from prompt_toolkit.key_binding import KeyBindings  # pylint: disable=import-error

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.engine import MLCEngine
from mlc_llm.serve.engine_base import _query_engine_metrics
from mlc_llm.support import argparse
from mlc_llm.support.config import ConfigOverrideBase


def _print_help_str():
    help_str = """You can use the following special commands:
  /help                    print the special commands
  /exit                    quit the cli
  /stats                   print out stats of last request (tok/s and exact token counts)
  /metrics                 print out full engine metrics
  /reset                   restart a fresh chat
  /random-tokens <N>       generate N random tokens and run the model
  /set [overrides]         override settings in the generation config. For example,
                           `/set temperature=0.5;top_p=0.8;seed=23;max_tokens=100;stop=str1,str2`
                           Note: Separate stop words in the `stop` option with commas (,).
  Multi-line input: Use escape+enter to start a new line.
"""
    print(help_str)


def _get_vocab_size(engine) -> int:
    """Try to get vocab size from the engine's model directory.

    Reads the HuggingFace ``tokenizer.json`` vocab table when available.
    Falls back to 32000 (common LLaMA-style default) if the file is absent
    or cannot be parsed.
    """
    model_path = None
    if hasattr(engine, "engine_config") and hasattr(engine.engine_config, "model"):
        model_path = engine.engine_config.model

    if model_path is not None:
        tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            try:
                with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                    tokenizer_data = json.load(f)
                vocab = tokenizer_data.get("model", {}).get("vocab", {})
                if vocab:
                    return len(vocab)
            except Exception:  # pylint: disable=broad-except
                pass

    # Default fallback for most common models (LLaMA / Qwen style)
    return 32000


def _set_up_key_bindings():
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    return kb


@dataclasses.dataclass
class ChatCompletionOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding chat completions."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

    @staticmethod
    def from_str(source: str) -> "ChatCompletionOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="chat completion override values")
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--frequency_penalty", type=float, default=None)
        parser.add_argument("--presence_penalty", type=float, default=None)
        parser.add_argument("--max_tokens", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--stop", type=str, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ChatCompletionOverride(
            temperature=results.temperature,
            top_p=results.top_p,
            frequency_penalty=results.frequency_penalty,
            presence_penalty=results.presence_penalty,
            max_tokens=results.max_tokens,
            seed=results.seed,
            stop=results.stop.split(",") if results.stop is not None else None,
        )


@dataclasses.dataclass
class ModelConfigOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None
    opt: Optional[str] = None

    @staticmethod
    def from_str(source: str) -> "ModelConfigOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="model config override values")
        parser.add_argument("--tensor_parallel_shards", type=int, default=None)
        parser.add_argument("--pipeline_parallel_stages", type=int, default=None)
        parser.add_argument("--opt", type=str, default=None)
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--sliding_window_size", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--attention_sink_size", type=int, default=None)

        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ModelConfigOverride(
            tensor_parallel_shards=results.tensor_parallel_shards,
            pipeline_parallel_stages=results.pipeline_parallel_stages,
            opt=results.opt,
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            prefill_chunk_size=results.prefill_chunk_size,
            attention_sink_size=results.attention_sink_size,
        )


class ChatState:
    """Simple helper class to manage chat state.

    Chat state wraps around a  engine instance
    and exposes the minimum set of tools to perform
    interactive chat. It provides support for mlc_llm chat.
    It also can be used to do interactive debugging
    with different engine instance.

    Examples
    --------
    .. code:: python

        from openai import OpenAI
        from mlc_llm import MLCEngine
        from mlc_llm.serve import PopenServer
        from mlc_llm.interface.chat import ChatState

        def chat_with_engine(model):
            # hookup with MLCEngine
            ChatState(MLCEngine(model)).chat()

        def chat_with_server(model):
            # hookup with AsyncMLCEngine backed api server
            with PopenServer(model) as server:
                ChatState(
                    OpenAI(base_url=server.openai_v1_base_url, api_key="None")
                ).chat()
    """

    history: List[Dict[str, Any]]
    history_begin: int
    # kwargs passed to completions
    overrides: ChatCompletionOverride
    # Underlying engine
    engine: Union[JSONFFIEngine, MLCEngine]
    last_finished_request_usage: Optional[openai_api_protocol.CompletionUsage]

    def __init__(self, engine: Union[JSONFFIEngine, MLCEngine]):
        self.engine = engine
        self.history = []
        self.history_window_begin = 0
        self.overrides = ChatCompletionOverride()
        # model is mainly used for compact reasons
        self.model = "chat_model"
        self.last_finished_request_usage = None

    def slide_history(self):
        """Slide history to fit into context window"""
        history_window_size = len(self.history) - self.history_window_begin
        assert history_window_size % 2 == 0
        self.history_window_begin += ((history_window_size + 3) // 4) * 2

    def process_system_prompts(self):
        """Process system prompts"""
        # TODO(mlc-team): possibly leverage debug option
        # pass a simple prompt to warm up
        for _ in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": ""}],
            max_tokens=1,
            model=self.model,
            stream=True,
        ):
            pass

    def generate(self, prompt: str):
        """Run one generation with the prompt.

        Parameters
        ----------
        prompt: str
            The input prompt
        """
        self.history.append({"role": "user", "content": prompt})
        output_text = ""
        finish_reason_length = False
        messages = self.history[self.history_window_begin :]

        for response in self.engine.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            stream_options={"include_usage": True},
            **dataclasses.asdict(self.overrides),
        ):
            if response.usage is not None:
                self.last_finished_request_usage = response.usage
                continue
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                if isinstance(choice.delta.content, str):
                    output_text += choice.delta.content
                    print(choice.delta.content, end="", flush=True)
                if choice.finish_reason == "length":
                    finish_reason_length = True
        if finish_reason_length:
            print(" [output truncated due to context length limit...]")
        # print additional \n when generation ends
        print()
        # record the history
        self.history.append({"role": "assistant", "content": output_text})
        if finish_reason_length:
            self.slide_history()

    def _get_template_overhead(self) -> int:
        """Measure and cache the number of tokens the chat template adds.

        Runs a single calibration request with a 1-token prompt (the letter "a")
        against an empty history to determine how many tokens the template
        structure contributes (system prompt, role markers, etc.).  The result
        is cached so subsequent calls are free.

        Returns
        -------
        overhead : int
            Number of template tokens added on top of the raw user content.
        """
        if hasattr(self, "_template_overhead_cache"):
            return self._template_overhead_cache

        if not hasattr(self.engine, "tokenizer"):
            self._template_overhead_cache = 0
            return 0

        # "a" encodes to a single token in virtually every tokenizer.
        calibration_text = "a"
        content_token_count = len(self.engine.tokenizer.encode(calibration_text))

        # Save current usage so calibration does not overwrite it.
        saved_usage = self.last_finished_request_usage

        overhead = 0
        for response in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": calibration_text}],
            max_tokens=1,
            model=self.model,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if response.usage is not None:
                overhead = max(0, response.usage.prompt_tokens - content_token_count)
                break

        self._template_overhead_cache = overhead
        # Restore usage so the calibration run is invisible to /stats.
        self.last_finished_request_usage = saved_usage
        return overhead

    def generate_random_tokens(self, token_size: int, max_decode_tokens: Optional[int] = None):
        """Generate random tokens targeting *token_size* total prompt tokens.

        Automatically compensates for chat-template overhead (system prompt,
        role markers, etc.) so that the prompt-token count reported by ``/stats``
        matches *token_size* as closely as possible.

        The token IDs are sampled uniformly from ``[1, vocab_size - 1]`` (token 0
        is skipped as it is typically the padding / BOS token).  Up to three
        decode-encode adjustment iterations are performed to correct for
        tokenizer round-trip drift.

        Parameters
        ----------
        token_size : int
            Target total prompt token count (including template overhead).
        max_decode_tokens : Optional[int]
            Maximum tokens to generate in the decode step.  When ``None`` the
            current override value (or the model default) is used.  The override
            is restored after the call so it does not affect subsequent prompts.
        """
        vocab_size = _get_vocab_size(self.engine)
        template_overhead = self._get_template_overhead()

        # How many random content tokens we need to fill the target.
        content_needed = max(1, token_size - template_overhead)

        if hasattr(self.engine, "tokenizer"):
            token_ids = [random.randint(1, vocab_size - 1) for _ in range(content_needed)]
            prompt = self.engine.tokenizer.decode(token_ids)

            # Iteratively correct for decode/encode drift (at most 3 passes).
            for _ in range(3):
                actual_content = len(self.engine.tokenizer.encode(prompt))
                if actual_content == content_needed:
                    break
                if actual_content < content_needed:
                    extra = [
                        random.randint(1, vocab_size - 1)
                        for _ in range(content_needed - actual_content)
                    ]
                    token_ids = self.engine.tokenizer.encode(prompt) + extra
                else:
                    token_ids = self.engine.tokenizer.encode(prompt)[:content_needed]
                prompt = self.engine.tokenizer.decode(token_ids)

            actual_content = len(self.engine.tokenizer.encode(prompt))
        else:
            # Fallback when no tokenizer is directly accessible.
            prompt = " ".join(
                str(t) for t in [random.randint(1, vocab_size - 1) for _ in range(content_needed)]
            )
            actual_content = content_needed

        expected_total = actual_content + template_overhead
        print(
            f"[random-tokens] target={token_size}, "
            f"template_overhead={template_overhead}, "
            f"content_tokens={actual_content}, "
            f"expected_total~{expected_total}"
        )

        # Temporarily apply max_decode_tokens if requested.
        saved_max_tokens = self.overrides.max_tokens
        if max_decode_tokens is not None:
            self.overrides.max_tokens = max_decode_tokens
        try:
            self.generate(prompt)
        finally:
            self.overrides.max_tokens = saved_max_tokens

    def stats(self):
        """Print statistics of the prefill/decode speed and exact token counts."""

        def get_stats_text():
            """Build the stats string."""
            if self.last_finished_request_usage is None:
                return "N/A"
            usage = self.last_finished_request_usage
            extra = usage.extra

            # Speed metrics (may be absent when extra is None)
            if extra is not None:
                prefill_speed_val = extra.get("prefill_tokens_per_s", None)
                decode_speed_val = extra.get("decode_tokens_per_s", None)
                prefill_speed = (
                    f"{prefill_speed_val:.1f}" if prefill_speed_val is not None else "N/A"
                )
                decode_speed = f"{decode_speed_val:.1f}" if decode_speed_val is not None else "N/A"
            else:
                prefill_speed = "N/A"
                decode_speed = "N/A"

            # Exact token counts from CompletionUsage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            return (
                f"prefill: {prefill_speed} tok/s, decode: {decode_speed} tok/s\n"
                f"prompt tokens: {prompt_tokens}, "
                f"completion tokens: {completion_tokens}, "
                f"total tokens: {total_tokens}"
            )

        print(get_stats_text(), flush=True)

    def metrics(self):
        """Print metrics as prometheus text"""
        print(_query_engine_metrics(self.engine).prometheus_text(), flush=True)

    def reset(self):
        """Reset the chat history"""
        self.history = []
        self.history_window_begin = 0

    def chat(
        self,
        random_tokens: Optional[int] = None,
        max_decode_tokens: Optional[int] = None,
    ):
        """Start an interactive chat session.

        Parameters
        ----------
        random_tokens : Optional[int]
            When provided, automatically generate this many random tokens and
            run the model once before entering the interactive prompt loop.
            Useful for quick benchmarking without typing a prompt manually.
        max_decode_tokens : Optional[int]
            Maximum decode tokens for the automatic ``random_tokens`` run.
            Has no effect when ``random_tokens`` is ``None``.
        """
        _print_help_str()

        self.process_system_prompts()  # pylint: disable=protected-access

        # If --random-tokens was supplied on the CLI, run it immediately.
        if random_tokens is not None and random_tokens > 0:
            print(f"[random-tokens] Auto-running with {random_tokens} random tokens...")
            self.generate_random_tokens(random_tokens, max_decode_tokens=max_decode_tokens)
            self.stats()

        # Multi-line input support: set escape+enter as start a new line
        kb = _set_up_key_bindings()

        while True:
            prompt = get_prompt(
                ">>> ",  # pylint: disable=protected-access
                key_bindings=kb,
                multiline=True,
            )
            if prompt[:4] == "/set":
                overrides = ChatCompletionOverride.from_str(prompt.split()[1])
                for key, value in dataclasses.asdict(overrides).items():
                    if value is not None:
                        setattr(self.overrides, key, value)
            elif prompt[:6] == "/stats":
                self.stats()
            elif prompt[:8] == "/metrics":
                self.metrics()
            elif prompt[:6] == "/reset":
                self.reset()
            elif prompt[:5] == "/exit":
                self.engine.terminate()
                break
            elif prompt[:5] == "/help":
                _print_help_str()
            elif prompt[:14] == "/random-tokens":
                # Syntax: /random-tokens <N> [max_decode=M]
                # e.g.  : /random-tokens 512
                #         /random-tokens 512 128
                parts = prompt.split()
                if len(parts) < 2:
                    print(
                        "Usage: /random-tokens <N> [max_decode_tokens]\n"
                        "  N                 - target prompt token count\n"
                        "  max_decode_tokens - optional decode token limit"
                    )
                else:
                    try:
                        n = int(parts[1])
                        if n <= 0:
                            print("Token size must be a positive integer.")
                        else:
                            mdt = None
                            if len(parts) >= 3:
                                mdt = int(parts[2])
                            self.generate_random_tokens(n, max_decode_tokens=mdt)
                    except ValueError:
                        print(
                            f"Invalid argument in '/random-tokens {' '.join(parts[1:])}'. "
                            "Both N and max_decode_tokens must be positive integers."
                        )
            else:
                self.generate(prompt)


def chat(
    model: str,
    device: str,
    model_lib: Optional[str],
    overrides: ModelConfigOverride,
    random_tokens: Optional[int] = None,
    max_decode_tokens: Optional[int] = None,
):
    """Chat cli entry"""
    # By default we use JSONFFIEngine
    ChatState(
        JSONFFIEngine(
            model,
            device,
            model_lib=model_lib,
            mode="interactive",
            engine_config=EngineConfig(
                max_single_sequence_length=overrides.context_window_size,
                prefill_chunk_size=overrides.prefill_chunk_size,
                sliding_window_size=overrides.sliding_window_size,
                attention_sink_size=overrides.attention_sink_size,
                tensor_parallel_shards=overrides.tensor_parallel_shards,
                pipeline_parallel_stages=overrides.pipeline_parallel_stages,
                opt=overrides.opt,
            ),
        )
    ).chat(random_tokens=random_tokens, max_decode_tokens=max_decode_tokens)
