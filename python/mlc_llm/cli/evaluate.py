"""Command line entrypoint of chat."""

from mlc_llm.interface.evaluate import ModelConfigOverride, chat
from mlc_llm.interface.help import HELP
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.chat`."""
    parser = ArgumentParser("MLC LLM Chat CLI")

    parser.add_argument("model", type=str, help=HELP["model"] + " (required)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib", type=str, default=None, help=HELP["model_lib"] + ' (default: "%(default)s")'
    )
    parser.add_argument(
        "--maths500",
        action="store_true",
        help="If set, runs the model on maths500 dataset instead of interactive chat",
    )
    parser.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to the dataset text file you want to evaluate",
    )
    parser.add_argument(
        "--overrides",
        type=ModelConfigOverride.from_str,
        default="",
        help=HELP["modelconfig_overrides"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    chat(
        model=parsed.model,
        device=parsed.device,
        model_lib=parsed.model_lib,
        overrides=parsed.overrides,
        maths500=parsed.maths500,
        eval_data_path=parsed.eval_data_path,
    )
