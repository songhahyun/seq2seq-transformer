import argparse

from src.config import Config
from src.data_pipeline import set_seed
from src.evaluate import run_evaluate
from src.train import run_train
from src.translate import run_translate


def parse_args():
    """Parse command-line arguments for train, evaluate, and translate modes."""
    parser = argparse.ArgumentParser(description="Train, evaluate, or translate.")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "translate"],
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to translate. Required when --mode translate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to checkpoints/best.pt when available.",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["greedy", "beam"],
        default=None,
        help="Decoding strategy for evaluate/translate. Defaults to config value.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size when --decode-strategy beam is used. Defaults to config value.",
    )
    return parser.parse_args()


def main():
    """Apply CLI overrides and dispatch to the selected execution mode."""
    args = parse_args()
    config = Config()
    set_seed(config.random_seed)
    if args.decode_strategy is not None:
        config.decode_strategy = args.decode_strategy
    if args.beam_size is not None:
        config.beam_size = args.beam_size

    print(f"[INFO] device = {config.device}")

    if args.mode == "train":
        run_train(config)
        return

    if args.mode == "evaluate":
        run_evaluate(config, checkpoint_path=args.checkpoint)
        return

    if args.text is None:
        raise ValueError("--text is required when --mode translate")
    run_translate(config, text=args.text, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
