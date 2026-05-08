import argparse
import os

import torch.optim as optim

from src.config import Config
from src.data_pipeline import (
    create_dataloaders,
    extract_pairs,
    load_ko_en_dataset,
    maybe_take_subset,
    prepare_tokenizers,
    set_seed,
    split_pairs,
)
from src.evaluate import generate_predictions, print_sample_translations
from src.metrics import compute_bleu, compute_chrf
from src.model_utils import build_model
from src.train import (
    create_loss_fn,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


def import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb`."
        ) from exc
    return wandb


def config_for_wandb(config):
    config_dict = {
        key: value
        for key, value in vars(config).items()
        if key != "hf_token"
    }
    config_dict.update(
        {
            "sp_model_path_src": config.sp_model_path_src,
            "sp_vocab_path_src": config.sp_vocab_path_src,
            "sp_model_path_tgt": config.sp_model_path_tgt,
            "sp_vocab_path_tgt": config.sp_vocab_path_tgt,
        }
    )
    return config_dict


def log_checkpoint_artifacts(wandb, config, run_name: str | None):
    artifact_name = run_name or "seq2seq-transformer-checkpoints"
    artifact = wandb.Artifact(artifact_name, type="model")

    for filename in ("best.pt", "latest.pt"):
        path = os.path.join(config.checkpoint_dir, filename)
        if os.path.exists(path):
            artifact.add_file(path)

    wandb.log_artifact(artifact)


def run_wandb_experiment(args):
    wandb = import_wandb()
    config = Config()
    set_seed(config.random_seed)
    if args.decode_strategy is not None:
        config.decode_strategy = args.decode_strategy
    if args.beam_size is not None:
        config.beam_size = args.beam_size

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config_for_wandb(config),
    )

    print(f"[INFO] device = {config.device}")
    print("[INFO] loading dataset...")

    dataset = load_ko_en_dataset(
        config.dataset_name,
        split=config.train_split,
        hf_token=config.hf_token,
    )
    pairs = extract_pairs(dataset, src_col="ko", tgt_col="en")

    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.random_seed,
    )

    train_pairs = maybe_take_subset(train_pairs, config.train_subset_size)
    valid_pairs = maybe_take_subset(valid_pairs, config.valid_subset_size)
    test_pairs = maybe_take_subset(test_pairs, config.test_subset_size)

    wandb.log(
        {
            "data/total_pairs": len(pairs),
            "data/train_pairs": len(train_pairs),
            "data/valid_pairs": len(valid_pairs),
            "data/test_pairs": len(test_pairs),
        }
    )

    print("[INFO] training/loading sentencepiece tokenizers...")
    sp_src, sp_tgt = prepare_tokenizers(train_pairs, config)

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_pairs,
        valid_pairs,
        test_pairs,
        sp_src,
        sp_tgt,
        config,
    )

    model = build_model(
        config=config,
        src_vocab_size=sp_src.get_piece_size(),
        tgt_vocab_size=sp_tgt.get_piece_size(),
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = create_loss_fn(config.pad_id)

    wandb.watch(model, log=args.watch_log, log_freq=args.watch_log_freq)
    wandb.config.update(
        {
            "src_actual_vocab_size": sp_src.get_piece_size(),
            "tgt_actual_vocab_size": sp_tgt.get_piece_size(),
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
        allow_val_change=True,
    )

    print("[INFO] training start...")
    best_valid_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
        )
        valid_loss = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            config.device,
        )

        print(
            f"[Epoch {epoch + 1}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
            },
            step=epoch + 1,
        )

        latest_checkpoint_path = f"{config.checkpoint_dir}/latest.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch + 1,
            train_loss=train_loss,
            valid_loss=valid_loss,
            src_vocab_size=sp_src.get_piece_size(),
            tgt_vocab_size=sp_tgt.get_piece_size(),
            path=latest_checkpoint_path,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_checkpoint_path = f"{config.checkpoint_dir}/best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch + 1,
                train_loss=train_loss,
                valid_loss=valid_loss,
                src_vocab_size=sp_src.get_piece_size(),
                tgt_vocab_size=sp_tgt.get_piece_size(),
                path=best_checkpoint_path,
            )
            wandb.run.summary["best_valid_loss"] = best_valid_loss
            wandb.run.summary["best_epoch"] = epoch + 1

    print("[INFO] evaluating on test set...")
    source_texts, predictions, references = generate_predictions(
        model=model,
        dataloader=test_loader,
        sp_tgt=sp_tgt,
        config=config,
        device=config.device,
    )

    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)

    wandb.log(
        {
            "test/bleu": bleu,
            "test/chrf": chrf,
        }
    )
    wandb.run.summary["test_bleu"] = bleu
    wandb.run.summary["test_chrf"] = chrf

    sample_table = wandb.Table(columns=["source", "reference", "prediction"])
    sample_count = min(args.num_samples, len(predictions))
    for i in range(sample_count):
        sample_table.add_data(source_texts[i], references[i], predictions[i])
    wandb.log({"samples/translations": sample_table})

    print(f"[TEST] BLEU : {bleu:.4f}")
    print(f"[TEST] chrF : {chrf:.4f}")
    print_sample_translations(
        source_texts=source_texts,
        predictions=predictions,
        references=references,
        n=sample_count,
    )

    if not args.no_artifact:
        log_checkpoint_artifacts(wandb, config, run.name)

    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a W&B tracked experiment.")
    parser.add_argument(
        "--project",
        type=str,
        default="seq2seq-transformer",
        help="W&B project name.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run W&B in offline mode.",
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="Do not upload checkpoint artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample translations to log.",
    )
    parser.add_argument(
        "--watch-log",
        type=str,
        default=None,
        choices=["gradients", "parameters", "all"],
        help="Data to log with wandb.watch.",
    )
    parser.add_argument(
        "--watch-log-freq",
        type=int,
        default=100,
        help="Logging frequency for wandb.watch.",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["greedy", "beam"],
        default=None,
        help="Decoding strategy for test evaluation. Defaults to config value.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size when --decode-strategy beam is used. Defaults to config value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_wandb_experiment(args)


if __name__ == "__main__":
    main()
