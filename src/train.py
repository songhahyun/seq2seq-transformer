import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from src.data_pipeline import (
    create_dataloaders,
    extract_pairs,
    load_ko_en_dataset,
    maybe_take_subset,
    prepare_tokenizers,
    split_pairs,
)
from src.model_utils import build_model


def create_loss_fn(pad_id: int):
    """Create cross-entropy loss that ignores PAD tokens."""
    return nn.CrossEntropyLoss(ignore_index=pad_id)


def create_lr_scheduler(optimizer, config, total_training_steps: int):
    """Create a linear warmup and linear decay learning-rate scheduler."""
    if not config.use_lr_scheduler:
        return None

    if config.lr_scheduler_type != "linear":
        raise ValueError(f"unsupported lr_scheduler_type: {config.lr_scheduler_type}")

    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be greater than 0")

    warmup_steps = min(config.warmup_steps, total_training_steps)
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(total_training_steps - warmup_steps, 1)
        steps_after_warmup = min(max(current_step - warmup_steps, 0), decay_steps)
        decay_ratio = 1.0 - (steps_after_warmup / decay_steps)
        return max(min_lr_ratio, decay_ratio)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    model,
    optimizer,
    config,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    src_vocab_size: int,
    tgt_vocab_size: int,
    path: str,
    scheduler=None,
):
    """Save model, optimizer, metrics, vocabulary sizes, and config state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config_state = {
        key: value
        for key, value in vars(config).items()
        if key != "hf_token"
    }
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "config": config_state,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def shift_tgt_for_teacher_forcing(tgt_ids):
    """Split target ids into decoder input and next-token labels."""
    # tgt_ids: [B, T]
    tgt_input = tgt_ids[:, :-1]
    tgt_output = tgt_ids[:, 1:]
    return tgt_input, tgt_output


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train the model for one epoch and return average batch loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        tgt_input, tgt_output = shift_tgt_for_teacher_forcing(tgt_ids)

        optimizer.zero_grad()
        logits = model(src_ids, tgt_input)  # [B, T, vocab]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """Evaluate one validation epoch without gradient updates."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        tgt_input, tgt_output = shift_tgt_for_teacher_forcing(tgt_ids)

        logits = model(src_ids, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def run_train(config):
    """Run the full training pipeline from data loading to checkpointing."""
    print("[INFO] loading dataset...")
    dataset = load_ko_en_dataset(
        config.dataset_name,
        split=config.train_split,
        hf_token=config.hf_token,
    )
    pairs = extract_pairs(dataset, src_col="ko", tgt_col="en")
    print(f"[INFO] total pairs = {len(pairs)}")

    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.random_seed,
    )

    train_pairs = maybe_take_subset(train_pairs, config.train_subset_size)
    valid_pairs = maybe_take_subset(valid_pairs, config.valid_subset_size)
    test_pairs = maybe_take_subset(test_pairs, config.test_subset_size)

    print(f"[INFO] train pairs = {len(train_pairs)}")
    print(f"[INFO] valid pairs = {len(valid_pairs)}")
    print(f"[INFO] test pairs  = {len(test_pairs)}")

    print("[INFO] training/loading sentencepiece tokenizers...")
    sp_src, sp_tgt = prepare_tokenizers(train_pairs, config)

    train_loader, valid_loader, _ = create_dataloaders(
        train_pairs, valid_pairs, test_pairs, sp_src, sp_tgt, config
    )

    print("[INFO] building model...")
    model = build_model(
        config=config,
        src_vocab_size=sp_src.get_piece_size(),
        tgt_vocab_size=sp_tgt.get_piece_size(),
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    total_training_steps = len(train_loader) * config.num_epochs
    scheduler = create_lr_scheduler(optimizer, config, total_training_steps)
    criterion = create_loss_fn(config.pad_id)

    if scheduler is not None:
        print(
            "[INFO] lr scheduler = "
            f"{config.lr_scheduler_type} warmup_steps={config.warmup_steps} "
            f"total_steps={total_training_steps}"
        )

    print("[INFO] training start...")
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device, scheduler=scheduler
        )
        valid_loss = validate_one_epoch(
            model, valid_loader, criterion, config.device
        )
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch+1}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | "
            f"lr={current_lr:.8f}"
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
            scheduler=scheduler,
        )
        print(f"[INFO] saved checkpoint: {latest_checkpoint_path}")

        improved = valid_loss < best_valid_loss - config.early_stopping_min_delta
        if improved:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
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
                scheduler=scheduler,
            )
            print(f"[INFO] saved best checkpoint: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                "[INFO] no validation improvement "
                f"({epochs_without_improvement}/{config.early_stopping_patience})"
            )

            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "[INFO] early stopping triggered: "
                    f"best_valid_loss={best_valid_loss:.4f}"
                )
                break
