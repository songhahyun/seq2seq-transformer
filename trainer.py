import torch
import torch.nn as nn


def create_loss_fn(pad_id: int):
    return nn.CrossEntropyLoss(ignore_index=pad_id)


def shift_tgt_for_teacher_forcing(tgt_ids):
    # tgt_ids: [B, T]
    tgt_input = tgt_ids[:, :-1]
    tgt_output = tgt_ids[:, 1:]
    return tgt_input, tgt_output


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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
            tgt_output.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
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
            tgt_output.reshape(-1)
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)