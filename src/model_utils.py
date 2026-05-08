import os

import sentencepiece as spm
import torch

from src.transformer_model import Seq2SeqTransformer


def build_model(config, src_vocab_size: int, tgt_vocab_size: int):
    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        pad_id=config.pad_id,
    ).to(config.device)


def load_tokenizers(config):
    if not os.path.exists(config.sp_model_path_src):
        raise FileNotFoundError(f"source tokenizer not found: {config.sp_model_path_src}")
    if not os.path.exists(config.sp_model_path_tgt):
        raise FileNotFoundError(f"target tokenizer not found: {config.sp_model_path_tgt}")

    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()
    sp_src.load(config.sp_model_path_src)
    sp_tgt.load(config.sp_model_path_tgt)
    return sp_src, sp_tgt


def resolve_checkpoint_path(config, checkpoint_path: str | None = None):
    if checkpoint_path:
        return checkpoint_path

    best_path = f"{config.checkpoint_dir}/best.pt"
    latest_path = f"{config.checkpoint_dir}/latest.pt"
    if os.path.exists(best_path):
        return best_path
    return latest_path


def load_checkpoint(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(config, checkpoint, sp_src, sp_tgt):
    model = build_model(
        config=config,
        src_vocab_size=checkpoint.get("src_vocab_size", sp_src.get_piece_size()),
        tgt_vocab_size=checkpoint.get("tgt_vocab_size", sp_tgt.get_piece_size()),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
