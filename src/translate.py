import torch

from src.data_pipeline import decode_ids, encode_text
from src.model_utils import (
    load_checkpoint,
    load_model_from_checkpoint,
    load_tokenizers,
    resolve_checkpoint_path,
)


def run_translate(config, text: str, checkpoint_path: str | None = None):
    """Load a trained model and translate one input text string."""
    sp_src, sp_tgt = load_tokenizers(config)
    resolved_checkpoint_path = resolve_checkpoint_path(config, checkpoint_path)

    print(f"[INFO] loading checkpoint: {resolved_checkpoint_path}")
    checkpoint = load_checkpoint(resolved_checkpoint_path, config.device)
    model = load_model_from_checkpoint(config, checkpoint, sp_src, sp_tgt)

    src_ids = encode_text(
        sp_src,
        text,
        config.max_length,
        config.bos_id,
        config.eos_id,
    )
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=config.device)

    with torch.no_grad():
        pred_ids = model.decode(
            src=src_tensor,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            max_len=config.max_decode_len,
            strategy=config.decode_strategy,
            beam_size=config.beam_size,
        )

    translation = decode_ids(
        sp_tgt,
        pred_ids[0].cpu().tolist(),
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        pad_id=config.pad_id,
    )
    print(translation)
    return translation
