import torch
from sacrebleu import corpus_bleu, corpus_chrf

from data_pipeline import decode_ids


@torch.no_grad()
def generate_predictions(model, dataloader, sp_tgt, config, device):
    model.eval()

    predictions = []
    references = []
    source_texts = []

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_texts = batch["tgt_texts"]
        src_text_batch = batch["src_texts"]

        pred_ids = model.greedy_decode(
            src=src_ids,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            max_len=config.max_decode_len,
        )

        pred_ids = pred_ids.cpu().tolist()

        decoded_preds = [
            decode_ids(
                sp_tgt,
                ids,
                bos_id=config.bos_id,
                eos_id=config.eos_id,
                pad_id=config.pad_id,
            )
            for ids in pred_ids
        ]

        predictions.extend(decoded_preds)
        references.extend(tgt_texts)
        source_texts.extend(src_text_batch)

    return source_texts, predictions, references


def compute_bleu(predictions, references):
    # sacrebleu expects list[str], refs=list[list[str]]
    return corpus_bleu(predictions, [references]).score


def compute_chrf(predictions, references):
    return corpus_chrf(predictions, [references]).score


def print_sample_translations(source_texts, predictions, references, n=10):
    print("\n" + "=" * 80)
    print(f"[Sample Translations: {n}]")
    print("=" * 80)

    n = min(n, len(predictions))
    for i in range(n):
        print(f"\n[{i+1}]")
        print(f"SRC : {source_texts[i]}")
        print(f"REF : {references[i]}")
        print(f"PRED: {predictions[i]}")