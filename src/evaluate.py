import torch

from src.data_pipeline import (
    create_dataloaders,
    decode_ids,
    extract_pairs,
    load_ko_en_dataset,
    maybe_take_subset,
    split_pairs,
)
from src.metrics import compute_bleu, compute_chrf
from src.model_utils import (
    load_checkpoint,
    load_model_from_checkpoint,
    load_tokenizers,
    resolve_checkpoint_path,
)


@torch.no_grad()
def generate_predictions(model, dataloader, sp_tgt, config, device):
    """Generate decoded predictions, references, and source texts for a dataloader."""
    model.eval()

    predictions = []
    references = []
    source_texts = []

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_texts = batch["tgt_texts"]
        src_text_batch = batch["src_texts"]

        pred_ids = model.decode(
            src=src_ids,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            max_len=config.max_decode_len,
            strategy=config.decode_strategy,
            beam_size=config.beam_size,
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


def print_sample_translations(source_texts, predictions, references, n=10):
    """Print a small aligned sample of source, reference, and prediction text."""
    print("\n" + "=" * 80)
    print(f"[Sample Translations: {n}]")
    print("=" * 80)

    n = min(n, len(predictions))
    for i in range(n):
        print(f"\n[{i+1}]")
        print(f"SRC : {source_texts[i]}")
        print(f"REF : {references[i]}")
        print(f"PRED: {predictions[i]}")


def run_evaluate(config, checkpoint_path: str | None = None):
    """Load a checkpoint, evaluate on the test split, and report metrics."""
    sp_src, sp_tgt = load_tokenizers(config)
    resolved_checkpoint_path = resolve_checkpoint_path(config, checkpoint_path)

    print(f"[INFO] loading checkpoint: {resolved_checkpoint_path}")
    checkpoint = load_checkpoint(resolved_checkpoint_path, config.device)
    model = load_model_from_checkpoint(config, checkpoint, sp_src, sp_tgt)

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

    _, _, test_loader = create_dataloaders(
        train_pairs, valid_pairs, test_pairs, sp_src, sp_tgt, config
    )

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

    print(f"[TEST] BLEU : {bleu:.4f}")
    print(f"[TEST] chrF : {chrf:.4f}")

    print_sample_translations(
        source_texts=source_texts,
        predictions=predictions,
        references=references,
        n=config.num_examples_for_samples,
    )

    return {
        "bleu": bleu,
        "chrf": chrf,
        "source_texts": source_texts,
        "predictions": predictions,
        "references": references,
    }
