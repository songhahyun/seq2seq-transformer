import torch
import torch.optim as optim

from config import Config
from data_pipeline import (
    set_seed,
    load_ko_en_dataset,
    extract_pairs,
    split_pairs,
    maybe_take_subset,
    prepare_tokenizers,
    create_dataloaders,
)
from transformer_model import Seq2SeqTransformer
from trainer import create_loss_fn, train_one_epoch, validate_one_epoch
from evaluator import (
    generate_predictions,
    compute_bleu,
    compute_chrf,
    print_sample_translations,
)


def main():
    config = Config()
    set_seed(config.random_seed)

    print(f"[INFO] device = {config.device}")
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

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_pairs, valid_pairs, test_pairs, sp_src, sp_tgt, config
    )

    print("[INFO] building model...")
    model = Seq2SeqTransformer(
        src_vocab_size=sp_src.get_piece_size(),
        tgt_vocab_size=sp_tgt.get_piece_size(),
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        pad_id=config.pad_id,
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = create_loss_fn(config.pad_id)

    print("[INFO] training start...")
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        valid_loss = validate_one_epoch(
            model, valid_loader, criterion, config.device
        )

        print(
            f"[Epoch {epoch+1}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}"
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


if __name__ == "__main__":
    main()