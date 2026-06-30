import os
import random
import tempfile
from functools import partial
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm


def set_seed(seed: int):
    """Set Python and PyTorch random seeds for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ko_en_dataset(dataset_name: str, split: str = "train", hf_token: str | None = None):
    """Load a Korean-English dataset split from Hugging Face Datasets."""
    dataset = load_dataset(dataset_name, split=split, token=hf_token)
    return dataset

def extract_pairs(dataset, src_col: str = "ko", tgt_col: str = "en") -> List[Tuple[str, str]]:
    """Extract non-empty source and target text pairs from a dataset."""
    pairs = []
    for item in dataset:
        src = str(item[src_col]).strip()
        tgt = str(item[tgt_col]).strip()
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def split_pairs(pairs, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Shuffle and split pairs into train, validation, and test sets."""
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)

    test_pairs = pairs[:n_test]
    valid_pairs = pairs[n_test:n_test + n_valid]
    train_pairs = pairs[n_test + n_valid:]

    return train_pairs, valid_pairs, test_pairs


def maybe_take_subset(pairs, subset_size=None):
    """Return the leading subset of pairs when a subset size is configured."""
    if subset_size is None:
        return pairs
    return pairs[:subset_size]


def write_texts_for_spm(texts: List[str], output_path: str):
    """Write normalized text lines to a temporary SentencePiece input file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ").strip() + "\n")


def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
):
    """Train a SentencePiece tokenizer with fixed special token ids."""
    output_dir = os.path.dirname(model_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


def prepare_tokenizers(train_pairs, config):
    """Train missing tokenizers and load source and target SentencePiece models."""
    src_texts = [src for src, _ in train_pairs]
    tgt_texts = [tgt for _, tgt in train_pairs]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix="_src.txt", encoding="utf-8") as f_src:
        src_tmp = f_src.name
    with tempfile.NamedTemporaryFile("w", delete=False, suffix="_tgt.txt", encoding="utf-8") as f_tgt:
        tgt_tmp = f_tgt.name

    try:
        write_texts_for_spm(src_texts, src_tmp)
        write_texts_for_spm(tgt_texts, tgt_tmp)

        if not os.path.exists(config.sp_model_path_src):
            train_sentencepiece(
                input_file=src_tmp,
                model_prefix=config.sp_model_prefix_src,
                vocab_size=config.src_vocab_size,
                character_coverage=config.character_coverage_ko,
            )

        if not os.path.exists(config.sp_model_path_tgt):
            train_sentencepiece(
                input_file=tgt_tmp,
                model_prefix=config.sp_model_prefix_tgt,
                vocab_size=config.tgt_vocab_size,
                character_coverage=config.character_coverage_en,
            )
    finally:
        if os.path.exists(src_tmp):
            os.remove(src_tmp)
        if os.path.exists(tgt_tmp):
            os.remove(tgt_tmp)

    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()
    sp_src.load(config.sp_model_path_src)
    sp_tgt.load(config.sp_model_path_tgt)

    return sp_src, sp_tgt


def encode_text(sp_model, text: str, max_length: int, bos_id: int, eos_id: int):
    """Encode text with SentencePiece and wrap it with BOS/EOS tokens."""
    ids = sp_model.encode(text, out_type=int)
    ids = [bos_id] + ids + [eos_id]
    ids = ids[:max_length]
    return ids


def decode_ids(sp_model, ids, bos_id: int, eos_id: int, pad_id: int):
    """Decode token ids after removing BOS/PAD and stopping at EOS."""
    cleaned = []
    for i in ids:
        if i in (bos_id, pad_id):
            continue
        if i == eos_id:
            break
        cleaned.append(i)
    if not cleaned:
        return ""
    return sp_model.decode(cleaned)


class TranslationDataset(Dataset):
    """PyTorch dataset that stores translation pairs and encoded token sequences."""

    def __init__(self, pairs, sp_src, sp_tgt, config):
        """Store translation pairs, optionally pre-tokenized in memory."""
        self.pretokenize = getattr(config, "pretokenize_dataset", True)

        if self.pretokenize:
            self.examples = [
                self._encode_pair(src_text, tgt_text, sp_src, sp_tgt, config)
                for src_text, tgt_text in pairs
            ]
            self.pairs = None
            self.sp_src = None
            self.sp_tgt = None
            self.config = None
        else:
            self.examples = None
            self.pairs = pairs
            self.sp_src = sp_src
            self.sp_tgt = sp_tgt
            self.config = config

    @staticmethod
    def _encode_pair(src_text, tgt_text, sp_src, sp_tgt, config):
        """Encode one source-target pair and keep the original text."""
        src_ids = encode_text(
            sp_src,
            src_text,
            config.max_length,
            config.bos_id,
            config.eos_id,
        )
        tgt_ids = encode_text(
            sp_tgt,
            tgt_text,
            config.max_length,
            config.bos_id,
            config.eos_id,
        )

        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
        }

    def __len__(self):
        """Return the number of translation pairs in the dataset."""
        if self.examples is not None:
            return len(self.examples)
        return len(self.pairs)

    def __getitem__(self, idx):
        """Encode one source-target pair and return text plus token ids."""
        if self.examples is not None:
            return self.examples[idx]

        src_text, tgt_text = self.pairs[idx]
        return self._encode_pair(src_text, tgt_text, self.sp_src, self.sp_tgt, self.config)


def pad_sequence(sequence, max_len, pad_id):
    """Pad or truncate a token id sequence to the requested length."""
    if len(sequence) < max_len:
        sequence = sequence + [pad_id] * (max_len - len(sequence))
    return sequence[:max_len]


def collate_fn(batch, pad_id):
    """Pad a batch dynamically and convert token lists into tensors."""
    max_src_len = max(len(item["src_ids"]) for item in batch)
    max_tgt_len = max(len(item["tgt_ids"]) for item in batch)

    src_batch = []
    tgt_batch = []
    src_texts = []
    tgt_texts = []

    for item in batch:
        src_batch.append(pad_sequence(item["src_ids"], max_src_len, pad_id))
        tgt_batch.append(pad_sequence(item["tgt_ids"], max_tgt_len, pad_id))
        src_texts.append(item["src_text"])
        tgt_texts.append(item["tgt_text"])

    src_batch = torch.tensor(src_batch, dtype=torch.long)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.long)

    return {
        "src_ids": src_batch,
        "tgt_ids": tgt_batch,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
    }


def create_dataloaders(train_pairs, valid_pairs, test_pairs, sp_src, sp_tgt, config):
    """Create train, validation, and test dataloaders for translation pairs."""
    train_dataset = TranslationDataset(train_pairs, sp_src, sp_tgt, config)
    valid_dataset = TranslationDataset(valid_pairs, sp_src, sp_tgt, config)
    test_dataset = TranslationDataset(test_pairs, sp_src, sp_tgt, config)

    num_workers = getattr(config, "num_workers", 0)
    pin_memory = bool(getattr(config, "pin_memory", False))
    persistent_workers = bool(getattr(config, "persistent_workers", False)) and num_workers > 0
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "collate_fn": partial(collate_fn, pad_id=config.pad_id),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_loader, valid_loader, test_loader
