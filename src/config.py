import os
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Central configuration for data, model, training, decoding, and runtime paths."""

    # dataset
    dataset_name: str = "shihyunlim/aihub-ko-en-everyday-expression"
    hf_token: str | None = os.getenv("HF_TOKEN")

    train_split: str = "train"
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # tokenizer
    sp_model_prefix_src: str = "data/spm_ko"
    sp_model_prefix_tgt: str = "data/spm_en"
    src_vocab_size: int = 8000
    tgt_vocab_size: int = 8000
    character_coverage_ko: float = 0.9995
    character_coverage_en: float = 1.0

    # sequence
    max_length: int = 128

    # model
    d_model: int = 128 # 256
    nhead: int = 4 # 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1

    # training
    batch_size: int = 32
    num_epochs: int = 40
    lr: float = 1e-4
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    checkpoint_dir: str = "checkpoints"

    # decoding
    max_decode_len: int = 128
    decode_strategy: str = "greedy"
    beam_size: int = 5

    # special token ids for sentencepiece defaults
    # SentencePiece trainer spec:
    # pad_id=0, unk_id=1, bos_id=2, eos_id=3
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # experiment
    num_examples_for_samples: int = 10
    train_subset_size: int = 200000  # e.g. 20000 for quick test
    valid_subset_size: int = 2000
    test_subset_size: int = 2000

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def sp_model_path_src(self) -> str:
        """Return the source SentencePiece model file path."""
        return f"{self.sp_model_prefix_src}.model"

    @property
    def sp_vocab_path_src(self) -> str:
        """Return the source SentencePiece vocabulary file path."""
        return f"{self.sp_model_prefix_src}.vocab"

    @property
    def sp_model_path_tgt(self) -> str:
        """Return the target SentencePiece model file path."""
        return f"{self.sp_model_prefix_tgt}.model"

    @property
    def sp_vocab_path_tgt(self) -> str:
        """Return the target SentencePiece vocabulary file path."""
        return f"{self.sp_model_prefix_tgt}.vocab"
