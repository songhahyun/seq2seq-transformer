# seq2seq-transformer

Korean-to-English sequence-to-sequence translation experiment built with PyTorch `nn.Transformer`. The project loads a Hugging Face dataset, trains SentencePiece tokenizers, trains a Transformer encoder-decoder model, and evaluates generated translations with BLEU and chrF.

## Features

- Hugging Face `datasets` based Korean-English data loading
- Korean and English SentencePiece tokenizer training
- PyTorch `nn.Transformer` encoder-decoder model
- Teacher forcing training loop
- Validation-loss based best checkpoint saving
- Greedy decoding for translation
- BLEU and chrF evaluation
- CLI and Jupyter notebook workflows

## Project Structure

```text
seq2seq-transformer/
+-- src/
|   +-- __init__.py            # Python package marker
|   +-- config.py              # Dataset, tokenizer, model, and training config
|   +-- data_pipeline.py       # Dataset loading, splitting, tokenization, DataLoader
|   +-- transformer_model.py   # Seq2Seq Transformer model
|   +-- model_utils.py         # Model, tokenizer, and checkpoint helpers
|   +-- train.py               # Training loop and checkpoint saving
|   +-- evaluate.py            # Evaluation workflow and prediction generation
|   +-- translate.py           # Single-text translation workflow
|   +-- metrics.py             # BLEU and chrF metrics
|   +-- main.py                # CLI entrypoint
+-- notebooks/
|   +-- 01_sentencepiece_tokenizer.ipynb
|   +-- 02_train_model.ipynb
|   +-- 03_evaluate_model.ipynb
+-- data/                      # Generated SentencePiece artifacts
+-- checkpoints/               # Generated model checkpoints
+-- pyproject.toml
+-- .env
+-- .gitignore
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install from `pyproject.toml`:

```bash
pip install ".[notebooks]"
```

For CUDA, install the PyTorch build that matches your CUDA version.

## Environment

Create a `.env` file in the repository root:

```env
HF_TOKEN=your_huggingface_token
```

`HF_TOKEN` is loaded in `src/config.py` and passed to `datasets.load_dataset()`. It may be optional if the dataset is public in your environment.

## Dataset

Default dataset:

```text
shihyunlim/aihub-ko-en-everyday-expression
```

The pipeline expects each row to contain:

- `ko`: Korean source sentence
- `en`: English target sentence

Dataset name, split settings, subset sizes, model size, and training parameters are configured in [src/config.py](src/config.py).

## CLI Usage

Run all commands from the repository root.

Train:

```bash
python -m src.main --mode train
```

Evaluate the best checkpoint:

```bash
python -m src.main --mode evaluate
```

Evaluate another checkpoint:

```bash
python -m src.main --mode evaluate --checkpoint checkpoints/latest.pt
```

Translate one sentence:

```bash
python -m src.main --mode translate --text "I like machine learning."
```

Translate with another checkpoint:

```bash
python -m src.main --mode translate --text "I like machine learning." --checkpoint checkpoints/latest.pt
```

Run a W&B tracked training and evaluation experiment:

```bash
wandb login
python -m src.wandb_experiment --project seq2seq-transformer --run-name baseline
```

Use offline mode when you want to log locally first:

```bash
python -m src.wandb_experiment --offline --run-name baseline
```

## Notebook Workflow

Run notebooks in this order:

1. `notebooks/01_sentencepiece_tokenizer.ipynb`
   - Load the dataset
   - Train or load SentencePiece tokenizers
   - Save `data/spm_*` artifacts

2. `notebooks/02_train_model.ipynb`
   - Build the model
   - Run the train/validation loop
   - Save `checkpoints/latest.pt` and `checkpoints/best.pt`

3. `notebooks/03_evaluate_model.ipynb`
   - Load `checkpoints/best.pt`
   - Evaluate on the test split
   - Compute BLEU and chrF with `src.metrics`

4. `notebooks/04_train_and_evaluate_wandb.ipynb`
   - Run train and evaluation in one notebook
   - Log losses, BLEU, chrF, sample translations, and checkpoint artifacts to W&B

## Generated Artifacts

SentencePiece artifacts:

```text
data/spm_ko.model
data/spm_ko.vocab
data/spm_en.model
data/spm_en.vocab
```

Model checkpoints:

```text
checkpoints/latest.pt
checkpoints/best.pt
```

`latest.pt` is overwritten after every epoch. `best.pt` is updated only when validation loss improves. Checkpoints contain model weights, optimizer state, epoch, train/validation losses, vocabulary sizes, and config values excluding `HF_TOKEN`.

## Key Config Defaults

```python
d_model = 128
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
batch_size = 16
num_epochs = 1
lr = 1e-4
train_subset_size = 20000
valid_subset_size = 2000
test_subset_size = 2000
checkpoint_dir = "checkpoints"
sp_model_prefix_src = "data/spm_ko"
sp_model_prefix_tgt = "data/spm_en"
```

The defaults are intended for quick experiments. Increase `num_epochs`, subset sizes, and model dimensions for better translation quality.

## Notes

- The model is trained from scratch; no pretrained translation model is used.
- Inference uses greedy decoding.
- `Config.device` uses `cuda` when available, otherwise `cpu`.
- `data/`, `checkpoints/`, and `.env` are ignored by git.
