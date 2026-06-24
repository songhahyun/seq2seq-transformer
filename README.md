# seq2seq-transformer

한국어 문장을 영어 문장으로 번역하는 seq2seq Transformer 실험 프로젝트입니다. Hugging Face 데이터셋을 불러오고, SentencePiece 토크나이저를 학습한 뒤, PyTorch `nn.Transformer` 기반 인코더-디코더 모델을 학습하고 평가했습니다.

## 주요 기능

- `src/data_pipeline.py`: Hugging Face `datasets` 기반 한영 병렬 데이터 로딩
- `src/data_pipeline.py`: 한국어와 영어 SentencePiece 토크나이저 학습
- `src/transformer_model.py`: PyTorch `nn.Transformer` 기반 seq2seq 모델 구현
- `src/train.py`: teacher forcing 학습 루프 구성
- `src/train.py`, `src/model_utils.py`: 검증 손실 기준으로 `best.pt` 체크포인트를 저장
- `src/transformer_model.py`, `src/translate.py`, `src/evaluate.py`: greedy decoding과 beam search decoding을 옵션으로 지원
- `src/metrics.py`, `src/evaluate.py`: BLEU와 chrF로 번역 품질을 평가
- `src/main.py`, `src/wandb_experiment.py`: CLI, Jupyter Notebook, W&B 실험 실행을 지원

## 프로젝트 구조

```text
seq2seq-transformer/
├── src/
│   ├── config.py              # 데이터셋, 토크나이저, 모델, 학습 설정
│   ├── data_pipeline.py       # 데이터 로딩, 분할, 토큰화, DataLoader 구성
│   ├── transformer_model.py   # seq2seq Transformer 모델
│   ├── model_utils.py         # 모델, 토크나이저, 체크포인트 유틸리티
│   ├── train.py               # 학습 루프와 체크포인트 저장 로직
│   ├── evaluate.py            # 평가와 예측 생성 로직
│   ├── translate.py           # 단일 문장 번역 로직
│   ├── metrics.py             # BLEU와 chrF 계산 로직
│   ├── main.py                # CLI 진입점
│   └── wandb_experiment.py    # W&B 학습 및 평가 실행 스크립트
├── notebooks/
│   ├── 01_sentencepiece_tokenizer.ipynb
│   ├── 02_train_model.ipynb
│   ├── 03_evaluate_model.ipynb
│   └── 04_train_and_evaluate_wandb.ipynb
├── data/                      # 생성된 SentencePiece 파일
├── reports/                   # 평가 리포트와 이미지
├── checkpoints/               # 생성된 모델 체크포인트 경로
├── pyproject.toml
├── .env.example
└── README.md
```

## 설치

Python 3.11 이상을 권장했습니다.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install ".[notebooks]"
```

CUDA를 사용할 경우, 로컬 CUDA 버전에 맞는 PyTorch 빌드가 필요한 항목입니다.

## 환경 변수

프로젝트 루트에 `.env` 파일을 생성했습니다.

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
WANDB_MODE=online
```

`HF_TOKEN`은 `src/config.py`에서 읽어 `datasets.load_dataset()`에 전달했습니다. 데이터셋 접근 권한이 필요 없는 환경에서는 생략 가능한 항목입니다. `.env.example` 파일은 템플릿입니다.

## 데이터셋

기본 데이터셋은 다음 항목입니다.

```text
shihyunlim/aihub-ko-en-everyday-expression
```

데이터 파이프라인은 각 행에 다음 컬럼이 있다고 가정했습니다.

- `ko`: 한국어 원문입니다.
- `en`: 영어 번역문입니다.

데이터셋 이름, 분할 비율, 부분 데이터 크기, 모델 크기, 학습 설정은 `src/config.py`에서 관리했습니다.

## CLI 사용법

모든 명령은 프로젝트 루트에서 실행했습니다.

학습을 실행했습니다.

```bash
python -m src.main --mode train
```

기본 체크포인트로 평가했습니다.

```bash
python -m src.main --mode evaluate
```

특정 체크포인트로 평가했습니다.

```bash
python -m src.main --mode evaluate --checkpoint checkpoints/latest.pt
```

단일 문장을 번역했습니다.

```bash
python -m src.main --mode translate --text "I like machine learning."
```

beam search로 번역했습니다.

```bash
python -m src.main --mode translate --text "I like machine learning." --decode-strategy beam --beam-size 5
```

## W&B 실험

W&B로 학습, 평가, 샘플 번역, 체크포인트 artifact 기록을 실행했습니다.

```bash
wandb login
python -m src.wandb_experiment --project seq2seq-transformer --run-name baseline
```

로컬 로그만 남길 때는 offline 모드를 사용했습니다.

```bash
python -m src.wandb_experiment --offline --run-name baseline
```

## 노트북 실행 순서

1. `notebooks/01_sentencepiece_tokenizer.ipynb`
   - 데이터셋을 불러왔습니다.
   - SentencePiece 토크나이저를 학습하거나 로드했습니다.
   - `data/spm_*` 파일을 저장했습니다.

2. `notebooks/02_train_model.ipynb`
   - 모델을 생성했습니다.
   - 학습과 검증 루프를 실행했습니다.
   - `checkpoints/latest.pt`와 `checkpoints/best.pt`를 저장했습니다.

3. `notebooks/03_evaluate_model.ipynb`
   - `checkpoints/best.pt`를 로드했습니다.
   - 테스트 분할에서 평가했습니다.
   - BLEU와 chrF를 계산했습니다.

4. `notebooks/04_train_and_evaluate_wandb.ipynb`
   - 학습과 평가를 하나의 노트북에서 실행했습니다.
   - 손실, BLEU, chrF, 샘플 번역, 체크포인트 artifact를 W&B에 기록했습니다.

## 생성 파일

SentencePiece 파일은 다음 경로에 생성했습니다.

```text
data/spm_ko.model
data/spm_ko.vocab
data/spm_en.model
data/spm_en.vocab
```

모델 체크포인트는 다음 경로에 생성했습니다.

```text
checkpoints/latest.pt
checkpoints/best.pt
```

`latest.pt`는 매 epoch마다 덮어썼습니다. `best.pt`는 검증 손실이 개선되었을 때만 갱신했습니다. 체크포인트에는 모델 가중치, optimizer 상태, epoch, 학습 손실, 검증 손실, vocabulary 크기, `HF_TOKEN`을 제외한 설정값을 저장했습니다.

## 주요 기본 설정

```python
d_model = 128
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
batch_size = 32
num_epochs = 16
lr = 1e-4
train_subset_size = 200000
valid_subset_size = 2000
test_subset_size = 2000
decode_strategy = "greedy"
beam_size = 5
checkpoint_dir = "checkpoints"
sp_model_prefix_src = "data/spm_ko"
sp_model_prefix_tgt = "data/spm_en"
```

기본 설정은 실험용 구성입니다. 더 높은 번역 품질이 필요하면 epoch 수, 데이터 크기, 모델 차원을 늘리는 방식으로 조정했습니다.

## 참고 사항

- 사전학습 번역 모델을 사용하지 않고 처음부터 학습했습니다.
- 기본 추론 방식은 greedy decoding입니다.
- `Config.device`는 CUDA가 있으면 `cuda`, 없으면 `cpu`를 사용했습니다.
- `data/`, `checkpoints/`, `.env`는 git에서 제외했습니다.
