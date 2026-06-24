# Model Train and Evaluation: 초기 실험

## 실험 결과 요약

- 실험 셋팅: 20,000개 학습 데이터, 10 epoch, batch size 32, learning rate `1e-4`, 3.74M parameter Seq2Seq Transformer로 학습했습니다.
- 실험 결과: Best validation loss는 `4.7113`, 테스트셋 BLEU는 `0.7415`, chrF는 `12.5541`로 실제 번역 품질은 낮았습니다.
- 다음 실험: 학습 데이터를 200,000개로 늘리고, 16 epoch, batch size 32로 약 100,000 update steps를 확보하는 설정을 제안했습니다.

## 1. 프로젝트 개요

본 프로젝트는 한국어 문장을 영어 문장으로 번역하는 Seq2Seq Transformer 기반 기계번역 실험입니다. 사전학습된 번역 모델을 사용하지 않고, 한국어-영어 병렬 말뭉치로 SentencePiece 토크나이저와 Transformer encoder-decoder 모델을 직접 학습했습니다.

주요 구성은 다음과 같습니다.

- 데이터 로딩: Hugging Face `datasets`를 사용했습니다.
- 토크나이저: 한국어 source와 영어 target에 대해 SentencePiece 모델을 사용했습니다.
- 모델: PyTorch `nn.Transformer` 기반 encoder-decoder 구조를 사용했습니다.
- 학습: teacher forcing 방식으로 cross-entropy loss를 최적화했습니다.
- 평가: 테스트셋 translation output을 BLEU와 chrF로 평가했습니다.
- 추론: greedy decoding을 사용했습니다.

### 데이터셋 설명

| 항목 | 내용 |
|---|---|
| Dataset | `shihyunlim/aihub-ko-en-everyday-expression` |
| Task | Korean-to-English translation |
| Source column | `ko` |
| Target column | `en` |
| 전체 문장쌍 | 1,690,052 |

데이터셋은 한국어 일상 표현과 그에 대응하는 영어 번역문으로 구성된 병렬 말뭉치입니다. 이번 초기 실험에서는 전체 데이터셋을 모두 사용하지 않고, 빠른 학습 및 평가를 위해 subset을 사용했습니다.

- Train subset: 20,000개
- Validation subset: 2,000개
- Test subset: 2,000개
- Split seed: 42

### 모델 구조 설명

모델은 Transformer encoder-decoder 구조입니다. 한국어 문장을 encoder 입력으로 넣고, decoder가 영어 문장을 autoregressive하게 생성하는 방식입니다.

핵심 구조는 다음과 같습니다.

- Source embedding + positional encoding을 encoder 입력으로 사용했습니다.
- Target embedding + positional encoding을 decoder 입력으로 사용했습니다.
- Encoder는 한국어 입력 문장의 contextual representation을 생성했습니다.
- Decoder는 encoder output과 이전 target token을 참조해 다음 영어 token을 예측했습니다.
- 출력 layer는 target vocabulary에 대한 token probability를 계산했습니다.

이번 초기 실험의 모델 크기는 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| Architecture | Seq2Seq Transformer |
| d_model | 128 |
| nhead | 4 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Feedforward dim | 256 |
| Dropout | 0.1 |
| Parameters | 3,743,040 |

## 2. 모델 학습 결과

### 실험 설정

| 항목 | 값 |
|---|---:|
| Dataset | `shihyunlim/aihub-ko-en-everyday-expression` |
| 전체 문장쌍 | 1,690,052 |
| Train data size | 20,000 |
| Valid data size | 2,000 |
| Test data size | 2,000 |
| Valid ratio | 0.1 |
| Test ratio | 0.1 |
| Random seed | 42 |
| Epochs | 10 |
| Batch size | 32 |
| Train batches | 625 |
| Valid batches | 63 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Max sequence length | 128 |
| Source vocab size | 8,000 |
| Target vocab size | 8,000 |
| Source tokenizer | `data/spm_ko.model` |
| Target tokenizer | `data/spm_en.model` |
| Device | CUDA |
| Checkpoint directory | `checkpoints` |

### 학습 Loss

| Epoch | Train loss | Valid loss |
|---:|---:|---:|
| 1 | 6.5478 | 5.7956 |
| 2 | 5.6076 | 5.4454 |
| 3 | 5.3409 | 5.2462 |
| 4 | 5.1667 | 5.1122 |
| 5 | 5.0350 | 5.0039 |
| 6 | 4.9304 | 4.9169 |
| 7 | 4.8422 | 4.8496 |
| 8 | 4.7654 | 4.8028 |
| 9 | 4.6974 | 4.7498 |
| 10 | 4.6375 | 4.7113 |

학습 결과는 다음과 같습니다.

- Best validation loss는 epoch 10의 `4.7113`입니다.
- 매 epoch validation loss가 개선되어 `checkpoints/best.pt`가 계속 갱신되었습니다.
- Train loss와 valid loss가 함께 감소했으므로 학습 자체는 정상적으로 진행되었습니다.
- 다만 최종 loss가 여전히 높기 때문에 실제 번역 품질은 평가 지표와 샘플 번역으로 확인해야 합니다.

![Train and validation loss](assets/260624_train_valid_loss.png)

## 3. 모델 평가 결과

### 평가 설정

| 항목 | 값 |
|---|---:|
| Evaluation notebook | `notebooks/03_evaluate_model.ipynb` |
| Checkpoint | `checkpoints/best.pt` |
| Checkpoint epoch | 10 |
| Checkpoint valid loss | 4.71127834774199 |
| Test data size | 2,000 |
| Test batches | 63 |
| Predictions | 2,000 |
| References | 2,000 |
| Decode strategy | Greedy |
| Max decode length | 128 |

### 정량 평가

| Metric | Score | 해석 |
|---|---:|---|
| BLEU | 0.7415 | 거의 0에 가까운 매우 낮은 점수입니다. Reference와 n-gram 단위 일치가 거의 없으며, 문장 수준 번역 정확도가 낮습니다. |
| chrF | 12.5541 | 문자 n-gram 겹침도 낮습니다. 일부 단어 형태가 겹치더라도 전체 문장 의미 전달은 제한적입니다. |

정량 평가 결과는 다음과 같이 해석했습니다.

- BLEU와 chrF는 모두 `sacrebleu` 기준 0~100 스케일의 corpus score입니다.
- BLEU `0.7415`는 사실상 번역 모델로 사용하기 어려운 수준입니다.
- chrF `12.5541`도 문장 구조와 의미가 reference와 충분히 맞지 않음을 보여줍니다.
- 테스트셋 2,000개에서 생성 문장은 `I'm sorry`, `a lot of the same`, `company`처럼 일부 상투적 표현으로 수렴하는 경향이 컸습니다.

### Sample Translation 평가

| No. | Reference | Prediction | 주요 오류 유형 | 코멘트 |
|---:|---|---|---|---|
| 1 | The satin pillow comes in king size, and queen size. | These is a lot of the FFF is a lot of the same. | 어색한 문장, 동어 반복, 문맥 불일치 | 문법적으로 `These is`가 어색했습니다. `FFF`, `a lot of the same`은 원문/정답의 베개 사이즈 정보와 무관했습니다. |
| 2 | Sorry, but your members are attacking us. | We have a lot of our company, and we'll be a lot of our company. | 동어 반복, 문맥 불일치 | 공격받고 있다는 의미가 사라졌고 `our company`가 반복되었습니다. |
| 3 | I am a hearty eater. | I'm sorry of my company. | 어색한 문장, 문맥 불일치 | 식성/식사량 의미를 반영하지 못했고 영어 표현도 자연스럽지 않았습니다. |
| 4 | Thank you for also sending the prototypes. | I'm sorrying you for your company. | 어색한 문장, 문맥 불일치 | `sorrying`은 비문이며, 시제품 발송에 대한 감사 의미가 손실되었습니다. |
| 5 | That is all for now, and thank you for your patience. | Yes, I'm sorry, and thank you for your order. | 부분 일치, 문맥 불일치 | 감사 표현은 일부 맞았지만 `patience`가 `order`로 바뀌어 상황이 달라졌습니다. |
| 6 | Your question has been sent. | I'm sorry, I'm going to buy a lot of the same. | 동어 반복, 문맥 불일치 | 질문 발송 완료 의미가 없고 구매 관련 문장으로 잘못 생성되었습니다. |
| 7 | OK, we can watch a horror film. | Yes, I'll be a lot of the same. | 부분 일치, 동어 반복, 문맥 불일치 | `Yes`는 대화 응답으로 일부 맞았지만 공포 영화 시청 의미는 빠졌습니다. |
| 8 | I'm not feeling so good. | I'm sorry, I'm sorry. | 동어 반복, 문맥 불일치 | 컨디션이 좋지 않다는 상태 설명 대신 사과 표현이 반복되었습니다. |
| 9 | I want to express it and it's hard to hide. | I'm sorry, and I'm going to make a lot of the same. | 동어 반복, 문맥 불일치 | 표현하고 싶다는 의미와 숨기기 어렵다는 뉘앙스가 모두 사라졌습니다. |
| 10 | You can also have Internet connection wherever you are in your home. | You can be able to be able to the same. | 어색한 문장, 동어 반복, 부분 일치 | `You can` 구조는 일부 남았지만 `be able to`가 반복되었고 인터넷 연결 의미가 빠졌습니다. |

샘플 번역 평가 결과는 다음과 같습니다.

- 샘플 10개 모두에서 문맥 불일치가 나타났습니다.
- 다수 문장에서 동어 반복과 비문이 함께 발생했습니다.
- 사과 표현, 회사 관련 표현, `a lot of the same` 같은 일반 문구가 반복되었습니다.
- 모델이 한국어 입력의 의미를 충분히 조건으로 사용하지 못하고 있음을 시사했습니다.

## 4. 종합 평가

이번 초기 실험의 결론은 다음과 같습니다.

- 20,000개 학습 샘플과 10 epoch 조건에서 loss는 안정적으로 감소했습니다.
- 그러나 BLEU/chrF와 샘플 번역을 함께 보면 실제 번역 성능은 낮았습니다.
- 현재 모델은 입력 문장의 의미를 충분히 반영하지 못했고, 특정 표현을 반복 생성하는 문제가 있었습니다.
- 따라서 다음 실험에서는 학습 step 수와 데이터 규모를 늘리는 방향이 필요합니다.

다음 실험 설정은 아래와 같이 조정하는 것이 적절합니다.

- 학습 데이터: `20,000 -> 200,000`
- Epoch: `10 -> 16`
- Batch size: `32` 유지
- 예상 update steps: `200,000 / 32 * 16 = 100,000`
- Learning rate: `1e-4` 유지
- Learning rate schedule: warmup/decay 스케줄러 추가

추가 점검 항목은 다음과 같습니다.

- 반복 생성 문제가 줄어드는지 확인해야 합니다.
- Greedy decoding과 beam search 결과를 비교해야 합니다.
- SentencePiece 토크나이저 출력과 디코딩 결과를 점검해야 합니다.
- 성능 개선이 제한적이면 모델 용량을 확대해야 합니다.
