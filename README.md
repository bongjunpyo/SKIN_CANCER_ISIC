# SKIN_CANCER_ISIC — 9클래스 피부 병변 분류

피부 병변 이미지를 9개 클래스로 분류하는 CNN 실험 노트북.
베이스라인 CNN → 전이학습(동결) → fine-tuning → hold-out Test 평가로 단계적으로 개선한다.

## 데이터셋

[Skin Cancer ISIC (9 classes)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) — kagglehub로 자동 다운로드 (공개 데이터셋, 인증 불필요).

| 구분 | 이미지 수 |
|---|---|
| Train (80/20으로 train/val 분할) | 2,239 |
| Test (hold-out) | 118 |

클래스(9): actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus,
pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion

## 방법

| 단계 | 방법 | 입력 스케일 |
|---|---|---|
| 1. 베이스라인 | Conv(16→32→64)+Dropout CNN, Adam(1e-3), 25ep | 0–255 → 모델 내부 `Rescaling(1./255)` |
| 2. 전이학습 | EfficientNetB0(ImageNet) 동결 + GAP + Dropout + Dense | 0–255 원본 (EfficientNet 내장 정규화) |
| 3. Fine-tuning | `block6a`↑만 해제(BN 동결 유지), Adam(1e-5), EarlyStopping(p=5, restore_best)+ReduceLROnPlateau | 〃 |
| 4. Test 평가 | classification_report(클래스별 P/R/F1) + 혼동행렬 heatmap | 〃 |

두 모델 모두 데이터 파이프라인은 0–255를 그대로 공급하며, 정규화 위치만 다르다.
EfficientNet 앞에 `Rescaling`을 추가하면 이중 스케일링으로 성능이 떨어지므로 주의.

## 결과

| 모델 | best val accuracy | Test accuracy |
|---|---|---|
| 베이스라인 CNN | ≈ 0.54 (최종 epoch 0.49~0.54 진동) | — |
| EfficientNetB0 (동결) | ≈ 0.62 | — |
| EfficientNetB0 (fine-tuned) | Colab 재실행 후 기입 | Colab 재실행 후 기입 |

> fine-tuning·Test 평가 셀은 추가 완료 상태이며, 수치는 Colab에서 노트북을 위→아래로 재실행해 채운다.

## 실행

Colab에서 `Skin_Cancer_ISIC.ipynb`를 열고 GPU 런타임으로 전체 실행 (위→아래 순서 실행 보장).

## 한계

- Test 118장 — 클래스별 지표의 신뢰구간이 넓음
- 클래스 불균형 미보정 (class_weight 미적용)
- 데이터 증강 미적용 — `RandomFlip/RandomRotation/RandomZoom`을 train 파이프라인에만 추가하는 것이 다음 단계
- 단일 시드·단일 분할 결과
