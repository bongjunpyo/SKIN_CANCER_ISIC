# RETROSPECTIVE — SKIN_CANCER_ISIC

## 목표

피부 병변 이미지(ISIC 9클래스)를 분류하는 모델을 만들고, 밑바닥 CNN과 전이학습의
성능 차이를 직접 확인한다. Colab 환경에서 학습·기록 (커밋 `2a264c8` "Colab을 통해 생성됨", 2026-04).

## 기술 선택 이유

- **TensorFlow/Keras**: Colab 기본 지원 + `image_dataset_from_directory`로 디렉토리 구조를
  바로 데이터셋화할 수 있어 전처리 코드가 최소화됨.
- **EfficientNetB0**: ImageNet 사전학습 백본 중 파라미터 대비 성능이 좋고, 전처리(정규화)가
  모델에 내장되어 파이프라인이 단순해짐. [확인 필요: B0를 고른 실제 기준 — 강의/튜토리얼 참고 여부]
- **kagglehub**: 초기에 kaggle CLI + kaggle.json 방식을 시도했으나 인증 파일 업로드가
  실패했고(노트북 초기 셀에 `mv: cannot stat 'kaggle.json'` 에러 기록), 인증이 필요 없는
  kagglehub 다운로드로 전환.

## 막힌 점과 해결

| 막힌 점 | 해결 |
|---|---|
| kaggle CLI 인증 실패 (`kaggle.json` 없음) | kagglehub 공개 데이터셋 다운로드로 대체 — 인증 자체가 불필요 |
| 밑바닥 CNN val acc가 0.49~0.54에서 진동 | EfficientNetB0 전이학습으로 전환, val acc ≈ 0.62 달성 |
| `ImageDataGenerator`로 증강을 시도했으나 `image_dataset_from_directory` 파이프라인과 연결하지 못함 | 당시 미해결 — 셀만 남고 미사용. 이후 정리 작업에서 제거하고, `tf.keras.layers.Random*` 증강 층 방식을 다음 단계로 문서화 |
| Test 셋을 로드만 하고 평가 미수행 | 이후 정리 작업에서 classification_report + 혼동행렬 평가 셀 추가 (Colab 재실행 예정) |

## 다시 한다면

1. 처음부터 **전이학습을 베이스라인**으로 잡고, 밑바닥 CNN은 비교용 참고로만 돌린다.
2. 증강은 `ImageDataGenerator`(구식, tf.data와 안 섞임) 대신 **Keras 전처리 층**
   (`RandomFlip/RandomRotation/RandomZoom`)을 모델/파이프라인에 넣는다.
3. 학습 전에 **클래스 분포부터 확인**하고 class_weight를 기본 적용한다 — 의료 데이터는
   거의 항상 불균형하다.
4. 노트북에 가설→실험→결과→결론 서술을 실험과 동시에 남긴다. 사후 복원은 기억에 의존하게 됨.
5. [확인 필요: 당시 학습에 쓴 총 시간과 GPU 종류 — 회고 수치 보강용]
