# CurseFilter

Streamlit에서 사용해보기<br/>
https://share.streamlit.io/heegyukim/cursefilter/main/app.py


# 데이터 다운로드
curse-data
```python
python prepare_data curse
```

# 모델 학습
### 욕설 분류 모델
```python
python train.py curse.yaml
# 지식 증류
python train_kd.py curse-kd.yaml
```
### 혐오표현 분류 모델
```python
python train.py apeach.yaml
# 지식 증류
python train_kd.py apeach-kd.yaml
```
학습이 완료되면 가장 좋은 지표와 그 모델의 체크포인트 경로를 출력합니다.

## 학습용 yaml 파일 구조
### 일반 학습
```yaml
dataset: curse # 데이터셋 종류, curse or apeach
model:
  huggingface_model_name: 'beomi/KcELECTRA-base' # 학습에 사용할 pretrained huggingface model
  label: curse # 분류할 데이터 라벨
  max_length: 64 # 학습 텍스트의 최대 토큰 길이

training:
  batch_size: 128    # 배치 크기
  lr: 0.0001         # 학습률
  monitor: val_acc   # 조기 종료와 체크포인트 저장을 위한 기준지표
  monitor_mode: max  # 조기 종료와 체크포인트 저장을 위한 기준지표
  patience: 10       # 기준지표가 몇 epoch동안 개선되지 않으면 종료할 것인지 나타냄
```

### 지식 증류 학습
```yaml
dataset: curse # 데이터셋 종류, curse or apeach

model:
  teacher:
    tokenizer: beomi/KcELECTRA-base                 # Teacher 모델의 토크나이저 이름
    checkpoint: ckpt\apeach_128_val_acc=0.8719.ckpt # Teacher 모델의 체크포인트 파일 경로
  student:
    huggingface_model_name: monologg/koelectra-small-v3-discriminator # Student 모델의 
    pretrained: true    # pretrained 모델을 사용할지 여부
    label: curse        # 클래스 라벨
    max_length: 64      # 학습 텍스트의 최대 토큰 길이
  kd:
    alpha: 0.3          # 지식 증류의 Alpha 값

training:
  batch_size: 128    # 배치 크기
  lr: 0.0001         # 학습률
  monitor: val_acc   # 조기 종료와 체크포인트 저장을 위한 기준지표
  monitor_mode: max  # 조기 종료와 체크포인트 저장을 위한 기준지표
  patience: 10       # 기준지표가 몇 epoch동안 개선되지 않으면 종료할 것인지 

```

## Reference
- [KcBERT](https://github.com/Beomi/KcBERT)
- [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret)
