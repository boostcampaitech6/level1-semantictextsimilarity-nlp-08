# level1 STS nlp 8조 

# [Semantic Text Similarity]
STS란 두 텍스트가 얼마나 유사한지 판단하는 NLP Task입니다. 일반적으로 두 개의 문장을 입력하고, 이러한 문장쌍이 얼마나 의미적으로 서로 유사한지를 판단합니다. STS는 두 문장이 서로 동등한 양방향성을 가정하고 진행되며, 수치화된 점수를 출력할 수도 있습니다. 이번 대회에서는 STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 AI모델을 구축하고, 0과 5사이의 유사도 점수를 예측하는 것을 목적으로 합니다.

# Data
- 기본 데이터(base)
    - 학습 데이터셋 : 9,324개
    - 검증 데이터셋 : 550개
    - 평가 데이터셋 : 1,100개
- 데이터 증강
    - 학습 데이터셋 : 23,920개
    - 검증 데이터셋 : 550개
    - 평가 데이터셋 : 1,100개

# 모델 실험 결과
|모델|test_pearson|리더보드 점수|적용 사항(batch_size / epoch / loss_func / lr / step_size / gamma)|
| --- | --- | --- | --- |
| klue/roberta-large | 0.9299 | x | 64 / 30 / MSE / 2e-5 /  |
| xlm-roberta-large | 0.9309 | 0.9242 | 64 / 30 / MSE / 2e-5 / 10 / 0.3 |
| xlm-roberta-large (custom) | 0.9241 | x | 64 / 20 / MSE / 1e-5 / 5 / 0.7 |
| snunlp/KR-ELECTRA-discriminator | 0.9313 | 0.9165 | 64 / 30 / MSE / 1.24e-5 / 5 / 0.7 |
| jhgan/ko-sbert-sts | 0.9037 | 0.8787 | 16 / 50 / MSE / 1e-5 / 5 / 0.7 |
| kykim/electra-kor-base | 0.9325 | 0.9207 | 64 / 20 / MSE / 2e-5 / 10 / 0.5 |

# 실행 방법
- 아래의 Arguments들을 사용해 실행하면 됩니다.
### 공통 Arguments
- --model_name : 사용할 모델 이름
- --max_epoch : 실험할 epoch
- --shuffle : train_data shuffle에 대해 True / False
- --train_path : train_data의 경로
- --dev_path : val_data의 경로
- --test_path : test_data의 경로
- --predict_path : predict_data의 경로
- --custom : custom model 사용에 대해 True / False
### train
- --batch_size : 실험할 batch_size
- --learning_rate : 실험할 learning_rate
- --loss_function : 실험할 loss_function
- --step_size : lr scheduler의 step_size
- --gamma : lr_scheduler의 gamma
```
python3 train.py
```

### inference
- --batch_size : 실험할 batch_size
- --learning_rate : 실험할 learning_rate
```
python3 inference.py
```

### sweep
```
python3 sweep.py
```