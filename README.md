# level1 STS nlp 8조 


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
|모델|test_pearson|리더보드 점수|적용 사항|
| --- | --- | --- | --- |
| klue/roberta-large | 0.9139 | 0.9216 |  |
| xlm-roberta-large | 0.9139 | 0.9216 | StepLR, 데이터 증강, special_token추가 |
| xlm-roberta-large (custom) | 0.9231 | 0.9198 |  |
| snunlp/KR-ELECTRA-discriminator | 0.9102 | 0.9221 |  |
| jhgan/ko-sbert-sts | 0.9236 | 0.9311 |  |

# 실행 방법
### train
```
python3 train.py
```

### inference
```
python3 inference.py
```

### sweep
```
python3 sweep.py
```