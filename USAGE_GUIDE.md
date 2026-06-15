# 사용 가이드 (Train/Test 분리 적용 후)

## 📋 데이터 분리 구조

- **Train 데이터**: 전체의 80% (학습 및 검증용)
- **Test 데이터**: 전체의 20% (최종 평가용, 학습 중 미사용)

## 🚀 명령어 사용법

### 1. 모델 학습 (Training)

```bash
cd /userHome/userhome4/sehoon/micro-tcn-main/micro-tcn-main
conda activate microtcn_env

# 기본 학습 (train subset 사용, 80% 데이터)
python train.py

# 특정 설정만 학습
python train.py --model_type tcn --train_subset train --val_subset train

# 전체 데이터로 학습 (권장하지 않음, test 데이터까지 사용)
python train.py --train_subset full --val_subset full
```

**중요 사항:**
- `--train_subset train`: 학습용 데이터 (80%)
- `--val_subset train`: 검증용 데이터 (학습 중 성능 모니터링, train과 동일)
- 학습 중에는 **절대 test 데이터를 사용하지 않습니다**

### 2. 모델 평가 (Evaluation)

```bash
# 최종 평가 - test subset 사용 (20% 데이터, 학습 중 미사용)
python test.py --eval_subset test

# 또는 전체 평가
python test.py --eval_subset full

# 특정 모델만 평가
python test.py --eval_subset test --model_dir ./lightning_logs/bulk/14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32
```

**중요 사항:**
- `--eval_subset test`: **최종 평가용** (20% 데이터)
- `--eval_subset train`: 학습 데이터로 평가 (과적합 확인용)
- `--eval_subset full`: 전체 데이터로 평가

### 3. 시각화 (Visualization)

#### TCN 전용 논문 스타일 시각화
```bash
# TCN 모델 시각화 (주파수 + 시간 도메인)
python visualize_tcn_paper_style.py

# 특정 샘플 지정
python visualize_tcn_paper_style.py --sample_idx 0

# 특정 체크포인트 사용
python visualize_tcn_paper_style.py \
    --tcn_checkpoint ./lightning_logs/bulk/14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32/lightning_logs/version_0/checkpoints/epoch=59-step=4379.ckpt
```

#### TCN vs FNO 비교 시각화
```bash
# TCN과 FNO 비교 (논문 스타일)
python compare_models_paper_style.py

# 특정 샘플 지정
python compare_models_paper_style.py --sample_idx 0
```

#### 기타 시각화
```bash
# TCN 성능 시각화 (파형 + 오차 그래프)
python visualize_final_tcn.py

# 여러 모델 비교
python visualize_results.py
```

## 📊 데이터 분리 확인

데이터가 올바르게 분리되었는지 확인하려면:

```python
from microtcn.data import CustomWaveDataset

# Train 데이터 확인
train_dataset = CustomWaveDataset('.', subset='train')
print(f"Train files: {len(train_dataset.input_files)}")

# Test 데이터 확인
test_dataset = CustomWaveDataset('.', subset='test')
print(f"Test files: {len(test_dataset.input_files)}")

# 전체 데이터 확인
full_dataset = CustomWaveDataset('.', subset='full')
print(f"Total files: {len(full_dataset.input_files)}")
```

## ✅ 가이드라인 준수 체크리스트

- [x] **코드 수정**: `CustomWaveDataset` 구현 완료 (seg???? 매칭)
- [x] **모델 재학습**: 사용자 데이터로 학습 완료
- [x] **평가 데이터 분리**: train/test 자동 분리 구현 완료

## 🔍 주의사항

1. **학습 시**: `--train_subset train` 사용 (80% 데이터)
2. **최종 평가 시**: `--eval_subset test` 사용 (20% 데이터)
3. **과적합 확인**: `--eval_subset train`으로 학습 데이터 성능 확인 가능
4. **일관성**: segment ID 기준으로 정렬되어 항상 동일한 분리 결과 보장

## 📝 예시 워크플로우

```bash
# 1. 모델 학습 (train 데이터 사용)
python train.py --train_subset train --val_subset train

# 2. 최종 평가 (test 데이터 사용 - 학습 중 미사용)
python test.py --eval_subset test

# 3. 시각화 (test 데이터에서 샘플 선택)
python visualize_tcn_paper_style.py --sample_idx 0
```

이제 가이드라인을 완벽하게 준수하는 프로젝트가 되었습니다! 🎉


