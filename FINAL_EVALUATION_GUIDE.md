# 최종 평가 및 비교 가이드

평가 완료 후 Best TCN을 선정하고 FNO와 최종 비교를 수행하는 방법입니다.

## 1. 평가 완료 확인

모든 모델(14개)의 평가가 완료되었는지 확인하세요:

```bash
python test.py --eval_subset test
```

평가가 완료되면 `test_results_test.p` 파일이 생성됩니다.

## 2. Best TCN 모델 선정

평가 결과를 분석하여 Aggregate Loss (L1 + STFT)가 가장 낮은 모델을 찾습니다:

```bash
python find_best_tcn.py
```

이 스크립트는:
- 모든 모델의 성능을 Aggregate Loss 기준으로 정렬하여 출력
- Best TCN 모델의 체크포인트 경로를 출력
- `best_tcn_checkpoint.txt` 파일에 체크포인트 정보 저장

**출력 예시:**
```
🏆 BEST TCN 모델 선정
================================================================================
모델 ID: 1-uTCN-300__causal__4-10-13__fraction-1.0-bs32
체크포인트 경로: ./lightning_logs/bulk/1-uTCN-300__causal__4-10-13__fraction-1.0-bs32/lightning_logs/version_0/checkpoints/epoch=59.ckpt
평균 Aggregate Loss: 3.5421
평균 L1 Loss: 3.54e-01
평균 STFT Loss: 4.550
평균 LUFS: 10.385
평가 샘플 수: 465
```

## 3. 최종 비교 (Best TCN vs FNO)

선정된 Best TCN과 FNO 모델을 비교합니다:

```bash
python compare_final.py --tcn_checkpoint <BEST_TCN_CHECKPOINT_PATH>
```

또는 `find_best_tcn.py`가 출력한 명령어를 그대로 사용할 수 있습니다.

**FNO 체크포인트 경로 지정 (선택사항):**
```bash
python compare_final.py \
    --tcn_checkpoint <BEST_TCN_CHECKPOINT_PATH> \
    --fno_checkpoint /userHome/userhome4/sehoon/neuraloperator-main/checkpoints/best_fno_model.pt
```

**다른 테스트 샘플 사용:**
```bash
python compare_final.py \
    --tcn_checkpoint <BEST_TCN_CHECKPOINT_PATH> \
    --sample_idx 10
```

## 4. 결과 확인

최종 비교 결과는 `final_comparison.png`에 저장됩니다.

이 파일에는 다음이 포함됩니다:
- **전체 파형 비교**: Target, Best TCN, FNO의 전체 파형
- **확대 뷰**: 처음 0.1초 구간의 상세 비교
- **오차 분석**: Pepero 스타일의 오차 시각화

**결과 해석:**
- 검은 선(정답)을 파란/빨간 선(예측)이 잘 따라가는 형태라면 성공
- MSE와 NRMSE가 낮을수록 좋은 성능
- 그래프에서 두 모델 중 어느 것이 더 정답에 가까운지 확인

## 5. 전체 워크플로우 요약

```bash
# 1. 평가 실행
python test.py --eval_subset test

# 2. Best TCN 선정
python find_best_tcn.py

# 3. 최종 비교 (체크포인트 경로는 find_best_tcn.py 출력에서 복사)
python compare_final.py --tcn_checkpoint <체크포인트_경로>

# 4. 결과 확인
# final_comparison.png 파일 확인
```

## 주의사항

1. **FNO 체크포인트 경로**: FNO 모델의 체크포인트 경로가 기본값과 다르면 `--fno_checkpoint` 옵션으로 지정하세요.

2. **테스트 샘플**: 기본적으로 첫 번째 테스트 샘플(sample_idx=0)을 사용합니다. 다른 샘플을 사용하려면 `--sample_idx` 옵션을 변경하세요.

3. **메모리**: GPU 메모리가 부족하면 배치 크기를 줄이거나 CPU를 사용하세요.

## 문제 해결

**Q: `test_results_test.p` 파일이 없어요**
- A: 먼저 `python test.py --eval_subset test`를 실행하여 평가를 완료하세요.

**Q: Best TCN 체크포인트를 찾을 수 없어요**
- A: `find_best_tcn.py`가 출력한 체크포인트 경로가 정확한지 확인하세요. 경로에 공백이나 특수문자가 있으면 따옴표로 감싸세요.

**Q: FNO 모델을 로드할 수 없어요**
- A: FNO 체크포인트 경로가 올바른지 확인하고, `--fno_checkpoint` 옵션으로 올바른 경로를 지정하세요.


