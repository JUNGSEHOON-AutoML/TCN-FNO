#!/usr/bin/env python3
"""
평가 결과를 분석하여 Best TCN 모델을 선정합니다.

Aggregate Loss (L1 + STFT)가 가장 낮은 모델을 찾아서
체크포인트 경로와 성능 정보를 출력합니다.
"""

import os
import sys
import glob
import pickle
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

# Resolve project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

parser = ArgumentParser()
parser.add_argument('--results_file', type=str, default='test_results_test.p',
                    help='평가 결과 pickle 파일 경로')
parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk',
                    help='모델 디렉토리 경로')
args = parser.parse_args()

# 평가 결과 로드
if not os.path.exists(args.results_file):
    print(f"에러: 평가 결과 파일을 찾을 수 없습니다: {args.results_file}")
    print("먼저 'python test.py --eval_subset test'를 실행하세요.")
    sys.exit(1)

print(f"평가 결과 로드 중: {args.results_file}")
overall_results = pickle.load(open(args.results_file, "rb"))

# 각 모델의 평균 Aggregate Loss 계산
model_scores = []

for model_id, results in overall_results.items():
    # 각 샘플의 Aggregate Loss 수집
    agg_scores = []
    l1_scores = []
    stft_scores = []
    lufs_scores = []
    
    for key, val in results.items():
        agg_scores.extend(val.get("Agg", []))
        l1_scores.extend(val.get("L1", []))
        stft_scores.extend(val.get("STFT", []))
        lufs_scores.extend(val.get("LUFS", []))
    
    if len(agg_scores) > 0:
        mean_agg = np.mean(agg_scores)
        mean_l1 = np.mean(l1_scores)
        mean_stft = np.mean(stft_scores)
        mean_lufs = np.mean(lufs_scores)
        
        # 체크포인트 경로 찾기
        model_dir = os.path.join(args.model_dir, model_id)
        checkpoint_candidates = glob.glob(os.path.join(model_dir,
                                                       "lightning_logs",
                                                       "version_*",
                                                       "checkpoints",
                                                       "*.ckpt"))
        
        # epoch=59 체크포인트 우선 선택
        checkpoint_path = None
        for cp in checkpoint_candidates:
            if "epoch=59" in os.path.basename(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None and len(checkpoint_candidates) > 0:
            # epoch=59가 없으면 마지막 체크포인트 사용
            checkpoint_path = sorted(checkpoint_candidates)[-1]
        
        model_scores.append({
            "model_id": model_id,
            "checkpoint_path": checkpoint_path,
            "mean_agg": mean_agg,
            "mean_l1": mean_l1,
            "mean_stft": mean_stft,
            "mean_lufs": mean_lufs,
            "num_samples": len(agg_scores)
        })

# Aggregate Loss 기준으로 정렬
model_scores.sort(key=lambda x: x["mean_agg"])

# 결과 출력
print("\n" + "="*80)
print("모델 성능 순위 (Aggregate Loss = L1 + STFT 기준, 낮을수록 좋음)")
print("="*80)
print(f"{'순위':<6} {'모델 ID':<50} {'Agg Loss':<12} {'L1':<12} {'STFT':<12} {'LUFS':<12}")
print("-"*80)

for rank, model in enumerate(model_scores, 1):
    print(f"{rank:<6} {model['model_id']:<50} "
          f"{model['mean_agg']:<12.4f} {model['mean_l1']:<12.4e} "
          f"{model['mean_stft']:<12.4f} {model['mean_lufs']:<12.4f}")

print("\n" + "="*80)
print("🏆 BEST TCN 모델 선정")
print("="*80)
best_model = model_scores[0]
print(f"모델 ID: {best_model['model_id']}")
print(f"체크포인트 경로: {best_model['checkpoint_path']}")
print(f"평균 Aggregate Loss: {best_model['mean_agg']:.4f}")
print(f"평균 L1 Loss: {best_model['mean_l1']:.4e}")
print(f"평균 STFT Loss: {best_model['mean_stft']:.4f}")
print(f"평균 LUFS: {best_model['mean_lufs']:.4f}")
print(f"평가 샘플 수: {best_model['num_samples']}")

# 체크포인트 경로를 파일로 저장
checkpoint_info_file = "best_tcn_checkpoint.txt"
with open(checkpoint_info_file, "w") as f:
    f.write(f"# Best TCN Model Information\n")
    f.write(f"MODEL_ID={best_model['model_id']}\n")
    f.write(f"CHECKPOINT_PATH={best_model['checkpoint_path']}\n")
    f.write(f"MEAN_AGG_LOSS={best_model['mean_agg']:.4f}\n")
    f.write(f"MEAN_L1_LOSS={best_model['mean_l1']:.4e}\n")
    f.write(f"MEAN_STFT_LOSS={best_model['mean_stft']:.4f}\n")
    f.write(f"MEAN_LUFS={best_model['mean_lufs']:.4f}\n")

print(f"\n체크포인트 정보가 저장되었습니다: {checkpoint_info_file}")

# 최종 비교 명령어 출력
if best_model['checkpoint_path']:
    print("\n" + "="*80)
    print("다음 명령어로 최종 비교를 실행하세요:")
    print("="*80)
    checkpoint_path_quoted = f'"{best_model["checkpoint_path"]}"' if ' ' in best_model['checkpoint_path'] else best_model['checkpoint_path']
    print(f"python compare_final.py --tcn_checkpoint {checkpoint_path_quoted}")
    print("="*80)
else:
    print("\n⚠️  경고: 체크포인트 경로를 찾을 수 없습니다.")
    print("모델 디렉토리를 확인하거나 수동으로 체크포인트 경로를 지정하세요.")

