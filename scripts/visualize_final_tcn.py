# NumPy compatibility patch for older PyTorch Lightning / Librosa versions
import numpy as np
try:
    np.object = object
    np.bool = bool
    np.int = int
    np.float = float
    np.long = int
    np.complex = complex
except AttributeError:
    pass

# Resolve project root directory
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

#!/usr/bin/env python3
"""
TCN 모델 시각화 스크립트 - 파형이 겹쳐지고 오차가 빼빼로처럼 나오는 그래프 생성
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from torch.utils.data import DataLoader
from pathlib import Path

# Set root directory
ROOT_DIR = Path("/userHome/userhome4/sehoon/micro-tcn-main/micro-tcn-main")
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from microtcn.tcn import TCNModel
from microtcn.data import CustomWaveDataset
from microtcn.utils import center_crop, causal_crop

# ★ 중요: 가장 성능이 좋았던 TCN 체크포인트 경로
# Config 14 (uTCN-324-16) - 최고 성능 모델
TCN_CKPT = "./lightning_logs/bulk/14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32/lightning_logs/version_0/checkpoints/epoch=59-step=4379.ckpt"

# 또는 Config 2 (uTCN-100 causal) - 경량 모델 중 최고 성능
# TCN_CKPT = "./lightning_logs/bulk/2-uTCN-100__causal__4-10-5__fraction-1.0-bs32/lightning_logs/version_6/checkpoints/epoch=59-step=4379.ckpt"


def visualize():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. TCN 모델 로드
    if not os.path.exists(TCN_CKPT):
        print(f"Error: 체크포인트가 없습니다. 경로를 확인해주세요: {TCN_CKPT}")
        print("\n사용 가능한 체크포인트를 찾는 중...")
        import glob
        checkpoints = glob.glob("./lightning_logs/bulk/*/lightning_logs/version_*/checkpoints/epoch=59*.ckpt")
        if checkpoints:
            print("사용 가능한 체크포인트:")
            for ckpt in checkpoints[:5]:
                print(f"  - {ckpt}")
        return

    print(f"Loading TCN model from {TCN_CKPT}...")
    model = TCNModel.load_from_checkpoint(TCN_CKPT, map_location=device)
    model.eval()
    model.freeze()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"Model loaded: {model.hparams.name if hasattr(model.hparams, 'name') else 'Unknown'}")
    
    # 2. 데이터 로드 (테스트셋)
    print("\nLoading test dataset...")
    dataset = CustomWaveDataset(
        root_dir=str(ROOT_DIR),
        subset="test",
        length=131072,  # Full file length for better visualization
        preload=False,
        half=False
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Test dataset size: {len(dataset)} samples")
    
    # 3. 추론 및 시각화
    print("\nRunning inference on random sample...")
    with torch.no_grad():
        input_audio, target_audio = next(iter(loader))
        input_audio = input_audio.to(device)
        target_audio = target_audio.to(device)

        # TCN 추론
        pred_audio = model(input_audio, None)

        # 길이 맞추기 (Crop)
        if hasattr(model.hparams, 'causal') and model.hparams.causal:
            target_crop = causal_crop(target_audio, pred_audio.shape[-1])
        else:
            target_crop = center_crop(target_audio, pred_audio.shape[-1])

        # Numpy 변환
        target_np = target_crop.squeeze().cpu().numpy()
        pred_np = pred_audio.squeeze().cpu().numpy()
        input_np = input_audio.squeeze().cpu().numpy()
        
        # Ensure same length
        min_len = min(len(target_np), len(pred_np), len(input_np))
        target_np = target_np[:min_len]
        pred_np = pred_np[:min_len]
        input_np = input_np[:min_len]
        
        error_np = np.abs(target_np - pred_np)

        # MSE 계산
        mse = np.mean((target_np - pred_np)**2)
        mae = np.mean(error_np)
        rmse = np.sqrt(mse)
        target_range = np.max(target_np) - np.min(target_np)
        nrmse = (rmse / target_range * 100) if target_range > 0 else 0.0
        
        print(f"\n{'='*60}")
        print("TCN MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"MSE:  {mse:.6e}")
        print(f"MAE:  {mae:.6e}")
        print(f"RMSE: {rmse:.6e}")
        print(f"NRMSE: {nrmse:.2f}%")
        print(f"{'='*60}")

        # --- 그래프 그리기 (사용자님이 원하시는 스타일) ---
        fig = plt.figure(figsize=(15, 12))
        
        # Create time axis (in seconds)
        sample_rate = 192000
        time_axis = np.arange(min_len) / sample_rate

        # 1. 전체 파형 (겹쳐 보이기)
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time_axis, target_np, 'k-', label='Target (Distorted Audio)', alpha=0.7, linewidth=1.0)
        ax1.plot(time_axis, pred_np, 'b--', label='TCN Prediction', alpha=0.8, linewidth=1.2)
        ax1.set_title(f"1. Full Waveform Comparison (MSE: {mse:.6e}, NRMSE: {nrmse:.2f}%)", 
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([time_axis[0], time_axis[-1]])

        # 2. 확대 (Zoom) - 위상 일치 확인
        ax2 = plt.subplot(3, 1, 2)
        center = len(target_np) // 4  # 1/4 지점에서 시작
        zoom = 500  # 500 샘플 확대
        zoom_end = min(center + zoom, len(target_np))
        zoom_time = time_axis[center:zoom_end]
        
        ax2.plot(zoom_time, target_np[center:zoom_end], 'k-', label='Target', linewidth=2.0, alpha=0.8)
        ax2.plot(zoom_time, pred_np[center:zoom_end], 'b--', label='TCN Prediction', linewidth=2.0, alpha=0.9)
        ax2.set_title(f"2. Zoomed View (Samples {center}-{zoom_end}) - Phase Tracking Detail", 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. 오차 막대 그래프 (빼빼로 스타일)
        ax3 = plt.subplot(3, 1, 3)
        
        # Sample every Nth point to avoid overcrowding
        step = max(1, min_len // 2000)  # Show ~2000 points max
        indices = np.arange(0, min_len, step)
        
        # Use stem plot for Pepero-like vertical bars
        markerline, stemlines, baseline = ax3.stem(time_axis[indices], error_np[indices], 
                                                   linefmt='r-', markerfmt='ro', basefmt=' ')
        plt.setp(markerline, 'alpha', 0.6, 'markersize', 2)
        plt.setp(stemlines, 'alpha', 0.6, 'linewidth', 0.5)
        
        ax3.set_title("3. Error Analysis (Pepero Style) - Absolute Error |Target - Pred|", 
                     fontsize=13, fontweight='bold')
        ax3.set_xlabel('Time (seconds)', fontsize=11)
        ax3.set_ylabel('Absolute Error', fontsize=11)
        ax3.legend([markerline], [f'Error (MSE: {mse:.6e})'], loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([time_axis[0], time_axis[-1]])

        plt.tight_layout()
        output_file = "tcn_success_result.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n>>> 결과 저장 완료: {output_file}")
        print(f"    파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")


if __name__ == "__main__":
    visualize()

