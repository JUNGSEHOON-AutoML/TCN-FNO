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
TCN 모델 전용 논문 스타일 시각화 (주파수 도메인 + 시간 도메인)

논문의 Figure 3 & 5 스타일:
- 왼쪽: 주파수 스펙트럼 (막대그래프) - 타겟과 TCN 예측의 주파수 성분 비교
- 오른쪽: 시간 파형 (물결) - 타겟과 TCN 예측의 파형 비교
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from argparse import ArgumentParser
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Set root directory
ROOT_DIR = Path("/userHome/userhome4/sehoon/micro-tcn-main/micro-tcn-main")
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

# Import TCN model and dataset
from microtcn.tcn import TCNModel
from microtcn.data import CustomWaveDataset
from microtcn.utils import center_crop, causal_crop


def load_tcn_model(checkpoint_path):
    """Load TCN model from checkpoint"""
    print(f"Loading TCN model from {checkpoint_path}...")
    model = TCNModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.freeze()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def compute_fft_spectrum(signal_data, sample_rate=192000, n_fft=None):
    """Compute FFT spectrum and extract dominant frequencies"""
    if n_fft is None:
        n_fft = len(signal_data)
    
    # Compute FFT
    fft_vals = fft(signal_data, n=n_fft)
    fft_freq = fftfreq(n_fft, 1/sample_rate)
    
    # Get positive frequencies only
    positive_freq_idx = fft_freq >= 0
    fft_freq = fft_freq[positive_freq_idx]
    fft_magnitude = np.abs(fft_vals[positive_freq_idx])
    
    # Normalize
    fft_magnitude = fft_magnitude / np.max(fft_magnitude) if np.max(fft_magnitude) > 0 else fft_magnitude
    
    return fft_freq, fft_magnitude


def extract_dominant_frequencies(freqs, magnitudes, top_n=15, min_freq=20, max_freq=5000):
    """Extract top N dominant frequencies from spectrum"""
    # Filter frequency range
    valid_idx = (freqs >= min_freq) & (freqs <= max_freq)
    filtered_freqs = freqs[valid_idx]
    filtered_mags = magnitudes[valid_idx]
    
    # Find peaks
    peaks, properties = find_peaks(filtered_mags, height=0.1, distance=len(filtered_freqs)//100)
    
    if len(peaks) == 0:
        # If no peaks found, use top N by magnitude
        top_indices = np.argsort(filtered_mags)[-top_n:][::-1]
        peak_freqs = filtered_freqs[top_indices]
        peak_mags = filtered_mags[top_indices]
    else:
        peak_freqs = filtered_freqs[peaks]
        peak_mags = filtered_mags[peaks]
        # Sort by magnitude and take top N
        sorted_idx = np.argsort(peak_mags)[-top_n:][::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_mags = peak_mags[sorted_idx]
    
    return peak_freqs, peak_mags


def plot_tcn_paper_style(target_np, tcn_pred_np, tcn_mse, tcn_nrmse,
                         save_path, sample_rate=192000):
    """논문 스타일 시각화: 주파수 도메인(왼쪽) + 시간 도메인(오른쪽) - TCN만"""
    
    # Ensure same length
    min_len = min(len(target_np), len(tcn_pred_np))
    target_np = target_np[:min_len]
    tcn_pred_np = tcn_pred_np[:min_len]
    
    # Create time axis (in seconds)
    time_axis = np.arange(min_len) / sample_rate
    
    # Compute FFT spectra
    print("Computing frequency spectra...")
    target_freqs, target_mags = compute_fft_spectrum(target_np, sample_rate)
    tcn_freqs, tcn_mags = compute_fft_spectrum(tcn_pred_np, sample_rate)
    
    # Extract dominant frequencies
    target_peak_freqs, target_peak_mags = extract_dominant_frequencies(target_freqs, target_mags, top_n=15)
    tcn_peak_freqs, tcn_peak_mags = extract_dominant_frequencies(tcn_freqs, tcn_mags, top_n=15)
    
    # Create figure with 1 row, 2 columns (Frequency and Time domain)
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3, width_ratios=[1, 1])
    
    # ========== TCN: Frequency Domain (Left) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("TCN: Frequency Domain (Spectrum)", fontsize=14, fontweight='bold')
    # Target spectrum (black bars)
    ax1.vlines(target_peak_freqs, 0, target_peak_mags, color='black', linewidth=2.5, 
               alpha=0.7, label='Target', linestyles='solid')
    # TCN prediction spectrum (blue bars)
    ax1.vlines(tcn_peak_freqs, 0, tcn_peak_mags, color='blue', linewidth=2.0, 
               alpha=0.6, label='TCN Prediction', linestyles='dashed')
    
    # Set axis limits
    if len(target_peak_freqs) > 0:
        max_freq = min(5000, max(target_peak_freqs) * 1.2) if len(target_peak_freqs) > 0 else 5000
    else:
        max_freq = 5000
    ax1.set_xlim(0, max_freq)
    
    max_y = max(
        np.max(target_peak_mags) if len(target_peak_mags) > 0 else 0, 
        np.max(tcn_peak_mags) if len(tcn_peak_mags) > 0 else 0
    ) * 1.2
    ax1.set_ylim(0, max_y if max_y > 0 else 1.0)
    
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Amplitude (Normalized)", fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========== TCN: Time Domain (Right) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    # Select a segment for better visualization
    zoom_start = min_len // 4
    zoom_samples = 1000
    zoom_end = min(zoom_start + zoom_samples, min_len)
    zoom_time = time_axis[zoom_start:zoom_end]
    
    ax2.plot(zoom_time, target_np[zoom_start:zoom_end], 'k-', 
             label='Target', linewidth=1.5, alpha=0.8)
    ax2.plot(zoom_time, tcn_pred_np[zoom_start:zoom_end], 'b--', 
             label=f'TCN (MSE: {tcn_mse:.4e})', linewidth=1.5, alpha=0.8)
    ax2.set_title(f"TCN: Time Domain (Waveform) - NRMSE: {tcn_nrmse:.2f}%", 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle('TCN Model Performance - Frequency & Time Domain Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved TCN paper-style visualization: {save_path}")


def compute_metrics(pred, target):
    """Compute MSE and NRMSE (%)"""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(pred_np), len(target_np))
    pred_np = pred_np[:min_len]
    target_np = target_np[:min_len]
    
    # MSE
    mse = np.mean((pred_np - target_np) ** 2)
    
    # NRMSE (%)
    rmse = np.sqrt(mse)
    target_range = np.max(target_np) - np.min(target_np)
    if target_range > 0:
        nrmse_percent = (rmse / target_range) * 100
    else:
        nrmse_percent = 0.0
    
    return mse, nrmse_percent


def main():
    parser = ArgumentParser(description="Visualize TCN model - Paper Style (Frequency & Time Domain)")
    parser.add_argument("--tcn_checkpoint", type=str,
                        default="lightning_logs/bulk/14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32/lightning_logs/version_0/checkpoints/epoch=59-step=4379.ckpt",
                        help="Path to TCN checkpoint")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root directory of dataset")
    parser.add_argument("--output", type=str, default="paper_style_comparison.png",
                        help="Output filename for visualization")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Specific sample index (random if not specified)")
    parser.add_argument("--length", type=int, default=131072,
                        help="Sequence length for evaluation")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load TCN model
    if not os.path.exists(args.tcn_checkpoint):
        print(f"Error: TCN 체크포인트가 없습니다. 경로를 확인해주세요: {args.tcn_checkpoint}")
        return
    
    tcn_model = load_tcn_model(args.tcn_checkpoint)
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.root_dir}...")
    test_dataset = CustomWaveDataset(
        root_dir=args.root_dir,
        subset="test",
        length=args.length,
        preload=False,
        half=False,
        fraction=1.0
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Select random sample
    if args.sample_idx is None:
        sample_idx = np.random.randint(0, len(test_dataset))
    else:
        sample_idx = args.sample_idx
    
    print(f"\nSelected sample index: {sample_idx}")
    
    # Load audio sample
    input_audio, target_audio = test_dataset[sample_idx]
    
    if torch.cuda.is_available():
        input_audio = input_audio.cuda()
        target_audio = target_audio.cuda()
    
    input_audio = input_audio.unsqueeze(0)  # Add batch dimension
    
    # TCN inference
    print("\nRunning TCN inference...")
    with torch.no_grad():
        tcn_pred = tcn_model(input_audio, None)
    
    # Crop target to match TCN prediction
    if hasattr(tcn_model.hparams, 'causal') and tcn_model.hparams.causal:
        target_crop = causal_crop(target_audio.unsqueeze(0), tcn_pred.shape[-1])
    else:
        target_crop = center_crop(target_audio.unsqueeze(0), tcn_pred.shape[-1])
    
    # Compute metrics
    print("\nComputing metrics...")
    tcn_mse, tcn_nrmse = compute_metrics(tcn_pred, target_crop)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("TCN MODEL PERFORMANCE")
    print("=" * 50)
    print(f"{'Metric':<20} {'Value':<25}")
    print("-" * 50)
    print(f"{'MSE':<20} {tcn_mse:<25.6e}")
    print(f"{'NRMSE (%)':<20} {tcn_nrmse:<25.2f}")
    print("=" * 50)
    
    # Create paper-style visualization
    print("\nCreating paper-style visualization...")
    plot_tcn_paper_style(
        target_crop.squeeze().cpu().numpy(),
        tcn_pred.squeeze().cpu().numpy(),
        tcn_mse,
        tcn_nrmse,
        args.output,
        sample_rate=192000
    )
    
    print(f"\n✓ TCN paper-style visualization complete! Results saved to: {args.output}")
    print(f"  Sample index: {sample_idx}")
    print(f"  TCN MSE: {tcn_mse:.6e}, NRMSE: {tcn_nrmse:.2f}%")


if __name__ == "__main__":
    main()

