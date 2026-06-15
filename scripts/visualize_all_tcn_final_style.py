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
모든 TCN 모델에 대해 final_comparison.png 스타일 시각화 생성

각 모델마다 개별 이미지 생성:
- 주파수 도메인 (왼쪽)
- 시간 도메인 (오른쪽)
1x2 그리드 형식
"""

import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from argparse import ArgumentParser
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Set root directory
ROOT_DIR = Path("/userHome/userhome4/sehoon/micro-tcn-main/micro-tcn-main")
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from microtcn.tcn import TCNModel
from microtcn.data import CustomWaveDataset
from microtcn.utils import center_crop, causal_crop


def find_tcn_checkpoints(model_dir="./lightning_logs/bulk"):
    """Find all TCN model checkpoints (exclude LSTM)"""
    models = sorted(glob.glob(os.path.join(model_dir, "*")))
    models = [m for m in models if os.path.isdir(m)]
    
    checkpoints = []
    for model_dir_path in models:
        model_id = os.path.basename(model_dir_path)
        
        # Skip LSTM models
        if "LSTM" in model_id or "lstm" in model_id.lower():
            continue
        
        # Find checkpoint files
        checkpoint_candidates = glob.glob(os.path.join(
            model_dir_path, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        ))
        
        if len(checkpoint_candidates) == 0:
            print(f"⚠️  Skipping {model_id}: No checkpoint found")
            continue
        
        # Prefer epoch=59 checkpoint, otherwise use the last one
        epoch59_checkpoints = [c for c in checkpoint_candidates if "epoch=59" in c]
        if len(epoch59_checkpoints) > 0:
            checkpoint_path = epoch59_checkpoints[0]
        else:
            checkpoint_path = sorted(checkpoint_candidates)[-1]
        
        checkpoints.append((model_id, checkpoint_path))
    
    return checkpoints


def load_tcn_model(checkpoint_path):
    """Load TCN model from checkpoint"""
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


def visualize_tcn_model(target_np, tcn_pred_np, tcn_mse, tcn_nrmse,
                        model_name, save_path, sample_rate=192000):
    """final_comparison.png 스타일: 1x2 그리드 (주파수 도메인 + 시간 도메인)"""
    
    # Ensure same length
    min_len = min(len(target_np), len(tcn_pred_np))
    target_np = target_np[:min_len]
    tcn_pred_np = tcn_pred_np[:min_len]
    
    # Create time axis (in seconds)
    time_axis = np.arange(min_len) / sample_rate
    
    # Compute FFT spectra
    target_freqs, target_mags = compute_fft_spectrum(target_np, sample_rate)
    tcn_freqs, tcn_mags = compute_fft_spectrum(tcn_pred_np, sample_rate)
    
    # Extract dominant frequencies
    target_peak_freqs, target_peak_mags = extract_dominant_frequencies(target_freqs, target_mags, top_n=15)
    tcn_peak_freqs, tcn_peak_mags = extract_dominant_frequencies(tcn_freqs, tcn_mags, top_n=15)
    
    # Create figure with 1 row, 2 columns (Frequency and Time domain)
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3, width_ratios=[1, 1])
    
    # ========== Frequency Domain (Left) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Frequency Domain (Spectrum)", fontsize=14, fontweight='bold')
    # Target spectrum (black bars)
    ax1.vlines(target_peak_freqs, 0, target_peak_mags, color='black', linewidth=2.5, 
               alpha=0.7, label='Target', linestyles='solid')
    # TCN prediction spectrum (blue bars)
    ax1.vlines(tcn_peak_freqs, 0, tcn_peak_mags, color='blue', linewidth=2.0, 
               alpha=0.6, label='TCN Prediction', linestyles='dashed')
    
    # Set axis limits
    if len(target_peak_freqs) > 0:
        max_freq = min(5000, max(target_peak_freqs) * 1.2)
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
    
    # ========== Time Domain (Right) ==========
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
    ax2.set_title(f"Time Domain (Waveform) - NRMSE: {tcn_nrmse:.2f}%", 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Overall title with model name
    fig.suptitle(f'{model_name} - Frequency & Time Domain Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def main():
    parser = ArgumentParser(description="Generate final-style visualizations for all TCN models")
    parser.add_argument("--model_dir", type=str, default="./lightning_logs/bulk",
                       help="Directory containing model checkpoints")
    parser.add_argument("--root_dir", type=str, default=".",
                       help="Root directory of dataset")
    parser.add_argument("--output_dir", type=str, default="images/tcn_models",
                       help="Output directory for visualizations")
    parser.add_argument("--eval_subset", type=str, default="test",
                       help="Evaluation subset: 'test', 'train', or 'val'")
    parser.add_argument("--eval_length", type=int, default=131072,
                       help="Sequence length for evaluation")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Test sample index to use for visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all TCN checkpoints
    print("🔍 Finding TCN model checkpoints...")
    checkpoints = find_tcn_checkpoints(args.model_dir)
    
    if len(checkpoints) == 0:
        print("❌ No TCN checkpoints found!")
        return
    
    print(f"✓ Found {len(checkpoints)} TCN models\n")
    
    # Load test dataset
    print(f"📂 Loading test dataset (subset: {args.eval_subset})...")
    test_dataset = CustomWaveDataset(
        root_dir=args.root_dir,
        subset=args.eval_subset,
        length=args.eval_length,
        preload=False,
        half=False,
        fraction=1.0
    )
    print(f"✓ Test dataset size: {len(test_dataset)} samples")
    print(f"✓ Using sample index: {args.sample_idx}\n")
    
    # Load audio sample once
    input_audio, target_audio = test_dataset[args.sample_idx]
    
    if torch.cuda.is_available():
        input_audio = input_audio.cuda()
        target_audio = target_audio.cuda()
    
    input_audio = input_audio.unsqueeze(0)
    
    # Process each model
    for idx, (model_id, checkpoint_path) in enumerate(checkpoints):
        print(f"[{idx+1}/{len(checkpoints)}] Processing: {model_id}")
        
        try:
            # Load model
            model = load_tcn_model(checkpoint_path)
            
            # Inference
            with torch.no_grad():
                pred = model(input_audio, None)
            
            # Crop target to match prediction
            if hasattr(model.hparams, 'causal') and model.hparams.causal:
                target_crop = causal_crop(target_audio.unsqueeze(0), pred.shape[-1])
            else:
                target_crop = center_crop(target_audio.unsqueeze(0), pred.shape[-1])
            
            # Compute metrics
            mse, nrmse = compute_metrics(pred, target_crop)
            
            # Create visualization filename
            safe_model_id = model_id.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(args.output_dir, f"{safe_model_id}_final_style.png")
            
            # Create visualization
            visualize_tcn_model(
                target_crop.squeeze().cpu().numpy(),
                pred.squeeze().cpu().numpy(),
                mse,
                nrmse,
                model_id,
                output_path,
                sample_rate=192000
            )
            
            print(f"  MSE: {mse:.6e}, NRMSE: {nrmse:.2f}%\n")
            
            # Free GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue
    
    print(f"✅ All visualizations saved to: {args.output_dir}")
    print(f"   Total: {len(checkpoints)} models processed")


if __name__ == "__main__":
    main()

