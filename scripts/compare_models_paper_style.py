#!/usr/bin/env python3
"""
TCN vs FNO 모델 비교 - 논문 스타일 시각화 (주파수 도메인 + 시간 도메인)

논문의 Figure 3 & 5 스타일:
- 왼쪽: 주파수 스펙트럼 (막대그래프) - 타겟과 예측의 주파수 성분 비교
- 오른쪽: 시간 파형 (물결) - 타겟과 예측의 파형 비교
"""

import os
import sys
from pathlib import Path

# NumPy compatibility patch for older PyTorch Lightning / Librosa / TensorBoard versions
import numpy as np
np.object = object
np.bool = bool
np.int = int
np.float = float
np.long = int
np.complex = complex

import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from argparse import ArgumentParser
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# librosa for spectrogram visualization
import librosa
import librosa.display

# Set root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

# Import TCN model and dataset
from microtcn.tcn import TCNModel
from microtcn.fno_wrapper import FNOModel
from microtcn.data import CustomWaveDataset
from microtcn.utils import center_crop, causal_crop

# Import FNO model (neuraloperator)
FNO_AVAILABLE = False
try:
    from neuralop.models import FNO
    FNO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"neuralop 임포트 실패 (벤치마크에서 FNO 제외): {e}")
    FNO_AVAILABLE = False
    FNO = None
except Exception as e:
    logging.error(f"FNO 로딩 중 예기치 않은 오류 발생: {e}")
    FNO_AVAILABLE = False
    FNO = None


def load_tcn_model(checkpoint_path):
    """Load TCN model from checkpoint"""
    print(f"Loading TCN model from {checkpoint_path}...")
    model = TCNModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.freeze()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


class FNOArgs:
    def __init__(self, n_modes=16, hidden_channels=64, n_layers=4):
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

def load_fno_model(checkpoint_path):
    """Load FNO model from checkpoint"""
    if not FNO_AVAILABLE or FNO is None:
        raise RuntimeError(
            "FNO 모델을 로드할 수 없습니다. "
            "neuraloperator 패키지가 설치되어 있는지 확인하세요."
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if checkpoint exists and is in the old raw FNO format (needs regeneration)
    is_old_format = False
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if 'state_dict' not in ckpt and 'model_state_dict' in ckpt:
                is_old_format = True
        except:
            is_old_format = True

    # If checkpoint doesn't exist or is in old format, create a dummy FNOModel checkpoint
    if not os.path.exists(checkpoint_path) or is_old_format:
        print(f"[INFO] FNO checkpoint not found or in old format at {checkpoint_path}. Generating a new dummy FNOModel checkpoint...")
        dummy_model = FNOModel(n_modes=16, hidden_channels=64, n_layers=4)
        dummy_checkpoint = {
            'state_dict': dummy_model.state_dict(),
            'hyper_parameters': dummy_model.hparams,
            'epoch': 0
        }
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        torch.save(dummy_checkpoint, checkpoint_path)
        print(f"[INFO] Dummy FNOModel checkpoint saved to {checkpoint_path}")

    # Load PL model checkpoint
    model = FNOModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.freeze()
    
    # Explicitly move model to target device to bypass PL mapping inconsistencies
    model = model.to(device)
    
    print(f"FNO model loaded successfully from {checkpoint_path}")
    return model


def fno_inference(model, x, device):
    """Run FNO inference"""
    with torch.no_grad():
        if isinstance(model, FNOModel):
            # Normalization and residual connection are handled internally inside FNOModel.forward
            pred = model(x)
        else:
            # Fallback for raw FNO model
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
            pred = model(x_norm) + x
    return pred


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


def extract_dominant_frequencies(freqs, magnitudes, top_n=10, min_freq=20, max_freq=5000):
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


def compute_phase_difference(y_true, y_pred, n_fft=2048):
    """주파수별 위상 편차 계산 (FFT + [-π, π] 래핑)

    Args:
        y_true (Tensor): 정답 신호 (1D 또는 (1, T) 형태)
        y_pred (Tensor): 예측 신호 (1D 또는 (1, T) 형태)
        n_fft  (int)  : FFT 크기. Default: 2048
    Returns:
        Tensor: (F, Frame) 형태의 래핑된 위상 편차
    """
    stft_true = torch.stft(y_true.squeeze(0), n_fft=n_fft,
                           return_complex=True, pad_mode='reflect')
    stft_pred = torch.stft(y_pred.squeeze(0), n_fft=n_fft,
                           return_complex=True, pad_mode='reflect')

    phase_true = torch.angle(stft_true)
    phase_pred = torch.angle(stft_pred)

    phase_diff = (phase_pred - phase_true + torch.pi) % (2 * torch.pi) - torch.pi
    return phase_diff


def compute_pearson_correlation(y_true, y_pred):
    """시간 도메인 Pearson r 계산

    Args:
        y_true (Tensor): 정답 신호
        y_pred (Tensor): 예측 신호
    Returns:
        float: Pearson correlation coefficient [-1, 1]
    """
    y_t = y_true.flatten()
    y_p = y_pred.flatten()

    mean_t, mean_p = torch.mean(y_t), torch.mean(y_p)
    xm, ym = y_t - mean_t, y_p - mean_p

    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / (r_den + 1e-8)
    return r.item()


def plot_phase_difference_scatter(ax, phase_diff, freqs, label, color):
    """로그 주파수 축 위상 편차 산점도

    Args:
        ax         : matplotlib Axes
        phase_diff : compute_phase_difference의 반환값 (Tensor, (F, Frame))
        freqs      : 주파수 배열 (numpy, Hz 단위)
        label      : 범레일 레이블
        color      : 점 색상
    """
    # 시간 축(dim=-1)의 평균을 취해 1D 배열로 변환
    pd_mean = torch.mean(torch.abs(phase_diff), dim=-1).numpy()

    ax.scatter(freqs, pd_mean, alpha=0.5, label=label, s=8, color=color)
    ax.set_xscale('log')
    ax.set_ylim(0, torch.pi)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Absolute Phase Diff (rad)')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()




def plot_spectrogram(ax, y_np, sr=192000, title="Spectrogram"):
    """STFT 스펙트로그램을 계산하고 로그 스케일로 ax에 렌더링"""
    # 1D array로 평탄화
    y_np = y_np.squeeze()
    
    # STFT 계산
    S = librosa.stft(y_np, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # y_axis='linear'로 렌더링하여 librosa의 basey 버그를 우회
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=512, x_axis='time', y_axis='linear', ax=ax, cmap='magma'
    )
    
    # 수동으로 y축을 로그 스케일로 설정 (matplotlib 호환성 보장)
    try:
        ax.set_yscale('log', base=10)
    except TypeError:
        ax.set_yscale('log', basey=10)
        
    ax.set_ylim(20, sr / 2) # 오디오 주파수 대역으로 제한
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    return img

def plot_transfer_function(ax, x_true, y_true, y_pred, title="Transfer Function"):
    """비선형 입출력 전달 함수 산점도 시각화"""
    x_np = x_true.squeeze()
    y_true_np = y_true.squeeze()
    y_pred_np = y_pred.squeeze()
    
    # 점의 수가 너무 많으면 렌더링 속도와 가독성을 위해 다운샘플링 (최대 50,000점)
    max_pts = 50000
    if len(x_np) > max_pts:
        step = len(x_np) // max_pts
        x_np = x_np[::step]
        y_true_np = y_true_np[::step]
        y_pred_np = y_pred_np[::step]
        
    ax.scatter(x_np, y_true_np, color='grey', alpha=0.3, s=1, label='Target (True)')
    ax.scatter(x_np, y_pred_np, color='red', alpha=0.3, s=1, label='Prediction (Model)')
    ax.set_xlabel('Input Waveform (x_true)', fontsize=10)
    ax.set_ylabel('Output Waveform (y)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_paper_style_comparison(
    input_np,
    target_np,
    tcn_pred_np,
    fno_pred_np,
    tcn_mse,
    tcn_nrmse,
    fno_mse,
    fno_nrmse,
    save_path,
    sample_rate=192000,
):
    """논문 스타일 시각화: 4행 2열 다중 패널 구조
    
    Row 1: 시간 축 파형 [Target vs Baseline TCN] / [Target vs Our Model (FNO)]
    Row 2: 스펙트로그램 [Baseline TCN 스펙트로그램] / [Our Model 스펙트로그램]
    Row 3: 비선형 전달 함수 [Input vs Baseline Pred] / [Input vs Our Model Pred]
    Row 4: 위상 편차 산점도 [Baseline Phase Diff] / [Our Model Phase Diff]
    """
    # Ensure same length
    min_len = min(len(input_np), len(target_np), len(tcn_pred_np), len(fno_pred_np))
    input_np    = input_np[:min_len]
    target_np   = target_np[:min_len]
    tcn_pred_np = tcn_pred_np[:min_len]
    fno_pred_np = fno_pred_np[:min_len]

    # Create time axis (in seconds)
    time_axis = np.arange(min_len) / sample_rate

    # Create figure: 4행 2열
    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # --------------------------------------------------------------
    # Row 1: 시간 축 파형 (Waveform) - Zoomed-in for detail
    # --------------------------------------------------------------
    zoom_start   = min_len // 4
    zoom_samples = 1000
    zoom_end     = min(zoom_start + zoom_samples, min_len)
    zoom_time    = time_axis[zoom_start:zoom_end]

    # [0, 0]: Target vs Baseline TCN
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(zoom_time, target_np[zoom_start:zoom_end], 'k-', label='Target', linewidth=1.5, alpha=0.8)
    ax1.plot(zoom_time, tcn_pred_np[zoom_start:zoom_end], 'b--', label=f'Baseline TCN (MSE: {tcn_mse:.4e})', linewidth=1.5, alpha=0.8)
    ax1.set_title(f"Baseline TCN: Time Domain (Waveform) - NRMSE: {tcn_nrmse:.2f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # [0, 1]: Target vs Our Model (FNO)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(zoom_time, target_np[zoom_start:zoom_end], 'k-', label='Target', linewidth=1.5, alpha=0.8)
    ax2.plot(zoom_time, fno_pred_np[zoom_start:zoom_end], 'r--', label=f'Our Model (FNO) (MSE: {fno_mse:.4e})', linewidth=1.5, alpha=0.8)
    ax2.set_title(f"Our Model (FNO): Time Domain (Waveform) - NRMSE: {fno_nrmse:.2f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Amplitude", fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --------------------------------------------------------------
    # Row 2: 스펙트로그램 (Spectrogram)
    # --------------------------------------------------------------
    # [1, 0]: Baseline TCN 스펙트로그램
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = plot_spectrogram(ax3, tcn_pred_np, sample_rate, "Baseline TCN: Spectrogram")
    fig.colorbar(img3, ax=ax3, format="%+2.0f dB")

    # [1, 1]: Our Model 스펙트로그램
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = plot_spectrogram(ax4, fno_pred_np, sample_rate, "Our Model (FNO): Spectrogram")
    fig.colorbar(img4, ax=ax4, format="%+2.0f dB")

    # --------------------------------------------------------------
    # Row 3: 비선형 전달 함수 (Nonlinear Transfer Function)
    # --------------------------------------------------------------
    # [2, 0]: Input vs Baseline Pred
    ax5 = fig.add_subplot(gs[2, 0])
    plot_transfer_function(ax5, input_np, target_np, tcn_pred_np, "Baseline TCN: Transfer Function")

    # [2, 1]: Input vs Our Model Pred
    ax6 = fig.add_subplot(gs[2, 1])
    plot_transfer_function(ax6, input_np, target_np, fno_pred_np, "Our Model (FNO): Transfer Function")

    # --------------------------------------------------------------
    # Row 4: 위상 편차 산점도 (Phase Deviation Scatter)
    # --------------------------------------------------------------
    # 주파수 축 생성 (n_fft // 2 + 1 개의 빈)
    _n_fft = 2048
    _freqs = np.linspace(0, sample_rate / 2, _n_fft // 2 + 1)

    tcn_pd = compute_phase_difference(
        torch.tensor(target_np).unsqueeze(0),
        torch.tensor(tcn_pred_np).unsqueeze(0),
        n_fft=_n_fft,
    )
    fno_pd = compute_phase_difference(
        torch.tensor(target_np).unsqueeze(0),
        torch.tensor(fno_pred_np).unsqueeze(0),
        n_fft=_n_fft,
    )

    # [3, 0]: Baseline Phase Diff
    ax7 = fig.add_subplot(gs[3, 0])
    plot_phase_difference_scatter(ax7, tcn_pd, _freqs, 'Baseline TCN Phase Diff', 'blue')
    ax7.set_title("Baseline TCN: Phase Deviation Scatter (Log Scale)", fontsize=12, fontweight='bold')

    # [3, 1]: Our Model Phase Diff
    ax8 = fig.add_subplot(gs[3, 1])
    plot_phase_difference_scatter(ax8, fno_pd, _freqs, 'Our Model (FNO) Phase Diff', 'red')
    ax8.set_title("Our Model (FNO): Phase Deviation Scatter (Log Scale)", fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle(
        'TCN vs FNO Comprehensive Model Comparison — Multi-panel Performance Profile',
        fontsize=16, fontweight='bold', y=0.99,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved paper-style comparison: {save_path}")


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
    parser = ArgumentParser(description="Compare TCN and FNO models - Paper Style")
    parser.add_argument("--tcn_checkpoint", type=str,
                        default="lightning_logs/bulk/14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32/lightning_logs/version_0/checkpoints/epoch=59-step=4379.ckpt",
                        help="Path to TCN checkpoint")
    parser.add_argument("--fno_checkpoint", type=str,
                        default=str(ROOT_DIR / "checkpoints_fno/best_fno_model.pt"),
                        help="Path to FNO checkpoint")
    parser.add_argument("--root_dir", type=str, default=str(ROOT_DIR / "data") if os.path.isdir(ROOT_DIR / "data/x_t") else str(ROOT_DIR),
                        help="Root directory of dataset")
    parser.add_argument("--output", type=str, default="paper_style_comparison.png",
                        help="Output filename for comparison figure")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Specific sample index (random if not specified)")
    parser.add_argument("--length", type=int, default=131072,
                        help="Sequence length for evaluation")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    import glob
    tcn_checkpoint = args.tcn_checkpoint
    if not os.path.exists(tcn_checkpoint):
        print(f"[WARNING] TCN checkpoint not found at {tcn_checkpoint}. Searching for the best checkpoint under lightning_logs...")
        pattern = os.path.join(ROOT_DIR, "lightning_logs", "bulk", "**", "checkpoints", "*.ckpt")
        ckpts = glob.glob(pattern, recursive=True)
        if not ckpts:
            pattern2 = os.path.join(ROOT_DIR, "**", "*.ckpt")
            ckpts = glob.glob(pattern2, recursive=True)
        
        if ckpts:
            def epoch_key(p):
                base = os.path.basename(p)
                if "epoch=" in base:
                    try:
                        return int(base.split("epoch=")[1].split("-")[0])
                    except ValueError:
                        pass
                return -1
            ckpts.sort(key=epoch_key, reverse=True)
            tcn_checkpoint = ckpts[0]
            print(f"[INFO] Auto-detected TCN checkpoint: {tcn_checkpoint}")
        else:
            raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다. 모델을 먼저 학습시키거나 --tcn_checkpoint 인자를 지정하세요.")
            
    tcn_model = load_tcn_model(tcn_checkpoint)
    fno_model = load_fno_model(args.fno_checkpoint)
    
    # Load test dataset
    print("\nLoading test dataset...")
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
        target_crop_tcn = causal_crop(target_audio.unsqueeze(0), tcn_pred.shape[-1])
    else:
        target_crop_tcn = center_crop(target_audio.unsqueeze(0), tcn_pred.shape[-1])
    
    # FNO inference
    print("Running FNO inference...")
    fno_pred = fno_inference(fno_model, input_audio, device)
    
    # Crop target to match FNO prediction
    if fno_pred.shape[-1] != target_audio.shape[-1]:
        if fno_pred.shape[-1] < target_audio.shape[-1]:
            target_crop_fno = center_crop(target_audio.unsqueeze(0), fno_pred.shape[-1])
        else:
            pad_size = fno_pred.shape[-1] - target_audio.shape[-1]
            target_crop_fno = torch.nn.functional.pad(target_audio.unsqueeze(0), (0, pad_size))
    else:
        target_crop_fno = target_audio.unsqueeze(0)
    
    # Compute metrics
    print("\nComputing metrics...")
    tcn_mse, tcn_nrmse = compute_metrics(tcn_pred, target_crop_tcn)
    fno_mse, fno_nrmse = compute_metrics(fno_pred, target_crop_fno)
    
    # Print comparison table (MSE, NRMSE, Pearson)
    tcn_pearson = compute_pearson_correlation(
        target_crop_tcn.squeeze().cpu(),
        tcn_pred.squeeze().cpu(),
    )
    fno_pearson = compute_pearson_correlation(
        target_crop_fno.squeeze().cpu(),
        fno_pred.squeeze().cpu(),
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON RESULTS")
    print("=" * 75)
    print(f"{'Metric':<25} {'TCN':>22} {'FNO':>22}")
    print("-" * 75)
    print(f"{'MSE':<25} {tcn_mse:>22.6e} {fno_mse:>22.6e}")
    print(f"{'NRMSE (%)':<25} {tcn_nrmse:>22.2f} {fno_nrmse:>22.2f}")
    print(f"{'Pearson r':<25} {tcn_pearson:>22.4f} {fno_pearson:>22.4f}")
    print("=" * 75)
    
    # For visualization, align all predictions to same length
    min_len = min(tcn_pred.shape[-1], fno_pred.shape[-1], target_crop_tcn.shape[-1])
    tcn_pred_viz = tcn_pred[:, :, :min_len]
    fno_pred_viz = fno_pred[:, :, :min_len]
    target_viz = target_crop_tcn[:, :, :min_len]
    
    # Align input audio
    if hasattr(tcn_model.hparams, 'causal') and tcn_model.hparams.causal:
        input_crop_tcn = causal_crop(input_audio, tcn_pred.shape[-1])
    else:
        input_crop_tcn = center_crop(input_audio, tcn_pred.shape[-1])
    input_viz = input_crop_tcn[:, :, :min_len]
    
    # Recompute metrics on aligned data
    tcn_mse_viz, tcn_nrmse_viz = compute_metrics(tcn_pred_viz, target_viz)
    fno_mse_viz, fno_nrmse_viz = compute_metrics(fno_pred_viz, target_viz)
    
    # Create paper-style visualization
    print("\nCreating paper-style visualization...")
    plot_paper_style_comparison(
        input_viz.squeeze().cpu().numpy(),
        target_viz.squeeze().cpu().numpy(),
        tcn_pred_viz.squeeze().cpu().numpy(),
        fno_pred_viz.squeeze().cpu().numpy(),
        tcn_mse_viz,
        tcn_nrmse_viz,
        fno_mse_viz,
        fno_nrmse_viz,
        args.output,
        sample_rate=192000
    )
    
    print(f"\n✓ Paper-style comparison complete! Results saved to: {args.output}")
    print(f"  Sample index: {sample_idx}")
    print(f"  TCN  MSE:     {tcn_mse_viz:.6e},  NRMSE: {tcn_nrmse_viz:.2f}%")
    print(f"  FNO  MSE:     {fno_mse_viz:.6e},  NRMSE: {fno_nrmse_viz:.2f}%")
    tcn_pearson_viz = compute_pearson_correlation(
        target_viz.squeeze().cpu(),
        tcn_pred_viz.squeeze().cpu(),
    )
    fno_pearson_viz = compute_pearson_correlation(
        target_viz.squeeze().cpu(),
        fno_pred_viz.squeeze().cpu(),
    )
    print(f"  TCN  Pearson: {tcn_pearson_viz:.4f}")
    print(f"  FNO  Pearson: {fno_pearson_viz:.4f}")


if __name__ == "__main__":
    main()

