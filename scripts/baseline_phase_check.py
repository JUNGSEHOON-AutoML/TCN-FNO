#!/usr/bin/env python3
"""
baseline_phase_check.py
========================
베이스라인 TCN 모델의 위상(Phase) 추종 한계를 정량화·시각화하는 스크립트.

기능:
  - 학습된 TCN 체크포인트를 자동 탐색하거나 --checkpoint 인자로 지정
  - Test 데이터셋 전체에 대해 추론 후 콘솔에 집계 지표 출력
      · 시간 도메인 MSE (평균 / 표준편차)
      · 위상 편차 RMS, rad (평균 / 표준편차)
  - 시각화 (2행 1열 Matplotlib 레이아웃)
      · 행 0: 2~3개 세그먼트에 대한 시간 파형 오버레이 (Target vs Prediction)
      · 행 1: 해당 세그먼트들의 위상 편차 산점도
              (X축 = log-scale 주파수 Hz, Y축 = 절대 위상 편차 rad)

사용법:
  # 자동 탐색 (lightning_logs/bulk 하위 best ckpt)
  python baseline_phase_check.py --root_dir /userHome/userhome4/sehoon/TCN_FNO

  # 체크포인트 직접 지정
  python baseline_phase_check.py \\
      --checkpoint path/to/epoch=59.ckpt \\
      --root_dir   /userHome/userhome4/sehoon/TCN_FNO \\
      --output     baseline_phase_check.png \\
      --n_segments 3
"""

# ──────────────────────────────────────────────────────────────────────────────
# Stdlib / path 설정
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import glob
import math
import logging
from typing import List, Tuple, Dict
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
np.object = object
np.bool = bool
np.int = int
np.float = float
np.long = int
np.complex = complex
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 프로젝트 루트 설정 (microtcn 패키지 임포트용)
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from microtcn.tcn import TCNModel
from microtcn.data import CustomWaveDataset
from microtcn.utils import center_crop, causal_crop

# ──────────────────────────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════════

def find_best_checkpoint(model_dir: str):
    """lightning_logs/bulk 하위에서 가장 최근 epoch 체크포인트를 탐색한다."""
    pattern = os.path.join(model_dir, "lightning_logs", "bulk",
                           "**", "checkpoints", "*.ckpt")
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        # 대체: model_dir 전체 재귀 탐색
        pattern2 = os.path.join(model_dir, "**", "*.ckpt")
        ckpts = glob.glob(pattern2, recursive=True)
    if not ckpts:
        return None
    # epoch 번호가 가장 높은 것 우선
    def epoch_key(p):
        base = os.path.basename(p)
        if "epoch=" in base:
            try:
                return int(base.split("epoch=")[1].split("-")[0])
            except ValueError:
                pass
        return -1
    ckpts.sort(key=epoch_key, reverse=True)
    return ckpts[0]


def load_tcn(checkpoint_path: str) -> TCNModel:
    """체크포인트에서 TCN 모델을 로드한다."""
    log.info(f"TCN 체크포인트 로드: {checkpoint_path}")
    model = TCNModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.freeze()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def crop_target(target: torch.Tensor, pred: torch.Tensor,
                causal: bool) -> torch.Tensor:
    """pred 길이에 맞게 target을 크로핑한다."""
    if causal:
        return causal_crop(target, pred.shape[-1])
    return center_crop(target, pred.shape[-1])


# ══════════════════════════════════════════════════════════════════════════════
#  위상 편차 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_phase_diff_stft(
    target: torch.Tensor,
    pred:   torch.Tensor,
    n_fft:  int = 2048,
    hop_length: int = 512,
    sample_rate: int = 192000,
) -> Tuple[np.ndarray, np.ndarray]:
    """torch.stft 기반 주파수별 절대 위상 편차 계산.

    Args:
        target      : (T,) 1D CPU Tensor
        pred        : (T,) 1D CPU Tensor
        n_fft       : FFT 크기
        hop_length  : Hop 길이
        sample_rate : 샘플링 주파수 (주파수 축 생성용)

    Returns:
        freqs      : (F,)        주파수 배열 (Hz)
        abs_pd_rms : (F,)        주파수별 위상 편차 RMS (rad), 시간 평균
    """
    window = torch.hann_window(n_fft).to(target.device)

    def _stft(x):
        return torch.stft(
            x, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True, pad_mode="reflect",
        )

    spec_t = _stft(target.float())   # (F, T_frames)
    spec_p = _stft(pred.float())

    phase_t = torch.angle(spec_t)    # (F, T_frames)
    phase_p = torch.angle(spec_p)

    # [-π, π] 래핑
    phase_diff = (phase_p - phase_t + math.pi) % (2 * math.pi) - math.pi
    abs_pd = torch.abs(phase_diff)   # (F, T_frames)

    # 시간 축 평균 → (F,) RMS
    abs_pd_rms = torch.sqrt(torch.mean(abs_pd ** 2, dim=-1)).cpu().numpy()

    # 주파수 축 (0 ~ sample_rate/2)
    F_bins = n_fft // 2 + 1
    freqs  = np.linspace(0.0, sample_rate / 2, F_bins)

    return freqs, abs_pd_rms


def phase_rms_scalar(
    target: torch.Tensor,
    pred:   torch.Tensor,
    n_fft:  int = 2048,
    hop_length: int = 512,
) -> float:
    """전체 스펙트럼 + 전체 시간에 걸친 스칼라 위상 편차 RMS (rad)."""
    window = torch.hann_window(n_fft).to(target.device)

    def _stft(x):
        return torch.stft(
            x.float(), n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True, pad_mode="reflect",
        )

    spec_t = _stft(target)
    spec_p = _stft(pred)

    phase_diff = (torch.angle(spec_p) - torch.angle(spec_t) + math.pi) \
                 % (2 * math.pi) - math.pi
    return float(torch.sqrt(torch.mean(phase_diff ** 2)).item())


# ══════════════════════════════════════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════════════════════════════════════

# 세그먼트별 컬러 팔레트
_COLORS = ["#2196F3", "#E91E63", "#4CAF50"]

def plot_baseline_phase(
    segments: List[Dict],
    output_path: str,
    sample_rate: int = 192000,
    n_fft: int = 2048,
    hop_length: int = 512,
):
    """2행 1열 레이아웃 시각화.

    Args:
        segments    : [{"target": Tensor(T), "pred": Tensor(T), "label": str}, ...]
        output_path : 저장 경로
        sample_rate : 샘플링 주파수
        n_fft, hop_length : STFT 파라미터
    """
    n_seg = len(segments)
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0F1117")

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        hspace=0.45,
        height_ratios=[1, 1.1],
        left=0.08, right=0.97, top=0.91, bottom=0.07,
    )

    ax_wave  = fig.add_subplot(gs[0])   # 시간 파형
    ax_phase = fig.add_subplot(gs[1])   # 위상 편차 산점도

    for ax in (ax_wave, ax_phase):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")

    # ── 행 0: 시간 파형 오버레이 ────────────────────────────────────────────
    ax_wave.set_title("Time-Domain Waveform — Target vs TCN Prediction",
                      color="#EEEEEE", fontsize=12, pad=8)
    ax_wave.set_xlabel("Sample index", color="#AAAAAA", fontsize=9)
    ax_wave.set_ylabel("Amplitude",    color="#AAAAAA", fontsize=9)
    ax_wave.axhline(0, color="#555566", linewidth=0.5, linestyle="--")

    legend_handles = []
    for i, seg in enumerate(segments):
        tgt  = seg["target"].cpu().float().numpy()
        prd  = seg["pred"].cpu().float().numpy()
        lbl  = seg["label"]
        col  = _COLORS[i % len(_COLORS)]
        t_ax = np.arange(len(tgt))

        # Target — 실선
        lh_t, = ax_wave.plot(t_ax, tgt, color=col,
                             linewidth=1.2, alpha=0.85,
                             label=f"[{lbl}] Target")
        # Prediction — 파선
        lh_p, = ax_wave.plot(t_ax, prd, color=col,
                             linewidth=1.0, alpha=0.65,
                             linestyle="--", label=f"[{lbl}] Pred")
        legend_handles += [lh_t, lh_p]

    ax_wave.legend(handles=legend_handles,
                   facecolor="#252535", edgecolor="#555566",
                   labelcolor="#DDDDDD", fontsize=8,
                   ncol=min(n_seg * 2, 4),
                   loc="upper right")

    # ── 행 1: 위상 편차 산점도 ─────────────────────────────────────────────
    ax_phase.set_title("Phase Difference Scatter — Absolute Phase Error (rad) vs Frequency",
                       color="#EEEEEE", fontsize=12, pad=8)
    ax_phase.set_xlabel("Frequency (Hz)  [log scale]", color="#AAAAAA", fontsize=9)
    ax_phase.set_ylabel("Absolute Phase Diff (rad)",   color="#AAAAAA", fontsize=9)
    ax_phase.set_xscale("log")
    ax_phase.set_xlim(20, sample_rate / 2)
    ax_phase.set_ylim(0, math.pi + 0.1)
    ax_phase.axhline(math.pi / 2, color="#FF8800", linewidth=0.7,
                     linestyle=":", alpha=0.7, label="π/2")
    ax_phase.axhline(math.pi,     color="#FF4444", linewidth=0.7,
                     linestyle=":", alpha=0.7, label="π (worst)")
    ax_phase.grid(True, which="both", color="#333344", linewidth=0.4)

    for i, seg in enumerate(segments):
        tgt = seg["target"].cpu()
        prd = seg["pred"].cpu()
        col = _COLORS[i % len(_COLORS)]
        lbl = seg["label"]

        freqs, abs_pd_rms = compute_phase_diff_stft(
            tgt, prd, n_fft=n_fft, hop_length=hop_length,
            sample_rate=sample_rate,
        )
        # 20 Hz 이상만 표시
        mask = freqs >= 20
        ax_phase.scatter(
            freqs[mask], abs_pd_rms[mask],
            s=6, alpha=0.5, color=col, label=lbl, zorder=3,
        )

    ax_phase.legend(facecolor="#252535", edgecolor="#555566",
                    labelcolor="#DDDDDD", fontsize=9, loc="upper left")

    # ── 전체 타이틀 ──────────────────────────────────────────────────────────
    fig.suptitle(
        "TCN Baseline — Phase Limitation Verification",
        color="#FFFFFF", fontsize=14, fontweight="bold", y=0.96,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"시각화 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = ArgumentParser(
        description="베이스라인 TCN 모델의 위상 추종 한계 검증 스크립트"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="TCN 체크포인트 경로 (미지정 시 자동 탐색)",
    )
    parser.add_argument(
        "--root_dir", type=str,
        default=str(ROOT_DIR),
        help="프로젝트 루트 (데이터셋 & 체크포인트 탐색 기준)",
    )
    parser.add_argument(
        "--output", type=str, default="baseline_phase_check.png",
        help="시각화 저장 파일명",
    )
    parser.add_argument(
        "--length", type=int, default=16384,
        help="오디오 세그먼트 길이 (샘플)",
    )
    parser.add_argument(
        "--n_segments", type=int, default=3,
        help="시각화에 사용할 세그먼트 수 (1~3 권장)",
    )
    parser.add_argument(
        "--n_fft", type=int, default=2048,
        help="STFT FFT 크기",
    )
    parser.add_argument(
        "--hop_length", type=int, default=512,
        help="STFT Hop 길이",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=192000,
        help="오디오 샘플링 주파수",
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None,
        help="집계 평가에 사용할 최대 샘플 수 (None=전체)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="세그먼트 선택용 랜덤 시드",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. 체크포인트 탐색 ──────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_best_checkpoint(args.root_dir)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        log.error(
            "체크포인트를 찾을 수 없습니다. "
            "--checkpoint 인자로 직접 경로를 지정하거나, "
            "--root_dir 아래 lightning_logs/bulk 디렉토리를 확인하세요."
        )
        sys.exit(1)

    # ── 2. 모델 로드 ────────────────────────────────────────────────────────
    model = load_tcn(ckpt_path)
    is_causal = getattr(model.hparams, "causal", False)
    log.info(f"Causal 모드: {is_causal}")

    # ── 3. 데이터셋 로드 ────────────────────────────────────────────────────
    log.info("Test 데이터셋 로드 중...")
    dataset = CustomWaveDataset(
        root_dir            = args.root_dir,
        subset              = "test",
        length              = args.length,
        preload             = False,
        half                = False,
        fraction            = 1.0,
        augment             = False,   # 평가 시 증강 비활성
    )
    total_samples = len(dataset)
    log.info(f"Test 샘플 수: {total_samples}")

    if total_samples == 0:
        log.error("데이터셋에 샘플이 없습니다. --root_dir 경로를 확인하세요.")
        sys.exit(1)

    # ── 4. 시각화용 세그먼트 선택 ───────────────────────────────────────────
    n_seg = min(args.n_segments, total_samples, 3)
    viz_indices = np.random.choice(total_samples, size=n_seg, replace=False)
    log.info(f"시각화 세그먼트 인덱스: {viz_indices.tolist()}")

    # ── 5. 전체 Test 집계 평가 ───────────────────────────────────────────────
    eval_n = total_samples
    if args.max_eval_samples is not None:
        eval_n = min(args.max_eval_samples, total_samples)

    all_mse        = []
    all_phase_rms  = []

    log.info(f"집계 평가 시작 (총 {eval_n}개 샘플)...")

    for idx in range(eval_n):
        inp, tgt = dataset[idx]   # (1, T)

        inp_b = inp.unsqueeze(0).float().to(device)  # (1, 1, T)
        tgt_b = tgt.unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_b = model(inp_b, None)              # (1, 1, T')

        tgt_crop = crop_target(tgt_b, pred_b, is_causal)  # (1, 1, T')

        pred_1d = pred_b.squeeze()    # (T',)
        tgt_1d  = tgt_crop.squeeze()  # (T',)

        # MSE
        mse_val = float(F.mse_loss(pred_1d, tgt_1d).item())
        all_mse.append(mse_val)

        # Phase RMS (스칼라)
        p_rms = phase_rms_scalar(
            tgt_1d.cpu(), pred_1d.cpu(),
            n_fft=args.n_fft, hop_length=args.hop_length,
        )
        all_phase_rms.append(p_rms)

        if (idx + 1) % max(1, eval_n // 10) == 0:
            log.info(f"  진행: {idx+1}/{eval_n}  "
                     f"MSE={mse_val:.4e}  PhaseRMS={p_rms:.4f} rad")

    # ── 6. 집계 지표 출력 ───────────────────────────────────────────────────
    mse_arr  = np.array(all_mse)
    prms_arr = np.array(all_phase_rms)

    print()
    print("=" * 60)
    print("  TCN BASELINE — PHASE LIMITATION RESULTS")
    print("=" * 60)
    print(f"  체크포인트  : {ckpt_path}")
    print(f"  평가 샘플   : {eval_n} / {total_samples}")
    print(f"  세그먼트 길이: {args.length} samples  "
          f"({args.length / args.sample_rate * 1000:.1f} ms)")
    print("-" * 60)
    print(f"  {'지표':<28} {'평균':>10} {'표준편차':>10}")
    print("-" * 60)
    print(f"  {'Time-domain MSE':<28} {mse_arr.mean():>10.4e} "
          f"{mse_arr.std():>10.4e}")
    print(f"  {'Phase RMS (rad)':<28} {prms_arr.mean():>10.4f} "
          f"{prms_arr.std():>10.4f}")
    print("=" * 60)
    print(f"  위상 편차 해석: π/2 ≈ {math.pi/2:.4f} rad (90°)  "
          f"π ≈ {math.pi:.4f} rad (180°)")
    mean_prms = prms_arr.mean()
    if mean_prms > math.pi / 2:
        print("  ⚠  평균 위상 편차 > π/2 → 심각한 위상 왜곡")
    elif mean_prms > math.pi / 4:
        print("  ⚠  평균 위상 편차 > π/4 → 유의미한 위상 오차 존재")
    else:
        print("  ✓  평균 위상 편차 < π/4 → 상대적으로 양호")
    print("=" * 60)
    print()

    # ── 7. 시각화용 세그먼트 추론 ───────────────────────────────────────────
    segments = []
    for seg_i, idx in enumerate(viz_indices):
        inp, tgt = dataset[idx]
        inp_b = inp.unsqueeze(0).float().to(device)
        tgt_b = tgt.unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_b = model(inp_b, None)

        tgt_crop = crop_target(tgt_b, pred_b, is_causal)

        pred_1d = pred_b.squeeze().cpu()
        tgt_1d  = tgt_crop.squeeze().cpu()

        # 파형 시각화는 최대 4096 샘플만 사용 (가독성)
        vis_len = min(4096, len(tgt_1d))
        segments.append({
            "target": tgt_1d[:vis_len],
            "pred"  : pred_1d[:vis_len],
            "label" : f"seg{seg_i+1} (idx={idx})",
        })

    # ── 8. 시각화 ───────────────────────────────────────────────────────────
    log.info("시각화 생성 중...")
    plot_baseline_phase(
        segments    = segments,
        output_path = args.output,
        sample_rate = args.sample_rate,
        n_fft       = args.n_fft,
        hop_length  = args.hop_length,
    )

    print(f"✓ 시각화 저장 완료: {os.path.abspath(args.output)}")
    print("  실행 완료.")


if __name__ == "__main__":
    main()
