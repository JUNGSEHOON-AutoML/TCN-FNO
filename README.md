# TCN vs FNO: Audio Modeling with Advanced Spectral Loss

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2102.06200)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Effects-blue)](https://csteinmetz1.github.io/tcn-audio-effects/)

This repository contains code for modeling non-linear loudspeaker distortions and analog audio effects using **Temporal Convolutional Networks (TCN)** and **Fourier Neural Operators (FNO)**. It introduces **AdvancedSpectralLoss**, a loss function regularized with complex STFT magnitude and group-delay penalty terms to solve phase tracking limitations in neural audio models.

---

## 📁 Repository Structure

```
TCN_FNO_Audio/
├── microtcn/               # Core packages and neural network modules
│   ├── __init__.py
│   ├── base.py             # Base Lightning Module, AdvancedSpectralLoss
│   ├── data.py             # CustomWaveDataset with volume augmentation
│   ├── lstm.py             # LSTM baseline model
│   ├── tcn.py              # TCN model implementation
│   └── utils.py            # Signal processing utility functions
├── scripts/                # Execution and evaluation scripts
│   ├── download_dataset.sh          # Dataset preparation script
│   ├── generate_synthetic_data.py   # Synthetic 192kHz wav generator
│   ├── baseline_phase_check.py      # Phase 1: Baseline phase deviation check
│   ├── train.py                     # Phase 2: Train TCN/LSTM models
│   ├── compare_models_paper_style.py# Phase 3: Benchmark TCN vs FNO
│   ├── evaluate_models.py           # Evaluate a single model checkpoint
│   ├── evaluate_all_models.py       # Batch evaluate all checkpoints
│   ├── find_best_tcn.py             # Helper to select best TCN model
│   └── plot.py / speed.py           # Benchmarking & plotting utilities
├── requirements.txt        # Conflict-free dependency definitions
├── setup.py                # Python package setup
└── README.md               # Setup and execution guide
```

---

## 🚀 Quick Start (Step-by-Step Scenario)

Follow these steps to clone the repository, install dependencies, prepare the data, and run the entire evaluation/training pipeline.

### 1. Clone the Repository
```bash
git clone https://github.com/[YourUsername]/TCN_FNO_Audio.git
cd TCN_FNO_Audio
```

### 2. Install Dependencies
Install verified, conflict-free dependency versions:
```bash
pip install -r requirements.txt
```
*Note: This repository requires `numba>=0.57.0` and `numpy==1.24.4` to avoid Numba runtime crashes.*

### 3. Prepare the Dataset
You can run the interactive setup script:
```bash
bash scripts/download_dataset.sh
```
It will ask if you want to download the full ~20GB Zenodo dataset or generate a small **synthetic dataset** under `data/x_t` and `data/y_t` for rapid testing (sanity check runs in <10 seconds).

---

## 🔍 Execution Phases

### Phase 1: Verify Baseline Phase Limitations
Identify and report phase tracking limitations of the baseline TCN model on the test dataset:
```bash
python scripts/baseline_phase_check.py --max_eval_samples 50
```
This will automatically find the best trained checkpoint under `lightning_logs/bulk/`, run inference, compute average phase difference (RMS, radians), and save a visualization plot (Target vs. Prediction waveform overlay and frequency-dependent phase deviation).

### Phase 2: Train with Advanced Spectral Loss
Train TCN/LSTM models using the new **AdvancedSpectralLoss** ($Loss = L_{time} + \lambda_{harm}L_{harm} + \lambda_{phase}L_{phase}$) with volume augmentation:
```bash
python scripts/train.py --loss advanced --lambda_harm 1e-4 --lambda_phase 1e-5 --augment True
```
*   `--loss advanced`: Activates `AdvancedSpectralLoss` featuring a Group-delay regularization penalty.
*   `--augment True`: Turns on volume scale augmentation to help the network learn level-dependent distortions.

### Phase 3: Final TCN vs FNO Benchmark
Run the final side-by-side performance comparison (benchmarking TCN vs. Fourier Neural Operator):
```bash
python scripts/compare_models_paper_style.py
```
This script leverages the standard `neuraloperator` library, compares inference quality (MSE, Phase RMS, Spectral distortion), and logs performance metrics.

---

## 📊 Results & Visualizations

### 🔍 1. Baseline Phase Limitation Check
By running the Phase 1 validation script, you can visualize the phase tracking errors of the baseline models. Standard time-domain loss functions often result in poor phase alignment, especially in high-frequency ranges.

<p align="center">
  <img src="baseline_phase_check.png" alt="TCN Baseline Phase Check" width="850">
</p>

*   **Waveform Overlay**: Shows the timing/phase deviation between target and predicted signals.
*   **Phase Deviation vs. Frequency**: Illustrates how phase tracking error diverges at higher frequencies.

### 🏆 2. TCN vs FNO Benchmarking (Advanced Spectral Loss)
Comparing FNO and TCN models trained with `AdvancedSpectralLoss` reveals superior performance across waveform, harmonics, transfer function (hysteresis), and phase tracking.

<p align="center">
  <img src="paper_style_comparison.png" alt="TCN vs FNO Benchmark Comparison" width="850">
</p>

The multi-panel visualization provides a comprehensive evaluation:
1.  **Waveform Overlay (Row 1)**: Zoomed-in sample-level target vs. prediction comparison.
2.  **Log-Spectrogram (Row 2)**: Evaluates the accuracy of harmonic tracking across the log-frequency spectrum.
3.  **Transfer Function & Hysteresis (Row 3)**: Illustrates how well the models capture non-linear distortion curves and memory-dependent hysteresis loops.
4.  **Phase Deviation (Row 4)**: Shows the phase error distribution over the frequency range.

---

## 🔬 Loss Function Details

**AdvancedSpectralLoss** combines:
1. **$L_{time}$**: L1 loss in the time-domain for waveform reconstruction.
2. **$L_{harm}$**: Complex STFT magnitude L1 loss for spectral shape matching.
3. **$L_{phase}$**: Group-delay smoothing penalty (1st-order frequency derivative of the phase difference) to force predicted phase gradients to match the target.

---

## 📄 Citation

If you use this codebase or models in your research, please cite our work:

```bibtex
@inproceedings{steinmetz2022efficient,
    title={Efficient neural networks for real-time modeling of analog dynamic range compression},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={152nd AES Convention},
    year={2022}
}
```
