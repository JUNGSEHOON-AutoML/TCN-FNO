#!/bin/bash
set -e

# Ensure correct conda environment python is used
export PATH="/userHome/userhome4/sehoon/miniconda3/envs/microtcn_env/bin:$PATH"

# Force physical GPU 1 usage via train.py argument instead of flaky environment variables
# export CUDA_VISIBLE_DEVICES=1

# Enable expandable segments to prevent CUDA memory fragmentation and OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [Step 1] Downloading Real Public Dataset ==="
# Dataset already downloaded and structured. Skipping.
# bash scripts/download_dataset.sh

echo "=== [Step 2] Training TCN Model ==="
# TCN model training completed successfully in the previous run.
# Checkpoint is saved at: lightning_logs/bulk/14-uTCN__noncausal__10-2-15__fraction-1.0-bs32__loss-advanced/
# python scripts/train.py --config_idx 14 --loss advanced --augment True --lambda_harm 1e-4 --lambda_phase 1e-5 --max_epochs 30 --limit_val_batches 100 --num_workers 4

echo "=== [Step 3] Training FNO Model ==="
# Train FNO model for 30 epochs under config 14 parameter constraints.
# --limit_val_batches 100 limits validation loop steps to speed up epoch runs.
# --num_workers 4 reduces worker process overhead to prevent swap/OOM crashes.
# --target_gpu 1 explicitly specifies physical GPU 1 to bypass driver mapping bugs.
# --batch_size 4 limits VRAM memory footprints to prevent CUDA OOM on shared GPU.
python scripts/train.py --config_idx 14 --model_type fno --loss advanced --augment True --lambda_harm 1e-4 --lambda_phase 1e-5 --max_epochs 30 --limit_val_batches 100 --num_workers 4 --target_gpu 1 --batch_size 4

echo "=== [Step 4] Generating Final Paper-style Visualization ==="
python scripts/run_evaluation.py

echo "=== All Done! Check paper_style_comparison.png ==="
