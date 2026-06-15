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

import os
import sys
import glob
import json
import torch
import pickle
import torchaudio
import numpy as np
import torchsummary
from thop import profile
import pyloudnorm as pyln
import pytorch_lightning as pl
from argparse import ArgumentParser

import auraloss

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset, CustomWaveDataset
from microtcn.utils import center_crop, causal_crop

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='.')
parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--preload', action="store_true", default=False)
parser.add_argument('--half', action="store_true", default=False)
parser.add_argument('--fast', action="store_true", default=False) # skip LSTM
parser.add_argument('--sample_rate', type=int, default=192000)
parser.add_argument('--eval_subset', type=str, default='test', 
                    help='Evaluation subset: "test" (20%% of data, recommended) or "val"/"train"')
parser.add_argument('--eval_length', type=int, default=8388608)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=32)

# parse them args
args = parser.parse_args()

# set the seed
pl.seed_everything(42)

# setup the dataloaders
# Use CustomWaveDataset for x_t/y_t format (with train/test split)
test_dataset = CustomWaveDataset(args.root_dir, 
                                 subset=args.eval_subset,  # "test" for final evaluation (20% of data)
                                 half=False,
                                 preload=args.preload,
                                 length=args.eval_length,
                                 target_sample_rate=args.sample_rate)

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               shuffle=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

overall_results = {}

if args.save_dir is not None:
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

# set up loss functions for evaluation
l1   = torch.nn.L1Loss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(192000)

models = sorted(glob.glob(os.path.join(args.model_dir, "*")))
# Filter out files, only keep directories
models = [m for m in models if os.path.isdir(m)]

print(f"\n{'='*80}")
print(f"평가할 모델 디렉토리 찾기: {args.model_dir}")
print(f"총 {len(models)}개의 모델 디렉토리를 찾았습니다.")
print(f"{'='*80}\n")

for idx, model_dir in enumerate(models):

    results = {}

    # Find checkpoint file - try version_0 first, then any version
    checkpoint_candidates = glob.glob(os.path.join(model_dir,
                                                   "lightning_logs",
                                                   "version_*",
                                                   "checkpoints",
                                                   "*.ckpt"))
    
    if len(checkpoint_candidates) == 0:
        print(f"Skipping {os.path.basename(model_dir)}: No checkpoint found")
        continue
    
    # Prefer epoch=59 checkpoint, otherwise use the last one
    epoch59_checkpoints = [c for c in checkpoint_candidates if "epoch=59" in c]
    if len(epoch59_checkpoints) > 0:
        checkpoint_path = epoch59_checkpoints[0]
    else:
        checkpoint_path = sorted(checkpoint_candidates)[-1]  # Use last checkpoint
    
    hparams_file = os.path.join(model_dir, "hparams.yaml")

    model_id = os.path.basename(model_dir)
    
    # Parse model directory name more safely
    try:
        parts = model_id.split('-')
        if len(parts) > 1:
            model_type = parts[1]
        else:
            model_type = "UNKNOWN"
        
        # Try to extract batch_size from directory name (format: ...-bs32)
        batch_size = 32  # Default batch size
        if len(parts) > 0:
            last_part = parts[-1]
            if last_part.startswith('bs'):
                try:
                    batch_size = int(last_part[2:])
                except (ValueError, IndexError):
                    pass
    except (ValueError, IndexError, AttributeError):
        model_type = "UNKNOWN"
        batch_size = 32  # Default
    
    # Extract epoch number from checkpoint filename
    try:
        epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])
    except (ValueError, IndexError):
        epoch = 0  # Default if cannot parse

    if model_type == "LSTM":
        if args.fast: continue
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    i = torch.rand(1,1,65536)
    p = torch.rand(1,1,2)
    #macs, params = profile(model, inputs=(i, p))

    print(f"\n{'='*80}")
    print(f"[{idx+1}/{len(models)}] 평가 중: {model_id}")
    print(f"  체크포인트: {os.path.basename(checkpoint_path)}")
    print(f"  Epoch: {epoch}, Batch size: {batch_size}, Model type: {model_type}")
    print(f"{'='*80}")
    #print(   f"MACs: {macs/10**9:0.2f} G     Params: {params/1e3:0.2f} k")

    model.cuda()
    model.eval()

    if args.half:
        model.half()

    # set the seed
    pl.seed_everything(42)

    for bidx, batch in enumerate(test_dataloader):

        sys.stdout.write(f" Evaluating {bidx}/{len(test_dataloader)}...\r")
        sys.stdout.flush()

        input, target = batch  # CustomWaveDataset returns only (input, target), no params

        # move to gpu
        input = input.to("cuda:0")
        target = target.to("cuda:0")

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input, None)  # No conditioning parameters for CustomWaveDataset

            # crop the input and target signals
            if model.hparams.causal:
                input_crop = causal_crop(input, output.shape[-1])
                target_crop = causal_crop(target, output.shape[-1])
            else:
                input_crop = center_crop(input, output.shape[-1])
                target_crop = center_crop(target, output.shape[-1])


        for idx, (i, o, t) in enumerate(zip(
                                            torch.split(input_crop, 1, dim=0),
                                            torch.split(output, 1, dim=0),
                                            torch.split(target_crop, 1, dim=0))):

            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            # Calculate LUFS - handle cases where audio is too short
            try:
                target_audio = t.squeeze().cpu().numpy()
                output_audio = o.squeeze().cpu().numpy()
                # Check if audio length is sufficient for LUFS calculation
                if len(target_audio) > meter.block_size and len(output_audio) > meter.block_size:
                    target_lufs = meter.integrated_loudness(target_audio)
                    output_lufs = meter.integrated_loudness(output_audio)
                    l1_lufs = np.abs(output_lufs - target_lufs)
                else:
                    # Audio too short, use default value
                    target_lufs = 0.0
                    output_lufs = 0.0
                    l1_lufs = 0.0
            except (ValueError, Exception) as e:
                # If LUFS calculation fails, use default value
                target_lufs = 0.0
                output_lufs = 0.0
                l1_lufs = 0.0

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            # CustomWaveDataset has no params, use batch and sample index for file naming
            params_key = f"batch{bidx}-sample{idx}"

            if args.save_dir is not None:
                ofile = os.path.join(args.save_dir, f"{params_key}-output--{model_id}.wav")
                ifile = os.path.join(args.save_dir, f"{params_key}-input.wav")
                tfile = os.path.join(args.save_dir, f"{params_key}-target.wav")

                torchaudio.save(ofile, o.view(1,-1).cpu().float(), args.sample_rate)
                if not os.path.isfile(ifile):
                    torchaudio.save(ifile, i.view(1,-1).cpu().float(), args.sample_rate)
                if not os.path.isfile(tfile):
                    torchaudio.save(tfile, t.view(1,-1).cpu().float(), args.sample_rate)

            if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "LUFS" : [l1_lufs],
                    "Agg" : [aggregate_loss]
                }
            else:
                results[params_key]["L1"].append(l1_loss)
                results[params_key]["L1i"].append(l1i_loss)
                results[params_key]["STFT"].append(stft_loss)
                results[params_key]["STFTi"].append(stfti_loss)
                results[params_key]["LUFS"].append(l1_lufs)
                results[params_key]["Agg"].append(aggregate_loss)

    # store in dict
    l1_scores = []
    lufs_scores = []
    stft_scores = []
    agg_scores = []
    print("-" * 64)
    print("Config      L1         STFT      LUFS")
    print("-" * 64)
    for key, val in results.items():
        print(f"{key}    {np.mean(val['L1']):0.2e}    {np.mean(val['STFT']):0.3f}       {np.mean(val['LUFS']):0.3f}")

        l1_scores += val["L1"]
        stft_scores += val["STFT"]
        lufs_scores += val["LUFS"]
        agg_scores += val["Agg"]

    print("-" * 64)
    print(f"Mean     {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}      {np.mean(lufs_scores):0.3f}")
    print()
    overall_results[model_id] = results

pickle.dump(overall_results, open(f"test_results_{args.eval_subset}.p", "wb" ))

# we can make some kind of scatter plot to visualize this