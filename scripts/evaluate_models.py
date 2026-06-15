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
import glob
import torch
import torchaudio
import numpy as np
from argparse import ArgumentParser
from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.utils import center_crop, causal_crop
from microtcn.data import CustomWaveDataset

def compute_mse(pred, target):
    """Compute Mean Squared Error"""
    return torch.mean((pred - target) ** 2).item()

def compute_pesq(pred, target, sample_rate=192000):
    """Compute PESQ score if pesq library is available"""
    try:
        from pesq import pesq
        # Convert to numpy and ensure mono
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        
        # Ensure same length
        min_len = min(len(pred_np), len(target_np))
        pred_np = pred_np[:min_len]
        target_np = target_np[:min_len]
        
        # PESQ requires 16kHz or 8kHz, so resample if needed
        if sample_rate == 44100:
            # Simple downsampling (for accurate results, use proper resampling)
            pred_np = pred_np[::2]  # Approximate 22kHz
            target_np = target_np[::2]
            sample_rate = 22050
        
        if sample_rate == 22050:
            pred_np = pred_np[::2]  # Approximate 11kHz
            target_np = target_np[::2]
            sample_rate = 11025
        
        # PESQ works with 8kHz or 16kHz
        if sample_rate >= 16000:
            # Downsample to 16kHz
            factor = int(sample_rate / 16000)
            pred_np = pred_np[::factor]
            target_np = target_np[::factor]
            sample_rate = 16000
        else:
            # Downsample to 8kHz
            factor = int(sample_rate / 8000)
            pred_np = pred_np[::factor]
            target_np = target_np[::factor]
            sample_rate = 8000
        
        score = pesq(sample_rate, target_np, pred_np, 'wb')
        return score
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: PESQ computation failed: {e}")
        return None

def evaluate_model(checkpoint_path, root_dir, test_subset="test", batch_size=1, half=False):
    """Evaluate a trained model on test set"""
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Determine model type from checkpoint or filename
    model_type = None
    if "lstm" in checkpoint_path.lower() or "LSTM" in checkpoint_path:
        model_type = "lstm"
    else:
        model_type = "tcn"
    
    # Get hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    nparams = hparams.get("nparams", 0)
    
    if model_type == "lstm":
        model = LSTMModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    else:
        model = TCNModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    
    model.eval()
    model.freeze()
    
    if torch.cuda.is_available():
        model = model.cuda()
        if half:
            model = model.half()
    
    # Load test dataset
    print(f"Loading test dataset from {root_dir}...")
    test_dataset = CustomWaveDataset(
        root_dir=root_dir,
        subset=test_subset,
        length=131072,  # Use full file length for evaluation
        preload=False,
        half=half,
        fraction=1.0
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluation metrics
    total_mse = 0.0
    total_pesq = 0.0
    pesq_count = 0
    count = 0
    
    print(f"Evaluating {len(test_dataset)} examples...")
    
    with torch.no_grad():
        for batch_idx, (input_audio, target_audio) in enumerate(test_dataloader):
            if torch.cuda.is_available():
                input_audio = input_audio.cuda()
                target_audio = target_audio.cuda()
                if half:
                    input_audio = input_audio.half()
                    target_audio = target_audio.half()
            
            # Forward pass (no params)
            pred = model(input_audio, None)
            
            # Crop target to match prediction size
            if hasattr(model.hparams, 'causal') and model.hparams.causal:
                target_crop = causal_crop(target_audio, pred.shape[-1])
            else:
                target_crop = center_crop(target_audio, pred.shape[-1])
            
            # Calculate MSE
            mse = compute_mse(pred, target_crop)
            total_mse += mse
            count += 1
            
            # Calculate PESQ (if available)
            if batch_size == 1:  # PESQ works on single examples
                pesq_score = compute_pesq(pred, target_crop, sample_rate=192000)
                if pesq_score is not None:
                    total_pesq += pesq_score
                    pesq_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches...")
    
    # Calculate averages
    avg_mse = total_mse / count if count > 0 else 0.0
    avg_pesq = total_pesq / pesq_count if pesq_count > 0 else None
    
    return {
        "mse": avg_mse,
        "pesq": avg_pesq,
        "num_examples": count
    }

def main():
    parser = ArgumentParser(description="Evaluate trained models on test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of dataset (should contain x_t/ and y_t/ folders)")
    parser.add_argument("--test_subset", type=str, default="test",
                        help="Subset name (not used for x_t/y_t format, but kept for compatibility)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision (FP16)")
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        test_subset=args.test_subset,
        batch_size=args.batch_size,
        half=args.half
    )
    
    # Print results
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {args.checkpoint}")
    print(f"Number of examples: {results['num_examples']}")
    print(f"Mean Squared Error (MSE): {results['mse']:.6e}")
    if results['pesq'] is not None:
        print(f"PESQ Score: {results['pesq']:.4f}")
    else:
        print("PESQ Score: N/A (pesq library not available or computation failed)")
    print("=" * 60)

if __name__ == "__main__":
    main()

