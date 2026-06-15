import os
import sys
import glob
from pathlib import Path
from itertools import product
from argparse import ArgumentParser

# NumPy compatibility patch for older PyTorch Lightning / Librosa versions
import numpy as np
np.object = object
np.bool = bool
np.int = int
np.float = float
np.long = int
np.complex = complex

# Resolve project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import torch
import torchsummary
import pytorch_lightning as pl

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.fno_wrapper import FNOModel
from microtcn.data import SignalTrainLA2ADataset, CustomWaveDataset

torch.backends.cudnn.benchmark = True

train_configs = [
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.01,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "TCN-300",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 16  # Reduced batch size for memory efficiency
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.10,
     "batch_size" : 32
    },
    {"name" : "LSTM-32",
     "model_type" : "lstm",
     "num_layers" : 1,
     "hidden_size" : 32,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 3,
     "dilation_growth" : 60,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1"
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 30,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
    },
    {"name" : "uTCN-324-16",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "channel_width" : 16,
    },
    # FFT Loss experiments
    {"name" : "uTCN-300-FFT",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "fft"
    },
    {"name" : "uTCN-300-L1FFT",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1+fft"
    },
    {"name" : "uTCN-300-FFTSTFT",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "fft+stft"
    },
    {"name" : "uTCN-300-ALL",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1+fft+stft"
    },
]

n_configs = len(train_configs)

# Pre-parse --config_idx to skip configurations before full parsing
config_idx = None
for arg_idx, arg in enumerate(sys.argv):
    if arg.startswith('--config_idx='):
        try:
            config_idx = int(arg.split('=')[1])
        except ValueError:
            pass
    elif arg == '--config_idx' and arg_idx + 1 < len(sys.argv):
        try:
            config_idx = int(sys.argv[arg_idx + 1])
        except ValueError:
            pass

for idx, tconf in enumerate(train_configs):

    if config_idx is not None and (idx + 1) != config_idx:
        continue

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--config_idx', type=int, default=None, help='Index of config to train')

    # add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='tcn', help='tcn, lstm, or fno')
    parser.add_argument('--root_dir', type=str, default='data' if os.path.isdir('data/x_t') else '.')
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--sample_rate', type=int, default=192000)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_subset', type=str, default='train', 
                        help='Training subset: "train" (80%% of data) or "full" (all data)')
    parser.add_argument('--val_subset', type=str, default='train',
                        help='Validation subset: "train" (same as training, recommended) or "val" (same as train)')
    parser.add_argument('--train_length', type=int, default=65536)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=131072)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--target_gpu', type=int, default=1, help='Physical GPU index to use directly')
    
    # Custom loss and augmentation overrides
    parser.add_argument('--loss', type=str, default=None, help='Override train loss (l1, stft, advanced, etc.)')
    parser.add_argument('--augment', type=str, default='True', help='Apply data augmentation (True/False)')

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    print(f"* Training config {idx+1}/{n_configs}")
    print(tconf)
  
    # let the model add what it wants
    if temp_args.model_type == 'tcn':
        parser = TCNModel.add_model_specific_args(parser)
    elif temp_args.model_type == 'lstm':
        parser = LSTMModel.add_model_specific_args(parser)
    elif temp_args.model_type == 'fno':
        parser = FNOModel.add_model_specific_args(parser)

    # parse them args
    args = parser.parse_args()

    # Handle alias and overrides
    if args.loss is not None:
        args.train_loss = args.loss
        tconf["train_loss"] = args.loss
        
    if args.model_type is not None:
        tconf["model_type"] = args.model_type
        tconf["name"] = f"u{args.model_type.upper()}"
        
    # Override configuration batch size if specified in CLI
    has_batch_size_cli = any(arg.startswith('--batch_size') for arg in sys.argv)
    if has_batch_size_cli:
        tconf["batch_size"] = args.batch_size
        
    augment_bool = args.augment.lower() in ['true', '1', 'yes', 't']

    # set the seed
    pl.seed_everything(42)

    # Check if max_epochs was specified in CLI
    has_max_epochs_cli = any(arg.startswith('--max_epochs') for arg in sys.argv)

    # init the trainer and model 
    if tconf["model_type"] == 'tcn':
        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += "__causal" if tconf['causal'] else "__noncausal"
        specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
    elif tconf["model_type"] == 'lstm':
        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += f"__{tconf['num_layers']}-{tconf['hidden_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
    elif tconf["model_type"] == 'fno':
        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += f"__{args.n_modes}-{args.hidden_channels}-{args.n_layers}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"

    if not has_max_epochs_cli:
        if "max_epochs" in tconf:
            args.max_epochs = tconf["max_epochs"]
        else:
            args.max_epochs = 60

    if "train_loss" in tconf:
        args.train_loss = tconf["train_loss"]
        specifier += f"__loss-{tconf['train_loss']}"

    # Use FP32 for numerical stability (FP16 causes NaN losses)
    args.precision = 32
    
    # Use GPU if available
    import torch
    if torch.cuda.is_available():
        # Respect the explicit target physical GPU index to bypass NVML mapping bugs
        args.gpus = [args.target_gpu]
        print(f"Forcing execution on physical GPU: {args.gpus}")
        # Enable gradient clipping to prevent NaN losses
        args.gradient_clip_val = 1.0
    else:
        args.gpus = None

    args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
    print(args.default_root_dir)
    trainer = pl.Trainer.from_argparse_args(args)

    # setup the dataloaders
    # Use CustomWaveDataset for x_t/y_t format (no conditioning parameters)
    # Train/Test split: 80% train, 20% test (deterministic split by segment ID)
    train_dataset = CustomWaveDataset(
        args.root_dir,
        subset=args.train_subset,  # "train" uses 80% of data (deterministic split)
        fraction=tconf["train_fraction"],
        half=True if args.precision == 16 else False,
        preload=args.preload,
        length=args.train_length,
        target_sample_rate=args.sample_rate,
        augment=augment_bool
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=args.shuffle,
                                                batch_size=tconf["batch_size"],
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    val_dataset = CustomWaveDataset(
        args.root_dir,
        subset=args.val_subset,  # Use "train" for validation (same as training data) or "val" for separate validation set
        preload=args.preload,
        target_sample_rate=args.sample_rate,
        half=True if args.precision == 16 else False,
        length=args.eval_length,
        fraction=1.0,
        augment=False
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                shuffle=False,
                                                batch_size=2, # Reduced to 2 to prevent FNO validation CUDA OOM
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    # create the model with args
    dict_args = vars(args)
    dict_args["nparams"] = 0  # No conditioning parameters for CustomWaveDataset

    if tconf["model_type"] == 'tcn':
        dict_args["nblocks"] = tconf["nblocks"]
        dict_args["dilation_growth"] = tconf["dilation_growth"]
        dict_args["kernel_size"] = tconf["kernel_size"]
        dict_args["causal"] = tconf["causal"]
        if "channel_width" in tconf:
            dict_args["channel_width"] = tconf["channel_width"]
        model = TCNModel(**dict_args)
    elif tconf["model_type"] == 'lstm':
        dict_args["num_layers"] = tconf["num_layers"]
        dict_args["hidden_size"] = tconf["hidden_size"]
        model = LSTMModel(**dict_args)
    elif tconf["model_type"] == 'fno':
        model = FNOModel(**dict_args)

    # summary (skip for LSTM models as torchsummary doesn't handle LSTM tuple outputs)
    # Also skip if torchsummary causes hook issues - we'll just print model info instead
    if tconf["model_type"] != 'lstm':
        try:
            # Count parameters manually to avoid torchsummary hook issues
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model Summary:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model type: {tconf['model_type']}, Model name: {tconf['name']}")
            
            # Try torchsummary but clear hooks immediately after
            if dict_args["nparams"] == 0:
                torchsummary.summary(model, [(1, 65536)], device="cpu")
            else:
                torchsummary.summary(model, [(1, 65536), (1, 2)], device="cpu")
            
            # CRITICAL: Remove all hooks registered by torchsummary to prevent interference
            # torchsummary registers forward hooks that cause IndexError during validation
            for module in model.modules():
                if hasattr(module, '_forward_hooks'):
                    module._forward_hooks.clear()
                if hasattr(module, '_forward_pre_hooks'):
                    module._forward_pre_hooks.clear()
                if hasattr(module, '_backward_hooks'):
                    module._backward_hooks.clear()
        except Exception as e:
            print(f"Warning: Could not generate detailed model summary: {e}")
            print(f"Model type: {tconf['model_type']}, Model name: {tconf['name']}")
            # Make sure to clear hooks even if summary failed
            try:
                for module in model.modules():
                    if hasattr(module, '_forward_hooks'):
                        module._forward_hooks.clear()
                    if hasattr(module, '_forward_pre_hooks'):
                        module._forward_pre_hooks.clear()
                    if hasattr(module, '_backward_hooks'):
                        module._backward_hooks.clear()
            except:
                pass

    # train!
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Clear GPU memory after each config
    import torch
    del model, trainer, train_dataloader, val_dataloader, train_dataset, val_dataset
    torch.cuda.empty_cache()
