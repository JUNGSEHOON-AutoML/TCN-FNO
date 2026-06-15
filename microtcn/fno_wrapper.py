import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from neuralop.models import FNO

from microtcn.base import Base
from microtcn.utils import center_crop

class FNOModel(Base):
    """
    Fourier Neural Operator (FNO) Wrapper for PyTorch Lightning.
    Inherits training and validation loops from microtcn.base.Base.
    """
    def __init__(
        self,
        n_modes=16,
        hidden_channels=64,
        n_layers=4,
        in_channels=1,
        out_channels=1,
        causal=False,
        lr=3e-4,
        **kwargs
    ):
        # Pass parameters to Base module class
        super(FNOModel, self).__init__(lr=lr, causal=causal, **kwargs)
        self.save_hyperparameters()

        # Initialize neuralop FNO 1D model
        # For 1D signal, n_modes is a tuple of length 1 (e.g. (16,))
        self.fno = FNO(
            n_modes=(self.hparams.n_modes,),
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            hidden_channels=self.hparams.hidden_channels,
            n_layers=self.hparams.n_layers
        )

    def forward(self, x, p=None):
        """
        Forward pass with internal instance normalization and residual connection.
        
        Args:
            x (Tensor): Input tensor of shape (batch, channels, length)
            p (Tensor, optional): Conditioning parameters (unused by FNO)
        """
        # x shape: (B, C, T)
        # 1. Normalization (Instance Norm) for FNO numerical stability
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        # 2. FNO Forward Pass
        fno_out = self.fno(x_norm)

        # 3. Un-normalize output and apply residual skip connection
        # out = x + fno_out * x_std
        out = x + fno_out * x_std

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Add parent Base args first
        parser = Base.add_model_specific_args(parent_parser)
        
        # Add FNO specific args
        parser.add_argument('--n_modes', type=int, default=16, 
                            help='Number of modes for FNO (1D)')
        parser.add_argument('--hidden_channels', type=int, default=64, 
                            help='Hidden channel width of FNO')
        parser.add_argument('--n_layers', type=int, default=4, 
                            help='Number of layers for FNO')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels')
        parser.add_argument('--out_channels', type=int, default=1, 
                            help='Number of output channels')
        parser.add_argument('--causal', action='store_true', default=False,
                            help='Causal option (always False for standard FNO)')

        return parser
