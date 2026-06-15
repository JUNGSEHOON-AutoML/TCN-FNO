import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from microtcn.base import Base

class LSTMModel(Base):
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 num_layers=1,
                 **kwargs):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()

        # input_size depends on whether we have conditioning parameters
        input_size = ninputs + (nparams if nparams > 0 else 0)
        self.lstm = torch.nn.LSTM(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   batch_first=False,
                                   bidirectional=False)
        
        self.linear = torch.nn.Linear(self.hparams.hidden_size, 
                                      self.hparams.noutputs)

    def forward(self, x, p):

        bs = x.size(0) # batch size
        s = x.size(-1) # samples
        x = x.permute(2,0,1) # shape for LSTM (seq, batch, channel)

        if p is not None and self.hparams.nparams > 0:
            p = p.permute(1,0,2) # change channel to seq dim
            p = p.repeat(s,1,1) # expand to every time step
            x = torch.cat((x, p), dim=-1) # append to input along feature dim

        out, _ = self.lstm(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,2,0) # put shape back (batch, channel, seq)

        return out

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        # First add base model args (includes train_loss, fft_loss options, etc.)
        parser = Base.add_model_specific_args(parent_parser)
        
        # Then add LSTM-specific args
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=1)
        parser.add_argument('--hidden_size', type=int, default=32)
        parser.add_argument('--num_layers', type=int, default=1)

        return parser