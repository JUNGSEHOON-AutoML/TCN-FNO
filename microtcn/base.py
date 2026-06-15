import os
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

import auraloss
from microtcn.utils import center_crop, causal_crop


class AdvancedSpectralLoss(torch.nn.Module):
    """고도화된 주파수/위상 손실 함수.

    수식:
        Loss = L_time + λ_harm * L_harm + λ_phase * L_phase

    - L_time  : 시간 도메인 L1 손실 (기본 파형 재구성)
    - L_harm  : 조화 스펙트럼 손실 (Complex STFT 기반 Magnitude L1)
    - L_phase : 군지연(Group-delay) 정규화 손실
                위상의 주파수 방향 1차 차분이 정답과 매끄럽게 일치하도록 유도

    Args:
        n_fft        (int)  : STFT FFT 크기.            Default: 2048
        hop_length   (int)  : STFT 홉 길이.             Default: 512
        lambda_harm  (float): 조화 스펙트럼 손실 가중치. Default: 1e-4
        lambda_phase (float): 군지연 위상 손실 가중치.   Default: 1e-5
    """

    def __init__(self, n_fft=2048, hop_length=512, lambda_harm=1e-4, lambda_phase=1e-5):
        super().__init__()
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.lambda_harm  = lambda_harm
        self.lambda_phase = lambda_phase
        # 윈도우 함수 사전 할당 (GPU 메모리 최적화)
        self.register_buffer('window', torch.hann_window(n_fft))

    def _wrap_phase(self, phase):
        """위상을 [-pi, pi] 범위로 래핑"""
        return (phase + torch.pi) % (2 * torch.pi) - torch.pi

    def _stft(self, x):
        """내부 STFT 연산 메서드"""
        B, C, T = x.shape
        x = x.view(B * C, T)  # 단일 채널로 병합
        stft_out = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, pad_mode='reflect'
        )
        return stft_out

    def forward_detailed(self, pred, target):
        """TensorBoard 로깅을 위한 개별 로스 반환 메서드

        Returns:
            tuple: (loss, l_time, l_harm, l_phase)
        """
        # 1. L_time: 시간 도메인 L1 Loss
        l_time = F.l1_loss(pred, target)

        # STFT 추출
        stft_pred   = self._stft(pred)
        stft_target = self._stft(target)

        # 2. L_harm: Magnitude L1
        mag_pred   = torch.abs(stft_pred)
        mag_target = torch.abs(stft_target)
        l_harm = F.l1_loss(mag_pred, mag_target)

        # 3. L_phase: Group-delay (위상의 주파수 방향 1차 차분) MAE
        phase_pred   = torch.angle(stft_pred)
        phase_target = torch.angle(stft_target)

        gd_pred   = torch.diff(phase_pred,   dim=1)  # 주파수 축(dim=1) 차분
        gd_target = torch.diff(phase_target, dim=1)

        # 차분 후 [-pi, pi] 래핑하여 오차 계산
        gd_diff_wrapped = self._wrap_phase(gd_pred - gd_target)
        l_phase = torch.mean(torch.abs(gd_diff_wrapped))

        # Total Loss 합산
        loss = l_time + (self.lambda_harm * l_harm) + (self.lambda_phase * l_phase)

        return loss, l_time, l_harm, l_phase

    def forward(self, pred, target):
        loss, _, _, _ = self.forward_detailed(pred, target)
        return loss


# 하위 호환성을 위해 FFTLoss를 별칭으로 유지
class FFTLoss(torch.nn.Module):
    """하위 호환용 FFT Magnitude 손실 (AdvancedSpectralLoss 사용을 권장)."""
    def __init__(self, use_phase=False, phase_weight=0.1):
        super(FFTLoss, self).__init__()
        self.use_phase    = use_phase
        self.phase_weight = phase_weight

    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft(pred,   dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        pred_mag   = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss   = F.l1_loss(pred_mag, target_mag)
        if self.use_phase:
            pred_phase   = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            phase_diff   = torch.abs(pred_phase - target_phase)
            phase_diff   = torch.min(phase_diff, 2 * np.pi - phase_diff)
            return mag_loss + self.phase_weight * torch.mean(phase_diff)
        return mag_loss

class Base(pl.LightningModule):
    """Base module with train and validation loops.

    Args:
        lr              (float): 학습률.                                              Default: 3e-4
        train_loss      (str)  : 학습 손실 종류. 지원 값:
                                 'l1', 'stft', 'l1+stft', 'fft', 'l1+fft',
                                 'fft+stft', 'l1+fft+stft', 'advanced'           Default: 'l1+stft'
        use_fft_loss    (bool) : FFT 손실 사용 여부 (하위 호환).                   Default: False
        fft_use_phase   (bool) : FFT 손실에 위상 항 포함 여부 (하위 호환).          Default: False
        fft_phase_weight(float): FFT 위상 항 가중치 (하위 호환).                   Default: 0.1
        lambda_harm     (float): AdvancedSpectralLoss 조화 스펙트럼 가중치 λ_harm.  Default: 1.0
        lambda_phase    (float): AdvancedSpectralLoss 군지연 위상 가중치 λ_phase.   Default: 0.1
        adv_n_fft       (int)  : AdvancedSpectralLoss STFT FFT 크기.               Default: 1024
        adv_hop_length  (int)  : AdvancedSpectralLoss STFT 홉 길이.                Default: 256
        save_dir        (str)  : 검증 오디오 저장 경로.                             Default: None
        num_examples    (int)  : 에폭당 로깅할 오디오 예시 수.                      Default: 4
    """
    def __init__(
        self,
        lr               = 3e-4,
        train_loss       = "l1+stft",
        use_fft_loss     = False,
        fft_use_phase    = False,
        fft_phase_weight = 0.1,
        lambda_harm      = 1e-4,
        lambda_phase     = 1e-5,
        adv_n_fft        = 2048,
        adv_hop_length   = 512,
        save_dir         = None,
        num_examples     = 4,
        **kwargs,
    ):
        super(Base, self).__init__()
        self.save_hyperparameters()

        # these lines need to be commented out when trying
        # to jit these models in `export.py`
        self.l1      = torch.nn.L1Loss()
        self.stft    = auraloss.freq.STFTLoss()
        self.fft     = FFTLoss(use_phase=fft_use_phase, phase_weight=fft_phase_weight)
        # 고도화된 손실 함수 (train_loss='advanced' 또는 직접 호출)
        self.advanced = AdvancedSpectralLoss(
            n_fft        = adv_n_fft,
            hop_length   = adv_hop_length,
            lambda_harm  = lambda_harm,
            lambda_phase = lambda_phase,
        )

    def forward(self, x, p):
        pass

    @torch.jit.unused   
    def training_step(self, batch, batch_idx):
        # Handle both (input, target) and (input, target, params) formats
        if len(batch) == 2:
            input, target = batch
            params = None
        else:
            input, target, params = batch

        # pass the input thrgouh the mode
        pred = self(input, params)

        # crop the input and target signals
        if self.hparams.causal:
            target = causal_crop(target, pred.shape[-1])
        else:
            target = center_crop(target, pred.shape[-1])

        # compute the error using appropriate loss
        if   self.hparams.train_loss == "l1":
            loss = self.l1(pred, target)
        elif self.hparams.train_loss == "stft":
            loss = self.stft(pred, target)
        elif self.hparams.train_loss == "fft":
            loss = self.fft(pred, target)
        elif self.hparams.train_loss == "l1+stft":
            l1_loss   = self.l1(pred, target)
            stft_loss = self.stft(pred, target)
            loss = l1_loss + stft_loss
        elif self.hparams.train_loss == "l1+fft":
            l1_loss  = self.l1(pred, target)
            fft_loss = self.fft(pred, target)
            loss = l1_loss + fft_loss
        elif self.hparams.train_loss == "fft+stft":
            fft_loss  = self.fft(pred, target)
            stft_loss = self.stft(pred, target)
            loss = fft_loss + stft_loss
        elif self.hparams.train_loss == "l1+fft+stft":
            l1_loss   = self.l1(pred, target)
            fft_loss  = self.fft(pred, target)
            stft_loss = self.stft(pred, target)
            loss = l1_loss + fft_loss + stft_loss
        elif self.hparams.train_loss == "advanced":
            # AdvancedSpectralLoss: L_time + λ_harm*L_harm + λ_phase*L_phase
            loss, l_time, l_harm, l_phase = self.advanced.forward_detailed(pred, target)
            self.log('train_loss/adv_time',  l_time,  on_step=True, logger=True)
            self.log('train_loss/adv_harm',  l_harm,  on_step=True, logger=True)
            self.log('train_loss/adv_phase', l_phase, on_step=True, logger=True)
        else:
            raise NotImplementedError(f"Invalid loss fn: {self.hparams.train_loss}")

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        # Handle both (input, target) and (input, target, params) formats
        if len(batch) == 2:
            input, target = batch
            params = None
        else:
            input, target, params = batch

        # pass the input thrgouh the mode
        pred = self(input, params)

        # crop the input and target signals
        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        # compute the validation error using all losses
        l1_loss   = self.l1(pred, target_crop)
        stft_loss = self.stft(pred, target_crop)
        fft_loss  = self.fft(pred, target_crop)

        # AdvancedSpectralLoss 세부 항목 계산
        adv_loss, adv_l_time, adv_l_harm, adv_l_phase = self.advanced.forward_detailed(pred, target_crop)

        # Aggregate loss depends on training loss configuration
        if self.hparams.train_loss == "advanced":
            aggregate_loss = adv_loss
        elif self.hparams.use_fft_loss or 'fft' in self.hparams.train_loss:
            aggregate_loss = l1_loss + stft_loss + fft_loss
        else:
            aggregate_loss = l1_loss + stft_loss

        self.log('val_loss',           aggregate_loss)
        self.log('val_loss/L1',        l1_loss)
        self.log('val_loss/STFT',      stft_loss)
        self.log('val_loss/FFT',       fft_loss)
        self.log('val_loss/adv_time',  adv_l_time)
        self.log('val_loss/adv_harm',  adv_l_harm)
        self.log('val_loss/adv_phase', adv_l_phase)

        # Only return audio samples for the first batch to avoid OOM memory leaks
        # (2415 batches * 12.6MB = 30GB accumulated in CPU memory)
        if batch_idx == 0:
            outputs = {
                "input" : input_crop.cpu().numpy(),
                "target": target_crop.cpu().numpy(),
                "pred"  : pred.cpu().numpy()
            }
            if params is not None:
                outputs["params"] = params.cpu().numpy()
            return outputs

        return None

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        # filter out None or empty values
        validation_step_outputs = [out for out in validation_step_outputs if out is not None]
        if not validation_step_outputs:
            return

        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : []
        }
        
        has_params = False
        if validation_step_outputs and "params" in validation_step_outputs[0]:
            outputs["params"] = []
            has_params = True

        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx,...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            
            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"target/{idx}", 
                                             t, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"pred/{idx}",   
                                             p, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                if has_params:
                    prm = outputs["params"][rand_idx].squeeze()
                    input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                    target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                    pred_filename = os.path.join(self.hparams.save_dir, 
                                    f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                else:
                    input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input.wav")
                    target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target.wav")
                    pred_filename = os.path.join(self.hparams.save_dir, 
                                    f"{idx}-pred-{self.hparams.train_loss}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                torchaudio.save(pred_filename, 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--train_loss', type=str, default="l1+stft",
                            help='Loss fn: l1, stft, l1+stft, fft, l1+fft, fft+stft, l1+fft+stft, advanced')
        parser.add_argument('--use_fft_loss', action='store_true', help='Use FFT loss (legacy)')
        parser.add_argument('--fft_use_phase', action='store_true', help='Include phase in FFT loss (legacy)')
        parser.add_argument('--fft_phase_weight', type=float, default=0.1,
                            help='Weight for phase term in FFT loss (legacy)')
        # --- AdvancedSpectralLoss 파라미터 ---
        parser.add_argument('--lambda_harm', type=float, default=1e-4,
                            help='λ_harm: weight for harmonic (STFT magnitude) loss in AdvancedSpectralLoss')
        parser.add_argument('--lambda_phase', type=float, default=1e-5,
                            help='λ_phase: weight for group-delay phase loss in AdvancedSpectralLoss')
        parser.add_argument('--adv_n_fft', type=int, default=2048,
                            help='FFT size for AdvancedSpectralLoss STFT')
        parser.add_argument('--adv_hop_length', type=int, default=512,
                            help='Hop length for AdvancedSpectralLoss STFT')
        # --- validation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser