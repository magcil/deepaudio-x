from typing import cast

import torch
import torch.nn as nn
import torchaudio


class AugmentMelSTFT(nn.Module):
    """
    Computes a Mel-scaled spectrogram from audio waveforms with optional
    frequency and time masking for data augmentation.

    This module applies pre-emphasis filtering, short-time Fourier transform
    (STFT), conversion to power spectrogram, Mel filterbank, log compression,
    and optional frequency/time masking augmentation during training.

    Args:
        n_mels (int): Number of Mel bands. Defaults to 128.
        sr (int): Sampling rate of input audio. Defaults to 32000.
        win_length (int): Window size for STFT. Defaults to 800.
        hopsize (int): Hop size for STFT. Defaults to 320.
        n_fft (int): Number of FFT points. Defaults to 1024.
        freqm (int): Maximum width for frequency masking. 0 disables masking. Defaults to 48.
        timem (int): Maximum width for time masking. 0 disables masking. Defaults to 192.
        htk (bool): Whether to use HTK formula for Mel scale. Defaults to False.
        fmin (float): Minimum frequency for Mel scale. Defaults to 0.0 Hz.
        fmax (float): Maximum frequency for Mel scale. Defaults to sr/2 minus half of fmax_aug_range.
        norm (int): Normalization mode for Mel filterbanks. Defaults to 1.
        fmin_aug_range (int): Frequency augmentation range for fmin. Must be >=1. Defaults to 10.
        fmax_aug_range (int): Frequency augmentation range for fmax. Must be >=1. Defaults to 2000.
    """

    def __init__(
        self,
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=48,
        timem=192,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=10,
        fmax_aug_range=2000,
    ):
        torch.nn.Module.__init__(self)
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        if fmin_aug_range < 1:
            raise ValueError(f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation")
        if fmax_aug_range < 1:
            raise ValueError(f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation")

        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        """
        Compute augmented Mel spectrogram.

        Args:
            x (torch.Tensor): Input waveform tensor of shape (batch, time).

        Returns:
            torch.Tensor: Mel spectrogram tensor of shape (batch, n_mels, time_frames),
                          log-compressed and optionally augmented during training.
        """
        x = nn.functional.conv1d(
            x.unsqueeze(1),
            cast(torch.Tensor, self.preemphasis_coefficient),
        ).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=cast(torch.Tensor, self.window),
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr, fmin, fmax, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0), device=x.device
        )
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec

    def extra_repr(self):
        """
        Extra representation for printing.

        Returns:
            str: Information about window and hop size.
        """
        return f"winsize={self.win_length}, hopsize={self.hopsize}"
