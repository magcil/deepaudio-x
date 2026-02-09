import torch
import torch.nn as nn
import torchaudio


class AugmentMelSTFT(nn.Module):
    """
    A GPU-accelerated audio preprocessing module that converts raw waveforms to 
    augmented Mel-spectrograms.

    This module performs a complete audio front-end pipeline:
    1. Pre-emphasis filtering to balance the frequency spectrum.
    2. Short-Time Fourier Transform (STFT).
    3. Power spectrum calculation.
    4. Dynamic Mel-filterbank projection with frequency-range augmentation.
    5. Logarithmic scaling and normalization.
    6. SpecAugment (Frequency and Time masking) during training.
    """
    def __init__(
        self, 
        n_mels = 128, 
        sr = 32000, 
        win_length=800, 
        hopsize=320, 
        n_fft=1024, 
        freqm=48, 
        timem=192,
        fmin=0.0, 
        fmax=None, 
        fmin_aug_range=10, 
        fmax_aug_range=2000
    ):
        """
        Initializes the AugmentMelSTFT module.

        Args:
            n_mels (int): Number of Mel frequency bins.
            sr (int): Sampling rate of the input audio.
            win_length (int): Window size in samples for STFT.
            hopsize (int): Hop length (stride) in samples between successive STFT frames.
            n_fft (int): Length of the FFT size.
            freqm (int): Maximum size of frequency masks for SpecAugment. Set to 0 to disable.
            timem (int): Maximum size of time masks for SpecAugment. Set to 0 to disable.
            fmin (float): Minimum frequency for the Mel filterbank (Hz).
            fmax (float, optional): Maximum frequency for the Mel filterbank (Hz). 
                Defaults to Nyquist frequency minus half the max augmentation range.
            fmin_aug_range (int): Range in Hz for random jittering of the fmin parameter.
            fmax_aug_range (int): Range in Hz for random jittering of the fmax parameter.
        """
        torch.nn.Module.__init__(self)

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2

        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer(
            'window',
            torch.hann_window(win_length, periodic=False),
            persistent=False
        )

        if fmin_aug_range < 1:
            raise ValueError(f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation")
        if fmax_aug_range < 1:
            raise ValueError(f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation")
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x: torch.Tensor):
        """
        Processes raw audio waveforms into normalized Mel-spectrograms.

        Args:
            x (torch.Tensor): Input batch of audio waveforms. 
                Shape: (batch, samples).

        Returns:
            torch.Tensor: Augmented and normalized Mel-spectrogram. 
                Shape: (batch, n_mels, time_steps).
        """
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(
            x, 
            self.n_fft, 
            hop_length=self.hopsize, 
            win_length=self.win_length,
            center=True, 
            normalized=False, 
            window=self.window, 
            return_complex=False
        )
        x = (x ** 2).sum(dim=-1)

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,  
            self.n_fft, 
            self.sr,
            fmin, 
            fmax, 
            vtln_low=100.0, 
            vtln_high=-500., 
            vtln_warp_factor=1.0
        )

        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(
                mel_basis, (0, 1), 
                mode='constant', 
                value=0
            ),
            device=x.device
        )

        with torch.amp.autocast('cuda', enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5. 

        return melspec