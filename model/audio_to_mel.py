import torch
import torch.nn as nn
import torchlibrosa as tl
from einops import rearrange
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cuda'
    ):
        super().__init__()
        # Initialize STFT module
        self.stft = tl.STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=False).to(device)
        
        # Initialize Mel Filter Bank
        mel_basis = librosa_mel_fn(sr=sampling_rate,n_fft=n_fft,n_mels=n_mel_channels,fmin=mel_fmin,fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).to(device).float()
        self.register_buffer("mel_basis", mel_basis)
        
        # Additional properties
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin):
        # Compute STFT
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1,2)
        real, img = self.stft(audio)
        # Apply Mel filterbank
        mel_output = torch.matmul(self.mel_basis, rearrange(real ** 2 + img ** 2, 'a 1 b c -> a c b'))
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec