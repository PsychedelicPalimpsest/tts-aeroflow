"""Training-only waveform critics for acoustic decoder repair.

Multi-period and multi-resolution critics follow the approaches described in
HiFi-GAN (arXiv:2010.05646) and Vocos (arXiv:2306.00814). No discriminator is
needed at inference. Inputs must be equally sized, unpadded waveform crops.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=32):
        super().__init__()
        self.period = period
        widths = [1, channels, channels * 4, channels * 16, channels * 32, channels * 32]
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(a, b, (5, 1), stride=(3 if i < 4 else 1, 1), padding=(2, 0)))
            for i, (a, b) in enumerate(zip(widths, widths[1:]))
        ])
        self.score = weight_norm(nn.Conv2d(widths[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, audio):
        pad = (-audio.shape[-1]) % self.period
        if pad:
            audio = F.pad(audio, (0, pad), mode="reflect" if audio.shape[-1] > pad else "replicate")
        x = audio.reshape(audio.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.score(x), features


class ResolutionDiscriminator(nn.Module):
    """Judge local complex spectra at one resolution, without target phase L1."""
    def __init__(self, n_fft, channels=32):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft))
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, 3, padding=1)),
            *[weight_norm(nn.Conv2d(channels, channels, (5, 3), stride=(2, 1), padding=(2, 1)))
              for _ in range(3)],
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
        ])
        self.score = weight_norm(nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, audio):
        # FFT in float32 even when convolutional layers use mixed precision.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spectrum = torch.stft(
                audio.float(), n_fft=self.n_fft, hop_length=self.n_fft // 4,
                window=self.window.float(), return_complex=True,
                normalized=True, pad_mode="reflect" if audio.shape[-1] > self.n_fft // 2 else "constant",
            )
            x = torch.stack((spectrum.real, spectrum.imag), dim=1)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.score(x), features


class VocoderDiscriminators(nn.Module):
    def __init__(self, channels=32, periods=(2, 3, 5, 7, 11), resolutions=(512, 1024, 2048)):
        super().__init__()
        self.critics = nn.ModuleList([
            *[PeriodDiscriminator(p, channels) for p in periods],
            *[ResolutionDiscriminator(n, channels) for n in resolutions],
        ])

    def forward(self, audio):
        return [critic(audio) for critic in self.critics]
