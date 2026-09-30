"""Phase-tolerant reconstruction and adversarial vocoder objectives."""

import torch
from torch import nn
from torch.nn import functional as F


class MelReconstructionLoss(nn.Module):
    """L1 log-mel magnitude loss (HTK triangular bank, no torchaudio dependency)."""
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=240, n_mels=100):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        max_mel = 2595 * torch.log10(torch.tensor(1 + sample_rate / 2 / 700))
        points = 700 * (10 ** (torch.linspace(0, max_mel, n_mels + 2) / 2595) - 1)
        frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        rising = (frequencies[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
        falling = (points[2:, None] - frequencies[None]) / (points[2:] - points[1:-1])[:, None]
        self.register_buffer("bank", torch.minimum(rising, falling).clamp_min(0))

    def features(self, audio):
        s = torch.stft(audio.float(), self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        return (self.bank @ s.abs()).clamp_min(1e-5).log()

    def forward(self, real, generated):
        with torch.autocast(device_type=real.device.type, enabled=False):
            return F.l1_loss(self.features(generated), self.features(real))


def discriminator_loss(real_outputs, fake_outputs):
    """Hinge objective. The caller must detach fake audio for this update."""
    return sum(F.relu(1 - r.float()).mean() + F.relu(1 + f.float()).mean()
               for (r, _), (f, _) in zip(real_outputs, fake_outputs))


def generator_losses(real_outputs, fake_outputs):
    """Non-saturating hinge generator loss plus detached-real feature matching."""
    adversarial = sum(-score.float().mean() for score, _ in fake_outputs)
    matching = sum(
        sum(F.l1_loss(f.float(), r.detach().float()) for r, f in zip(real_maps, fake_maps)) / len(real_maps)
        for (_, real_maps), (_, fake_maps) in zip(real_outputs, fake_outputs)
    )
    return adversarial, matching
