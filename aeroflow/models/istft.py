"""
AeroFlow-v2 Alias-Free COLA iSTFT Waveform Synthesizer & STFT Analysis.
Parameters:
- N_fft: 1024
- Hop length: 240 (100 Hz frame rate at 24 kHz sample rate)
- Window: Periodic Hann window (1024 samples)
- Overlap ratio: 76.56%
- Constant Overlap-Add (COLA) compliant.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class iSTFTSynthesizer(nn.Module):
    """
    COLA-compliant Alias-Free Complex iSTFT Synthesizer.
    Reconstructs 24 kHz audio directly from complex Fourier tensors S in C^(513 x T).
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 240):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Register periodic Hann window buffer
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        S_complex: torch.Tensor,
        length: Optional[int] = None
    ) -> torch.Tensor:
        """
        S_complex: [B, 513, T] complex STFT tensor
        length: Optional target waveform length in samples
        Returns:
            audio: [B, N_samples] synthesized 24 kHz audio
        """
        S_complex = S_complex.to(torch.complex64)
        assert S_complex.is_complex(), "Input tensor must be complex dtype"
        assert S_complex.shape[1] == self.n_fft // 2 + 1, (
            f"Expected {self.n_fft // 2 + 1} frequency bins, got {S_complex.shape[1]}"
        )

        audio = torch.istft(
            S_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=length
        )
        return audio


class STFTAnalysis(nn.Module):
    """
    Matching STFT Analysis module for extracting ground-truth complex Fourier tensors.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 240):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        audio: [B, N_samples] 24 kHz waveform
        Returns:
            S_complex: [B, 513, T] complex STFT tensor
            magnitude: [B, 513, T] linear magnitude
            phase: [B, 513, T] phase angles in [-pi, pi]
        """
        audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        S_complex = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True
        )
        magnitude = torch.abs(S_complex)
        phase = torch.angle(S_complex)
        return S_complex, magnitude, phase
