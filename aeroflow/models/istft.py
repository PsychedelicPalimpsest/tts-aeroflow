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
        length: Optional[int] = None,
        lengths: Optional[torch.Tensor] = None,
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

        if lengths is not None:
            counts = lengths.detach().cpu().tolist()
            if len(counts) != S_complex.shape[0] or any(n < 1 for n in counts):
                raise ValueError("lengths must contain a positive length for each spectrum")
            total = length if length is not None else max(counts)
            if total < max(counts):
                raise ValueError("Output length is shorter than a recording")
            return torch.cat([
                torch.nn.functional.pad(
                    self(S_complex[i:i + 1, :, :1 + n // self.hop_length], length=n),
                    (0, total - n),
                ) for i, n in enumerate(counts)
            ])

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

    @torch.no_grad()
    def refine_phase(
        self, S_complex: torch.Tensor, iterations: int = 0,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """Optional Griffin-Lim refinement initialized with the learned phase.

        Holds the decoder magnitude fixed while projecting phase onto realizable
        waveforms. This is an inference experiment, not a perceptual guarantee:
        magnitudes were trained through iSTFT and may themselves be inaccurate.
        Zero iterations exactly preserves the original synthesis path.
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        spectrum = S_complex.to(torch.complex64)
        magnitude = spectrum.abs()
        audio = self(spectrum, length=length)
        for _ in range(iterations):
            projected = torch.stft(
                audio, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=self.window, center=True,
                pad_mode="reflect" if audio.shape[-1] > self.n_fft // 2 else "constant",
                normalized=False, onesided=True, return_complex=True,
            )
            if projected.shape != spectrum.shape:
                raise ValueError("length must preserve the input STFT frame count")
            phase = projected / projected.abs().clamp_min(1e-8)
            spectrum = magnitude * phase
            audio = self(spectrum, length=length)
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
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
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

        if lengths is not None:
            # Centered STFT must reflect each recording's own endpoint, not
            # a padded batch endpoint. Zero-pad spectra only after analysis.
            if lengths.shape != (audio.shape[0],):
                raise ValueError("lengths must contain one sample count per recording")
            counts = lengths.detach().cpu().tolist()
            if any(n <= self.n_fft // 2 or n > audio.shape[-1] for n in counts):
                raise ValueError("Audio lengths must exceed n_fft/2 and fit the batch")
            frames = 1 + audio.shape[-1] // self.hop_length
            spectra = [self(audio[i:i + 1, :n])[0] for i, n in enumerate(counts)]
            S_complex = torch.cat([
                torch.nn.functional.pad(s, (0, frames - s.shape[-1])) for s in spectra
            ])
            return S_complex, S_complex.abs(), torch.angle(S_complex)

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
