"""
AeroFlow-v2: Ultra-Fast Non-Autoregressive Text-to-Speech Engine
Champion architecture featuring:
- Conformer Phoneme Encoder with RoPE.
- Monotonic Alignment Search (Viterbi MAS) + Energy-Constrained Monotonic Duration Predictor.
- 32-Channel Continuous Latent Flow Matching with ConvNeXt-ODE Vector Field.
- 6-Step Non-Uniform Heun Solver with polynomial schedule (rho = 1.5).
- ConvNeXt-V2 Complex STFT Decoder (Log-Mag + Unit Phase Vectors).
- Alias-Free Constant Overlap-Add (COLA) iSTFT Synthesis.
- 100% Non-Adversarial Convex Spectral Training Suite.
"""

from aeroflow.frontend.text_norm import TextNormalizer
from aeroflow.frontend.phonemizer import Phonemizer, PHONEME_TO_ID, ID_TO_PHONEME
from aeroflow.models.encoder import ConformerEncoder
from aeroflow.models.alignment import (
    EnergyConstrainedDurationPredictor,
    maximum_path_viterbi,
    alignment_to_durations,
    expand_text_representations
)
from aeroflow.models.flow_matching import (
    VectorFieldNetwork,
    NonUniformHeunSolver,
    OptimalTransportCFM
)
from aeroflow.models.decoder import ComplexSTFTDecoder
from aeroflow.models.istft import iSTFTSynthesizer, STFTAnalysis
from aeroflow.models.pipeline import AeroFlowTTS
from aeroflow.losses.losses import (
    AeroFlowLoss,
    CFMLoss,
    DurationLoss,
    MultiResolutionSTFTLoss,
    InstantaneousFrequencyLoss
)
from aeroflow.dataset.dataset import (
    HiFiTTSDataset,
    collate_hifi_tts,
    create_synthetic_batch
)

__version__ = "2.0.0"
__all__ = [
    "AeroFlowTTS",
    "TextNormalizer",
    "Phonemizer",
    "PHONEME_TO_ID",
    "ID_TO_PHONEME",
    "ConformerEncoder",
    "EnergyConstrainedDurationPredictor",
    "maximum_path_viterbi",
    "alignment_to_durations",
    "expand_text_representations",
    "VectorFieldNetwork",
    "NonUniformHeunSolver",
    "OptimalTransportCFM",
    "ComplexSTFTDecoder",
    "iSTFTSynthesizer",
    "STFTAnalysis",
    "AeroFlowLoss",
    "CFMLoss",
    "DurationLoss",
    "MultiResolutionSTFTLoss",
    "InstantaneousFrequencyLoss",
    "HiFiTTSDataset",
    "collate_hifi_tts",
    "create_synthetic_batch",
]
