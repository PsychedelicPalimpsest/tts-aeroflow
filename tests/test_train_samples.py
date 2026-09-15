"""
Tests for periodic training sample generation (scripts/train_kaggle.py).
"""

import importlib.util
from pathlib import Path

import soundfile as sf
import torch

from aeroflow import AeroFlowTTS


def _import_train_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train_kaggle.py"
    spec = importlib.util.spec_from_file_location("train_kaggle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sample_prompts():
    module = _import_train_module()
    assert len(module.DEFAULT_SAMPLE_PROMPTS) >= 2
    assert all(p.strip() for p in module.DEFAULT_SAMPLE_PROMPTS)


def test_save_training_samples_writes_valid_wavs(tmp_path):
    module = _import_train_module()
    torch.manual_seed(0)
    model = AeroFlowTTS()
    model.train()
    paths = module.save_training_samples(model, ["Hi there."], tmp_path, 42)
    assert len(paths) == 1
    assert paths[0].name == "step_000042_0.wav"
    assert model.training  # mode restored
    wav, sr = sf.read(str(paths[0]))
    assert sr == 24000 and len(wav) > 0
    assert (tmp_path / "prompts.txt").exists()


def test_save_training_samples_never_raises(tmp_path):
    module = _import_train_module()
    model = AeroFlowTTS()
    model.train()

    def boom(prompt, alpha=1.0):
        raise RuntimeError("synth broken")

    model.synthesize = boom
    assert module.save_training_samples(model, ["Hi."], tmp_path, 7) == []
    assert model.training  # mode restored even on failure
    assert module.save_training_samples(model, [], tmp_path, 8) == []
