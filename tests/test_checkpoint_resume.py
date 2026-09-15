"""
Tests for checkpoint RNG restore robustness (scripts/train_kaggle.py).

Regression: resuming from a checkpoint whose stored RNG states are not
ByteTensors (older writers, foreign checkpoint_latest.pt files) crashed
startup with `TypeError: RNG state must be a torch.ByteTensor`.
Restore must coerce-or-skip every entry and never fail startup.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _import_train_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train_kaggle.py"
    spec = importlib.util.spec_from_file_location("train_kaggle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_setup():
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_rng_round_trip_reproduces_torch_stream(tmp_path):
    tk = _import_train_module()
    model, opt = _tiny_setup()
    ckpt = tmp_path / "checkpoint_latest.pt"
    tk.save_atomic_checkpoint(ckpt, model, opt, None, None, 123, 4, 1.5, False)
    expected = torch.rand(16)  # first draws from the saved RNG state

    torch.manual_seed(999)  # perturb away from saved state
    torch.rand(100)
    model2, opt2 = _tiny_setup()
    step, epoch, best = tk.resume_from_checkpoint(
        ckpt, model2, opt2, None, None, torch.device("cpu"), False)
    assert (step, epoch, best) == (123, 4, 1.5)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    # Exact torch RNG replay: first draw after resume matches the first
    # draw from the saved state.
    assert torch.equal(torch.rand(16), expected)


def test_corrupt_rng_entries_never_fail_startup(tmp_path):
    tk = _import_train_module()
    model, opt = _tiny_setup()
    ckpt = tmp_path / "checkpoint_latest.pt"
    tk.save_atomic_checkpoint(ckpt, model, opt, None, None, 7, 1, 2.0, False)

    for bad_rng in (
        {"cpu": [1, 2, 3], "numpy": None, "python": None},          # list, not tensor
        {"cpu": "not-a-state", "numpy": "xx", "python": "yy"},      # strings
        {"cpu": torch.zeros(10, dtype=torch.float32)},              # wrong dtype
        {"cpu": torch.zeros(10, dtype=torch.uint8)},                # wrong size
        ["not", "a", "dict"],                                       # non-dict
        {},                                                         # missing entirely
    ):
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state["rng_state"] = bad_rng
        bad_path = tmp_path / "bad.pt"
        torch.save(state, bad_path)
        model2, opt2 = _tiny_setup()
        step, epoch, best = tk.resume_from_checkpoint(
            bad_path, model2, opt2, None, None, torch.device("cpu"), False)
        assert (step, epoch, best) == (7, 1, 2.0)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


def test_foreign_checkpoint_refused_explicitly(tmp_path):
    tk = _import_train_module()
    model, opt = _tiny_setup()
    foreign = tmp_path / "checkpoint_latest.pt"
    torch.save({"some": "other-format", "rng_state": {}}, foreign)
    with pytest.raises(ValueError, match="not an AeroFlow checkpoint"):
        tk.resume_from_checkpoint(foreign, model, opt, None, None,
                                  torch.device("cpu"), False)
