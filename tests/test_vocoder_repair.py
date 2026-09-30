"""Behavioral checks for padding independence and adversarial decoder training."""

import torch
import torch.nn.functional as F

from aeroflow.models.pipeline import AeroFlowTTS
from aeroflow.models.discriminators import VocoderDiscriminators
from aeroflow.losses.vocoder import MelReconstructionLoss, discriminator_loss, generator_losses
from scripts.train_vocoder import paired_crops, split_indices, StepBatches


def test_recording_reconstruction_is_independent_of_batch_padding():
    torch.manual_seed(14)
    model = AeroFlowTTS().eval()
    # Exercise nonzero learned GRN parameters, not only its identity initialization.
    for block in model.decoder.blocks:
        block.grn.gamma.data.fill_(0.2)
        block.grn.beta.data.fill_(0.1)
    short = torch.randn(1, 4800)
    long = torch.randn(1, 9600)
    batch = torch.cat((F.pad(short, (0, 4800)), long))
    with torch.no_grad():
        alone = model.reconstruct(short)
        together = model.reconstruct(batch, torch.tensor([4800, 9600]))
        spectrum = model.stft_analysis(short)[1]
        padded = model.stft_analysis(batch, torch.tensor([4800, 9600]))[1]
        mask = torch.arange(padded.shape[-1])[None] < torch.tensor([[21], [41]])
        latent = model.acoustic_encoder(spectrum)
        batch_latent = model.acoustic_encoder(padded, mask)
    # Batched convolution and FP32 reductions can differ by a few ulps.
    torch.testing.assert_close(batch_latent[0:1, :, :21], latent, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(together[0:1, :4800], alone, rtol=2e-4, atol=2e-6)
    assert together[0, 4800:].count_nonzero() == 0


def test_vocoder_updates_only_decoder_and_generator_gradient_crosses_critics():
    torch.manual_seed(6)
    model = AeroFlowTTS().eval().requires_grad_(False)
    model.decoder.requires_grad_(True)
    critics = VocoderDiscriminators(channels=2, periods=(2, 5), resolutions=(512,))
    real = torch.randn(1, 2400) * .1
    generated = model.reconstruct(real)
    d_loss = discriminator_loss(critics(real), critics(generated.detach()))
    d_loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critics.parameters())
    critics.zero_grad(set_to_none=True)
    critics.requires_grad_(False)
    with torch.no_grad():
        real_maps = critics(real)
    adv, features = generator_losses(real_maps, critics(generated))
    loss = 45 * MelReconstructionLoss()(real, generated) + adv + 2 * features
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.acoustic_encoder.parameters())
    assert all(p.grad is None for p in model.vector_field.parameters())
    assert all(p.grad is None for p in critics.parameters())
    for component in model.decoder.out_proj.weight.grad.chunk(3):
        assert torch.isfinite(component).all() and component.abs().sum() > 0
    before = model.decoder.out_proj.weight.detach().clone()
    torch.optim.AdamW(model.decoder.parameters(), lr=1e-4).step()
    assert not torch.equal(before, model.decoder.out_proj.weight)


def test_mel_loss_does_not_penalize_polarity_and_zero_phase_is_finite():
    y = torch.randn(1, 2400)
    assert MelReconstructionLoss()(y, -y).item() == 0
    model = AeroFlowTTS()
    # Zero phase predictions must not produce NaN derivatives.
    with torch.no_grad():
        model.decoder.out_proj.weight[513:].zero_()
        model.decoder.out_proj.bias[513:].zero_()
    output = model.decoder(torch.randn(1, 32, 21))[0]
    (output.real.sum() + output.imag.sum()).backward()
    assert torch.isfinite(model.decoder.out_proj.weight.grad).all()


def test_training_crops_exclude_padding_and_resume_preserves_sampling():
    real = torch.stack((torch.arange(20), torch.arange(20) + 100)).float()
    real[0, 12:] = -999
    predicted = real + 7
    a, b = paired_crops(real, predicted, torch.tensor([12, 20]), 10, seed=9)
    assert (a != -999).all()
    torch.testing.assert_close(b - a, torch.full_like(a, 7))
    train, validation = split_indices(20, 3, 42)
    assert not set(train) & set(validation)
    assert set(train + validation) == set(range(20))
    batches = list(StepBatches(train, 4, 0, 10, 5))
    resumed = list(StepBatches(train, 4, 6, 10, 5))
    assert resumed == batches[6:]
    assert all(set(batch) <= set(train) for batch in batches)
