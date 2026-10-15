import math

import pytest
import torch

from models.itarflow import (
    ImprovedTARFlow,
    apply_rotary_pos_emb,
    create_itarflow,
    precompute_rope_freqs,
)


def _randomize_flow(model: ImprovedTARFlow, seed: int = 0):
    """Make the flow non-trivial.

    At init the final layer (proj_out) is zero and the per-layer attn/mlp scales
    are zero, so every block is an identity map (alpha=mu=0). That makes shape and
    invertibility checks pass trivially, so we perturb those parameters to give the
    flow a real, autoregressive transformation to exercise.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for meta_block in model.blocks:
            torch.nn.init.normal_(meta_block.final_layer.proj_out.weight, std=0.1)
            torch.nn.init.normal_(meta_block.final_layer.proj_out.bias, std=0.1)
            for layer in meta_block.blocks:
                torch.nn.init.normal_(layer.attn_scale, std=0.1)
                torch.nn.init.normal_(layer.mlp_scale, std=0.1)


def test_rope_matches_complex_rotation():
    """The real-valued implementation preserves adjacent-pair complex rotation."""
    x = torch.randn(2, 3, 5, 8)
    rope_freqs = precompute_rope_freqs(8, 5)

    actual = apply_rotary_pos_emb(x, rope_freqs)

    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    freqs_complex = torch.complex(rope_freqs[..., 0], rope_freqs[..., 1])
    expected = torch.view_as_real(
        x_complex * freqs_complex.unsqueeze(0).unsqueeze(0)
    ).reshape_as(x)

    torch.testing.assert_close(actual, expected)


def test_forward_pass_shape(itarflow_model, tiny_itarflow_config):
    """forward maps data -> latent z plus a per-sample log-det."""
    cfg = tiny_itarflow_config
    batch_size = 2

    x = torch.randn(batch_size, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(batch_size, cfg.static_in)

    z, total_logdet = itarflow_model(x, static)

    assert z.shape == (batch_size, cfg.num_timesteps, cfg.out_channels)
    assert total_logdet.shape == (batch_size,)


def test_reverse_pass_shape(itarflow_model, tiny_itarflow_config):
    """reverse maps latent noise -> sampled series plus a per-sample log-det."""
    cfg = tiny_itarflow_config
    batch_size = 2

    z = torch.randn(batch_size, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(batch_size, cfg.static_in)

    samples, total_reverse_logdet = itarflow_model.reverse(z, static)

    assert samples.shape == (batch_size, cfg.num_timesteps, cfg.out_channels)
    assert total_reverse_logdet.shape == (batch_size,)


def test_invertibility(itarflow_model, tiny_itarflow_config):
    """forward and reverse should be inverses in both directions.

    This also exercises the autoregressive KV-cache decode path in reverse, which
    must agree with the parallel forward pass for the reconstructions to hold.
    """
    cfg = tiny_itarflow_config
    _randomize_flow(itarflow_model)
    itarflow_model.eval()

    torch.manual_seed(0)
    static = torch.randn(2, cfg.static_in)

    # data -> latent -> data
    x = torch.randn(2, cfg.num_timesteps, cfg.out_channels)
    with torch.no_grad():
        z, _ = itarflow_model(x, static)
        x_recon, _ = itarflow_model.reverse(z, static)

    torch.testing.assert_close(
        x_recon,
        x,
        rtol=1e-4,
        atol=1e-4,
        msg="reverse(forward(x)) did not reconstruct x",
    )

    # latent -> data -> latent
    y = torch.randn(2, cfg.num_timesteps, cfg.out_channels)
    with torch.no_grad():
        samples, _ = itarflow_model.reverse(y, static)
        y_recon, _ = itarflow_model(samples, static)

    torch.testing.assert_close(
        y_recon,
        y,
        rtol=1e-4,
        atol=1e-4,
        msg="forward(reverse(y)) did not reconstruct y",
    )


def test_logdet_sign_consistency(itarflow_model, tiny_itarflow_config):
    """Forward and reverse log-dets are equal and opposite on a round trip."""
    cfg = tiny_itarflow_config
    _randomize_flow(itarflow_model)
    itarflow_model.eval()

    torch.manual_seed(1)
    x = torch.randn(2, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(2, cfg.static_in)

    with torch.no_grad():
        z, fwd_logdet = itarflow_model(x, static)
        _, rev_logdet = itarflow_model.reverse(z, static)

    torch.testing.assert_close(fwd_logdet, -rev_logdet, rtol=1e-4, atol=1e-4)


def test_compute_loss_and_gradients(itarflow_model, tiny_itarflow_config):
    """compute_loss returns a scalar and gradients flow to model parameters."""
    cfg = tiny_itarflow_config
    _randomize_flow(itarflow_model)

    batch_size = 2
    targets = torch.randn(batch_size, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(batch_size, cfg.static_in)

    itarflow_model.zero_grad()
    loss = itarflow_model.compute_loss(static, targets, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss not a scalar"

    loss.backward()

    proj_in = itarflow_model.blocks[0].proj_in
    assert proj_in.weight.grad is not None
    assert torch.any(proj_in.weight.grad != 0)


def test_masked_loss_matches_valid_subset(itarflow_model, tiny_itarflow_config):
    cfg = tiny_itarflow_config
    _randomize_flow(itarflow_model)

    targets = torch.randn(3, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(3, cfg.static_in)
    inputs = targets[:, :-1]
    mask = torch.tensor([1.0, 1.0, 0.0])

    masked_loss = itarflow_model.compute_loss(static, inputs, targets, mask=mask)
    valid_loss = itarflow_model.compute_loss(static[:2], inputs[:2], targets[:2])

    torch.testing.assert_close(masked_loss, valid_loss)


def test_training_pipeline_helpers(itarflow_model):
    assert itarflow_model.get_device() == next(itarflow_model.parameters()).device

    optimizer = itarflow_model.setup_optimiser(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]

    grouped_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    trainable_params = [
        param for param in itarflow_model.parameters() if param.requires_grad
    ]
    assert len(grouped_params) == len(trainable_params)
    assert len({id(param) for param in grouped_params}) == len(grouped_params)


def test_compute_loss_rejects_sequence_conditioning(
    itarflow_model, tiny_itarflow_config
):
    """Sequence conditioning is not supported yet and must raise."""
    cfg = tiny_itarflow_config
    targets = torch.randn(2, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(2, cfg.static_in)

    with pytest.raises(ValueError):
        itarflow_model.compute_loss(
            static, targets, targets, pred_and_obs_idx=([0], [1])
        )


def test_create_itarflow_factory():
    """The Hydra entry point builds a working model from kwargs."""
    model = create_itarflow(
        num_timesteps=8,
        out_channels=2,
        channels=16,
        num_layers=2,
        num_blocks=2,
        num_heads=2,
        static_in=3,
        num_static_tokens=4,
    )
    assert isinstance(model, ImprovedTARFlow)

    x = torch.randn(2, 8, 2)
    static = torch.randn(2, 3)
    z, total_logdet = model(x, static)
    assert z.shape == x.shape
    assert total_logdet.shape == (2,)


def test_zero_init_residual_identity(itarflow_model, tiny_itarflow_config):
    itarflow_model.eval()
    cfg = tiny_itarflow_config
    torch.manual_seed(2)
    targets = torch.randn(2, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(2, cfg.static_in)

    with torch.no_grad():
        actual, _ = itarflow_model.forward(targets, static)
    assert torch.allclose(actual, targets, atol=1e-6), (
        f"iTarFlow should be identity map at init, max abs diff: {(actual - targets).abs().max()}"
    )


def test_single_batch_overfit(itarflow_model, tiny_itarflow_config):
    itarflow_model.train()
    cfg = tiny_itarflow_config
    targets = 3 * torch.randn(100, cfg.num_timesteps, cfg.out_channels)
    static = torch.randn(100, cfg.static_in)
    optimizer = torch.optim.AdamW(itarflow_model.parameters(), lr=3e-3)

    for _ in range(200):
        optimizer.zero_grad()
        loss = itarflow_model.compute_loss(
            static,
            None,
            targets,
        )
        loss.backward()
        optimizer.step()

    # Lower bound of MLE objective is the entropy of the data distribution
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # Hence, overfitting to finite samples would require the nll to be lower than the entropy of the data generating
    # distribution

    # Since our data distribution is a multivariate gaussian, the entropy is:
    # H = n/2* log (2*pi*e) + 1/2 log det(\sigma)

    entropy_of_data_distribution = (
        (cfg.num_timesteps * cfg.out_channels) / 2
    ) * math.log(2 * math.pi * math.e) + (1 / 2) * math.log(
        cfg.num_timesteps * cfg.out_channels * 9
    )
    final_loss = loss.item()

    # Now we re-scale our final loss since tarflow mean reduces along the dims
    nll = final_loss * (cfg.num_timesteps * cfg.out_channels)

    assert nll < entropy_of_data_distribution, (
        "NLL is not lower than data distribution entropy, model hasn't overfit"
    )
