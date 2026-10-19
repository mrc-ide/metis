import math
from dataclasses import replace
from typing import cast

import pytest
import torch

from models.itarflow import (
    ImprovedTARFlow,
    InContextMetaBlock,
    apply_rotary_pos_emb,
    create_itarflow,
    precompute_rope_freqs,
)
from models.utils.cross_attention import CrossAttentionConfig


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
    """An unconditional model must reject supplied sequence conditioning."""
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


def _conditioned_model(config, **kwargs):
    torch.manual_seed(0)
    model = ImprovedTARFlow(
        replace(
            config,
            cross_attn=CrossAttentionConfig(n_head=4, n_kv_head=2, cond_dim=3),
            **kwargs,
        )
    )
    _randomize_flow(model)
    for block in model.blocks:
        assert isinstance(block, InContextMetaBlock)
        torch.nn.init.normal_(block.cross_attn.c_proj.weight, std=0.1)
    return model


def test_conditioned_tarflow_learns_2z(tiny_itarflow_config):
    """Learn a conditional distribution whose mean is twice the observation."""
    torch.manual_seed(42)
    cfg = replace(
        tiny_itarflow_config,
        num_timesteps=4,
        out_channels=1,
        channels=32,
        num_layers=1,
        num_blocks=3,
        clip_range=5.0,
        cross_attn=CrossAttentionConfig(n_head=4, n_kv_head=2, cond_dim=1),
    )
    model = ImprovedTARFlow(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    noise_std = 0.3

    # Fresh observations every step prevent memorising a fixed training batch.
    for _ in range(300):
        observed = torch.randn(64, cfg.num_timesteps, 1)
        predicted = 2 * observed + noise_std * torch.randn_like(observed)
        targets = torch.cat([predicted, observed], dim=-1)
        static = torch.zeros(64, cfg.static_in)
        loss = model.compute_loss(static, None, targets, pred_and_obs_idx=([0], [1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        observed = torch.randn(64, cfg.num_timesteps, 1)
        predicted = 2 * observed + noise_std * torch.randn_like(observed)
        static = torch.zeros(64, cfg.static_in)
        targets = torch.cat([predicted, observed], dim=-1)
        correct_loss = model.compute_loss(
            static, None, targets, pred_and_obs_idx=([0], [1])
        )
        shuffled_targets = torch.cat([predicted, observed.roll(1, dims=0)], dim=-1)
        shuffled_loss = model.compute_loss(
            static, None, shuffled_targets, pred_and_obs_idx=([0], [1])
        )

        # Assess actual samples, rather than treating zero latent noise as a mean.
        repetitions = 64
        samples, _ = model.reverse(
            torch.randn(64 * repetitions, cfg.num_timesteps, 1),
            static.repeat_interleave(repetitions, dim=0),
            observed.repeat_interleave(repetitions, dim=0),
        )
        samples = samples.reshape(64, repetitions, cfg.num_timesteps, 1)
        mean_mse = (samples.mean(dim=1) - 2 * observed).square().mean().item()

    assert mean_mse < 0.3, f"Conditional mean failed to learn 2z: MSE={mean_mse:.4f}"
    assert correct_loss + 0.5 < shuffled_loss, (
        f"Model did not learn to use observations: correct={correct_loss.item():.4f}, "
        f"shuffled={shuffled_loss.item():.4f}"
    )


@pytest.mark.parametrize(
    "block_index, expected_positions", [(0, [1, 2, 3, 4]), (1, [3, 2, 1, 0])]
)
def test_reverse_uses_original_target_positions(
    tiny_itarflow_config, block_index, expected_positions
):
    model = _conditioned_model(tiny_itarflow_config, num_timesteps=5)
    block = model.blocks[block_index]
    calls = []

    def record_query(module, args, kwargs):
        assert args[1] is None  # Observations have already populated the cache.
        assert kwargs["kv_cache"].populated[module.layer_idx]
        calls.append((kwargs["current_x_pos"], tuple(x.clone() for x in args[2])))

    handle = block.cross_attn.register_forward_pre_hook(record_query, with_kwargs=True)
    try:
        with torch.no_grad():
            model.reverse(torch.randn(2, 5, 2), torch.randn(2, 3), torch.randn(2, 5, 3))
    finally:
        handle.remove()

    assert [position for position, _ in calls] == expected_positions
    # Compute expected angles directly, not from the model's RoPE buffers/helper.
    head_dim = block.cross_attn.head_dim
    frequencies = torch.tensor(
        [10000.0 ** (-2 * j / head_dim) for j in range(head_dim // 2)]
    )
    for (_, (cos, sin)), target in zip(calls, expected_positions, strict=True):
        angles = (target * frequencies).reshape(1, 1, 1, -1)
        torch.testing.assert_close(cos, angles.cos())
        torch.testing.assert_close(sin, angles.sin())


@pytest.mark.parametrize("timesteps", [1, 5, 8])
def test_conditioned_round_trip_and_logdet(tiny_itarflow_config, timesteps):
    model = _conditioned_model(tiny_itarflow_config, num_timesteps=timesteps)
    x = torch.randn(2, timesteps, 2)
    static = torch.randn(2, 3)
    cond = torch.randn(2, timesteps, 3)
    with torch.no_grad():
        z, logdet = model(x, static, cond)
        reconstructed, reverse_logdet = model.reverse(z, static, cond)
        samples, _ = model.reverse(x, static, cond)
        reconstructed_z, _ = model(samples, static, cond)
    torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reconstructed_z, x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logdet, -reverse_logdet, atol=1e-5, rtol=1e-5)


def test_conditioned_block_observation_causality(tiny_itarflow_config):
    """Both orientations use contemporaneous observations, never later ones."""
    model = _conditioned_model(tiny_itarflow_config)
    x = torch.randn(2, 8, 2)
    static = torch.randn(2, 3)
    cond = torch.randn(2, 8, 3)
    t = 3  # transformed in both orientations
    for block in model.blocks:
        baseline = block(x, static, cond)[0]
        future = cond.clone()
        future[:, t + 1 :] += torch.randn_like(future[:, t + 1 :]) * 10
        actual = block(x, static, future)[0]
        torch.testing.assert_close(actual[:, : t + 1], baseline[:, : t + 1])
        present = cond.clone()
        present[:, t] += torch.randn_like(present[:, t]) * 10
        actual = block(x, static, present)[0]
        assert not torch.allclose(actual[:, t], baseline[:, t])


def test_conditioned_jacobian(tiny_itarflow_config):
    model = _conditioned_model(tiny_itarflow_config, num_timesteps=3, out_channels=1)
    x = torch.randn(1, 3, 1, requires_grad=True)
    static = torch.randn(1, 3)
    cond = torch.randn(1, 3, 3)
    # SDPA's math backend supports the derivatives used by the Jacobian check.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        jac = torch.autograd.functional.jacobian(
            lambda value: model(value, static, cond)[0], x
        ).reshape(3, 3)
        _, logdet = model(x, static, cond)
    torch.testing.assert_close(
        torch.linalg.slogdet(jac).logabsdet, logdet[0] * 3, atol=1e-5, rtol=1e-5
    )


def test_conditioned_loss_and_learning(tiny_itarflow_config):
    cfg = replace(
        tiny_itarflow_config,
        cross_attn=CrossAttentionConfig(n_head=4, n_kv_head=2, cond_dim=3),
    )
    model = ImprovedTARFlow(cfg)
    targets = torch.randn(2, 8, 5)
    static = torch.randn(2, 3)
    pred, obs = [1, 3], [0, 2, 4]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(4):
        optimizer.zero_grad()
        loss = model.compute_loss(static, None, targets, pred_and_obs_idx=(pred, obs))
        direct = model.compute_loss(
            static[:1], None, targets[:1], pred_and_obs_idx=(pred, obs)
        )
        masked = model.compute_loss(
            static,
            None,
            targets,
            mask=torch.tensor([1.0, 0.0]),
            pred_and_obs_idx=(pred, obs),
        )
        torch.testing.assert_close(masked, direct)
        loss.backward()
        optimizer.step()
    for block in model.blocks:
        assert isinstance(block, InContextMetaBlock)
        for parameter in (block.cross_attn.c_q.weight, block.cond_proj.weight):
            assert parameter.grad is not None
            assert parameter.grad.abs().sum() > 0


def test_conditioned_zero_head_matches_unconditional(tiny_itarflow_config):
    plain = ImprovedTARFlow(tiny_itarflow_config)
    _randomize_flow(plain)
    conditioned = _conditioned_model(tiny_itarflow_config)
    conditioned.load_state_dict(plain.state_dict(), strict=False)
    for block in conditioned.blocks:
        torch.nn.init.zeros_(block.cross_attn.c_proj.weight)
    x, static, cond = torch.randn(2, 8, 2), torch.randn(2, 3), torch.randn(2, 8, 3)
    for actual, expected in zip(conditioned(x, static, cond), plain(x, static)):
        torch.testing.assert_close(actual, expected)


def test_conditioning_validation(tiny_itarflow_config):
    model = _conditioned_model(tiny_itarflow_config)
    x, static = torch.randn(2, 8, 2), torch.randn(2, 3)
    for cond in (
        None,
        torch.randn(1, 8, 3),
        torch.randn(2, 7, 3),
        torch.randn(2, 8, 2),
    ):
        for method in (model.forward, model.reverse):
            with pytest.raises(ValueError, match="cond_seq"):
                method(x, static, cond)
    with pytest.raises(ValueError, match="patch_size"):
        _conditioned_model(tiny_itarflow_config, patch_size=2)


def test_conditioned_hydra_config():
    from hydra import compose, initialize
    from hydra.utils import instantiate

    import train  # noqa: F401 - registers structured model schemas

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(overrides=["+model=itarflow_cross_attn"])
    model = instantiate(cfg.model)
    assert isinstance(model, ImprovedTARFlow)
    assert model.blocks[0].has_cross_attn


def test_conditioned_training_and_sampling_smoke(
    tiny_itarflow_config, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from omegaconf import OmegaConf

    from dataloader import DatasetMeansAndStds
    from train import TrainConfig, TrainingLoss, generate_comparison_plot

    model = _conditioned_model(tiny_itarflow_config, num_timesteps=3, out_channels=1)
    targets = torch.randn(2, 3, 4)
    static = torch.randn(2, 3)
    loss_module = torch.compile(
        TrainingLoss(model, ([1], [0, 2, 3])), backend="eager", fullgraph=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = loss_module(static, targets[:, :-1], targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    path = tmp_path / "conditioned.png"
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "models.itarflow.create_itarflow"},
            "dataset": {
                "num_timesteps": 3,
                "surrogate_input_variables": ["a", "b", "c"],
                "output_variables": ["obs_a", "pred", "obs_b", "obs_c"],
                "observed_output_variables": ["obs_a", "obs_b", "obs_c"],
            },
            "sampling": {"samples_name": str(path)},
        }
    )
    df = pl.DataFrame(
        {
            "param_index": [0] * 6,
            "rep": [0] * 3 + [1] * 3,
            "t": [0, 1, 2] * 2,
            "a": [1.0] * 6,
            "b": [2.0] * 6,
            "c": [3.0] * 6,
            "obs_a": [2.0, 4.0, 6.0] * 2,
            "pred": [10.0, 20.0, 30.0] * 2,
            "obs_b": [3.0, 6.0, 9.0] * 2,
            "obs_c": [4.0, 8.0, 12.0] * 2,
        }
    )
    input_stats = DatasetMeansAndStds(torch.zeros(3, 3), torch.ones(3, 3))
    means = torch.arange(4.0).unsqueeze(1).expand(4, 3)
    stds = torch.arange(1.0, 5.0).unsqueeze(1).expand(4, 3)
    output_stats = DatasetMeansAndStds(means, stds)
    captured = {}
    original_reverse = model.reverse

    def reverse(z, static, cond_seq=None):
        captured["cond"] = cond_seq
        result = original_reverse(z, static, cond_seq)
        captured["samples"] = result[0].detach()
        return result

    monkeypatch.setattr(model, "reverse", reverse)
    with torch.no_grad():
        generate_comparison_plot(
            cast(TrainConfig, cfg),
            model,
            df,
            pl.DataFrame({"param_index": [0], "rep_count": [2]}),
            input_stats,
            output_stats,
            subsample=None,
            device="cpu",
        )
    raw_obs = torch.tensor([[[2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]])
    expected_cond = ((raw_obs - means[[0, 2, 3]]) / stds[[0, 2, 3]]).transpose(1, 2)
    torch.testing.assert_close(captured["cond"], expected_cond)
    expected_samples = captured["samples"][0, :, 0] * stds[1] + means[1]
    np.testing.assert_allclose(
        np.asarray(plt.gcf().axes[0].lines[-1].get_ydata()), expected_samples.numpy()
    )
    plt.close("all")
    assert path.is_file()
