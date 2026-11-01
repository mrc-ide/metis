from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from models.sit import (
    FlowMatchingTransformer,
    FlowMatchingTransformerConfig,
    create_flow_matching_transformer,
)
from models.utils.cross_attention import CrossAttentionConfig


def make_tiny_config():
    return FlowMatchingTransformerConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.1,
        in_and_out_biases=True,
    )


def test_forward_pass_shape():
    """Velocity output shape should match (B, T, input_dim)."""
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    out = model(x, t, y)
    assert out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {out.shape}"
    )


def test_zero_init_output():
    """The zero-initialised final layer should produce zero initial velocity."""
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    model.eval()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        out = model(x, t, y)
    assert out.abs().max() < 1e-5, (
        f"Expected near-zero output at init, got max {out.abs().max()}"
    )


def test_gradient_flow():
    """Backprop through the model should produce non-zero gradients.

    At init, adaLN zero-init means gates are zero, so gradients don't flow
    back through the blocks to input_projection. We check final_layer gets
    gradients at init, then verify input_projection gets gradients after one
    optimizer step (once gates are non-zero).
    """
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    model.train()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)

    # At init: final layer should get gradients even with zero-init
    model.zero_grad()
    out = model(x, t, y)
    loss = out.sum()
    loss.backward()
    assert model.final_layer.linear.weight.grad is not None
    assert torch.any(model.final_layer.linear.weight.grad != 0), (
        "final_layer should get non-zero gradients at init"
    )

    # After one optimizer step, gates open up and input_projection should get gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.step()
    model.zero_grad()
    out = model(x, t, y)
    out.sum().backward()
    assert model.input_projection.weight.grad is not None
    assert torch.any(model.input_projection.weight.grad != 0), (
        "input_projection should get non-zero gradients after one step"
    )


def test_conditioning_dropout():
    """With force_drop_ids=1, conditioning should be the null embedding (zeros)."""
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    B = 3
    cond = torch.randn(B, cfg.cond_dim)

    # Force all dropped
    force_drop = torch.ones(B)
    dropped = model.cond_embedder(cond, train=False, force_drop_ids=force_drop)
    assert torch.allclose(dropped, torch.zeros_like(dropped)), (
        "Dropped embeddings should be zero"
    )

    # Force none dropped
    force_keep = torch.zeros(B)
    kept = model.cond_embedder(cond, train=False, force_drop_ids=force_keep)
    assert not torch.allclose(kept, torch.zeros_like(kept)), (
        "Kept embeddings should be non-zero"
    )


def test_flow_matching_loss_finite_and_differentiable():
    """compute_loss should return a finite scalar with gradients."""
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    model.train()
    B, T = 4, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)

    model.zero_grad()

    loss = model.compute_loss(static_input=y, x=None, targets=x_1)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"


def test_sampling_shape():
    """sampling should produce output with the same shape as input."""
    cfg = make_tiny_config()
    model = FlowMatchingTransformer(cfg)
    model.eval()
    B, T = 2, cfg.sequence_len
    x_0 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        x_out = model.sample(x_0, y, num_steps=5, method="euler")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )

    with torch.no_grad():
        x_out = model.sample(x_0, y, num_steps=5, method="heun")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )


def test_single_batch_overfit():
    """FlowMatchingTransformer should memorise a single batch."""
    torch.manual_seed(42)

    cfg = FlowMatchingTransformerConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
    )
    model = FlowMatchingTransformer(cfg)
    model.train()

    assert (
        cfg.sequence_len is not None
        and cfg.data_dim is not None
        and cfg.cond_dim is not None
    )
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
    target_velocity = x_1 - x_0

    # We don't have control of the rng when we use .compute_loss
    # so we lift the loss function out

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(200):
        optimizer.zero_grad()
        velocity = model(x_t, t, y)
        loss = torch.nn.functional.mse_loss(velocity, target_velocity)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 1e-3, f"Model failed to overfit single batch, loss={final_loss}"


def test_cross_attention_training():
    torch.manual_seed(42)
    seq_cond_dim = 2
    cfg = FlowMatchingTransformerConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
        cross_attn=CrossAttentionConfig(n_head=4, cond_dim=seq_cond_dim, n_kv_head=4),
    )
    model = FlowMatchingTransformer(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    cond_seq = torch.randn(B, cfg.sequence_len, seq_cond_dim)

    # Forward pass produces correct shape
    x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
    output = model(x_t, t, y, cond_seq=cond_seq)
    assert output.shape == (B, cfg.sequence_len, cfg.data_dim)

    # Loss computation works
    target_velocity = x_1 - x_0
    loss = torch.nn.functional.mse_loss(output, target_velocity)
    assert loss.ndim == 0, "Loss not a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    # Backward pass produces gradients and some are non-zero
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = model(x_t, t, y, cond_seq=cond_seq)
    torch.nn.functional.mse_loss(output, target_velocity).backward()

    for index, block in enumerate(model.blocks):
        for name, parameter in block.cross_attn.named_parameters():
            if name.endswith("weight"):
                assert parameter.grad is not None, (
                    f"No gradient for block {index} cross-attention {name}"
                )
                assert torch.isfinite(parameter.grad).all()
                assert torch.any(parameter.grad != 0), (
                    f"Zero gradient for block {index} cross-attention {name} after one step"
                )


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_loss_uses_linear_path_and_predicted_channels(monkeypatch, conditioned, masked):
    cfg = make_tiny_config()
    cfg.cond_dropout_prob = 0.0
    if conditioned:
        cfg.cross_attn = CrossAttentionConfig(n_head=4, n_kv_head=4, cond_dim=2)
    model = FlowMatchingTransformer(cfg)
    batch, length = 3, cfg.sequence_len
    targets = torch.randn(batch, length, cfg.data_dim + (2 if conditioned else 0))
    static = torch.randn(batch, cfg.cond_dim)
    pred_idx, obs_idx = [4, 0, 3, 1], [5, 2]
    predicted = targets[:, :, pred_idx] if conditioned else targets
    noise = torch.randn_like(predicted)
    times = torch.tensor([0.0, 0.4, 1.0])
    monkeypatch.setattr(torch, "randn_like", lambda *args, **kwargs: noise)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: times)
    captured = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append((args, kwargs)), with_kwargs=True
    )
    mask = torch.tensor([1.0, 0.0, 1.0]) if masked else None
    try:
        loss = model.compute_loss(
            static,
            None,
            targets,
            mask=mask,
            pred_and_obs_idx=(pred_idx, obs_idx) if conditioned else None,
        )
    finally:
        handle.remove()

    (actual_path, actual_times, actual_static), kwargs = captured[0]
    # Endpoints and an interior time must all use the same linear path.
    torch.testing.assert_close(actual_path[0], noise[0])
    torch.testing.assert_close(actual_path[1], 0.6 * noise[1] + 0.4 * predicted[1])
    torch.testing.assert_close(actual_path[2], predicted[2])
    torch.testing.assert_close(actual_times, times)
    torch.testing.assert_close(actual_static, static)
    if conditioned:
        torch.testing.assert_close(kwargs["cond_seq"], targets[:, :, obs_idx])
    else:
        assert kwargs["cond_seq"] is None

    # Initial velocity is zero: the loss is the squared target velocity.
    per_example = (predicted - noise).square().mean(dim=(1, 2))
    expected = per_example[[0, 2]].mean() if masked else per_example.mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.final_layer.linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "method,expected,calls", [("euler", 0.25, 2), ("heun", 0.375, 3)]
)
def test_sampler_time_dependent_velocity(monkeypatch, method, expected, calls):
    model = FlowMatchingTransformer(make_tiny_config())
    evaluated_times = []

    def velocity(x, t, y, cond_seq=None):
        evaluated_times.append(t.item())
        return torch.ones_like(x) * t

    monkeypatch.setattr(model, "forward", velocity)
    noise = torch.zeros(2, 16, 4)
    result = model.sample(noise, torch.zeros(2, 3), num_steps=2, method=method)
    torch.testing.assert_close(result, torch.full_like(noise, expected))
    assert len(evaluated_times) == calls
    assert 1.0 not in evaluated_times  # Heun's final step has no correction.


def test_factory_accepts_plain_cross_attention_mapping():
    config = asdict(make_tiny_config())
    config["cross_attn"] = {"n_head": 4, "n_kv_head": 4, "cond_dim": 2}
    model = create_flow_matching_transformer(**config)
    output = model(
        torch.randn(2, 16, 4),
        torch.rand(2),
        torch.randn(2, 3),
        cond_seq=torch.randn(2, 16, 2),
    )
    assert output.shape == (2, 16, 4)


@pytest.mark.parametrize(
    "preset,blocks,heads,dim",
    [("small", 6, 4, 64), ("default", 12, 8, 64), ("large", 16, 16, 128)],
)
def test_hydra_presets(preset, blocks, heads, dim):
    from hydra import compose, initialize
    from hydra.utils import instantiate

    import train  # noqa: F401 - registers structured model schemas

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(overrides=[f"+model=flow_matching_{preset}"])
    assert (cfg.model.n_blocks, cfg.model.n_head, cfg.model.embd_dim) == (
        blocks,
        heads,
        dim,
    )
    model = instantiate(cfg.model)
    assert isinstance(model, FlowMatchingTransformer)
    assert len(model.blocks) == blocks
    assert "sit" not in cfg.model and "interpolant" not in cfg.model


@pytest.mark.parametrize("dataset", ["hrp_2026_02_02", "truncated_magenta_2026_02_24"])
def test_hydra_training_and_sweep_configs(dataset):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    import train  # noqa: F401 - registers structured model schemas

    with initialize(version_base=None, config_path=f"../configs/train/{dataset}"):
        cfg = compose(config_name="flow_matching_sota")
    model_cfg = cfg.model
    assert model_cfg._target_ == "models.sit.create_flow_matching_transformer"
    assert "sit" not in model_cfg
    assert model_cfg.n_head == 16

    sweep = OmegaConf.load(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "sweep"
        / dataset
        / "flow_matching.yaml"
    )
    overrides = ["model=flow_matching_default"]
    for name, spec in sweep.parameters.items():
        if name.startswith("model."):
            value = spec.get("value", spec.get("min", None))
            if value is None:
                value = spec["values"][0]
            overrides.append(f"{name}={value}")
    with initialize(version_base=None, config_path="../configs/train"):
        cfg = compose(config_name="default", overrides=overrides)
    assert cfg.model.n_blocks == 6


def test_hydra_sequence_conditioning_config():
    from hydra import compose, initialize
    from hydra.utils import instantiate

    import train  # noqa: F401 - registers structured model schemas

    with initialize(version_base=None, config_path="../configs/train"):
        cfg = compose(
            config_name="default",
            overrides=[
                "model=flow_matching_small",
                "+model/cross_attn=default",
                "model.cross_attn.cond_dim=2",
            ],
        )
    model = instantiate(cfg.model)
    assert model.has_cross_attn
