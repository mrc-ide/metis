import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from inference import Surrogate


def make_checkpoint(kind, conditioned=False):
    cross_attn = (
        {"n_head": 2, "n_kv_head": 2, "cond_dim": 2, "qkv_bias": True}
        if conditioned
        else None
    )
    channels = 2 if conditioned else 4
    configs = {
        "mlp": {
            "_target_": "models.mlp.create_residual_mlp",
            "input_dim": 2,
            "output_dim": 16,
            "num_blocks": 1,
            "hidden_dim": 16,
            "num_layers_per_block": 2,
        },
        "transformer": {
            "_target_": "models.transformer.create_autoregressive_transformer",
            "static_input_dim": 2,
            "x_input_dim": channels,
            "sequence_len": 5,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "cross_attn": cross_attn,
        },
        "flow_matching": {
            "_target_": "models.sit.create_flow_matching_transformer",
            "solver": "heun",
            "num_steps": 2,
            "sequence_len": 4,
            "data_dim": channels,
            "cond_dim": 2,
            "n_blocks": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "embd_dim": 16,
            "cross_attn": cross_attn,
        },
        "itarflow": {
            "_target_": "models.itarflow.create_itarflow",
            "num_timesteps": 4,
            "out_channels": channels,
            "channels": 16,
            "patch_size": 1,
            "num_layers": 1,
            "num_blocks": 1,
            "num_heads": 2,
            "static_in": 2,
            "num_static_tokens": 2,
            "cross_attn": cross_attn,
        },
    }
    config = {
        "model": configs[kind],
        "dataset": {
            "path": "/nonexistent/training/data.parquet",
            "surrogate_input_variables": ["x", "y"],
            "output_variables": ["a", "b", "c", "d"],
            "observed_output_variables": ["c", "a"] if conditioned else None,
            "num_timesteps": 4,
        },
        "sampling": {"use_kv_cache": True},
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = instantiate(OmegaConf.create(config["model"]))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.05)
    return {
        "config": config,
        "model": model.state_dict(),
        "normalisation": {
            "inputs": {
                "mean": torch.tensor([[1.0] * 4, [5.0] * 4]),
                "std": torch.tensor([[2.0] * 4, [3.0] * 4]),
            },
            "targets": {
                "mean": torch.arange(16.0).reshape(4, 4) + 10,
                "std": (torch.arange(16.0).reshape(4, 4) + 1) / 10,
            },
        },
    }


def load(tmp_path, checkpoint):
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return Surrogate.from_checkpoint(path)


@pytest.mark.parametrize(
    "kind,conditioned",
    [
        ("mlp", False),
        ("transformer", False),
        ("transformer", True),
        ("flow_matching", False),
        ("flow_matching", True),
        ("itarflow", False),
        ("itarflow", True),
    ],
)
def test_predictions_match_manual_inference(tmp_path, kind, conditioned):
    checkpoint = make_checkpoint(kind, conditioned)
    surrogate = load(tmp_path, checkpoint)
    raw = torch.tensor([[3.0, 11.0], [7.0, 2.0]])
    observed = torch.arange(16.0).reshape(2, 4, 2) if conditioned else None
    stochastic = kind in ("flow_matching", "itarflow")
    samples = 3 if stochastic else 1
    kwargs = {"num_samples": samples, "seed": 42} if stochastic else {}
    rng_state = torch.random.get_rng_state()
    actual = surrogate.predict(raw, observed=observed, **kwargs)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)

    pred_idx, obs_idx = ([1, 3], [2, 0]) if conditioned else ([0, 1, 2, 3], [])
    stats = checkpoint["normalisation"]
    static = (raw - stats["inputs"]["mean"][:, 0]) / stats["inputs"]["std"][:, 0]
    mean, std = stats["targets"]["mean"].T, stats["targets"]["std"].T
    cond = (observed - mean[:, obs_idx]) / std[:, obs_idx] if conditioned else None
    shape = (2 * samples, 4, len(pred_idx))
    model = surrogate.model
    with torch.no_grad():
        if kind == "mlp":
            expected = model(static).reshape(shape)
        elif kind == "transformer":
            expected = torch.cat(
                list(
                    model.generate(
                        static,
                        max_output_steps=4,
                        conditioning_sequence=cond,
                    )
                ),
                dim=1,
            )
        else:
            noise = torch.randn(shape, generator=torch.Generator().manual_seed(42))
            static = static.repeat_interleave(samples, dim=0)
            cond = cond.repeat_interleave(samples, dim=0) if cond is not None else None
            if kind == "flow_matching":
                expected = model.sample(
                    noise, static, num_steps=2, method="heun", cond_seq=cond
                )
            else:
                expected, _ = model.reverse(noise, static, cond_seq=cond)
        expected = (expected * std[:, pred_idx] + mean[:, pred_idx]).reshape(
            2, samples, 4, -1
        )
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert not model.training
    assert surrogate.input_names == ["x", "y"]
    assert surrogate.observed_names == (["c", "a"] if conditioned else [])
    assert surrogate.prediction_names == (
        ["b", "d"] if conditioned else ["a", "b", "c", "d"]
    )
    if stochastic:
        torch.testing.assert_close(
            actual, surrogate.predict(raw, observed=observed, **kwargs)
        )
        assert not torch.equal(actual[:, 0], actual[:, 1])
    if conditioned:
        changed = surrogate.predict(raw, observed=observed + 10, **kwargs)
        assert not torch.equal(actual, changed)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_samples": 0}, "positive integer"),
        ({"num_samples": 2}, "deterministic"),
        ({"seed": 1}, "deterministic"),
        ({"solver": "heun"}, "only by Flow Matching"),
        ({"num_solver_steps": 2}, "only by Flow Matching"),
        ({"use_kv_cache": False}, "only by autoregressive"),
        ({"observed": torch.zeros(1, 4, 2)}, "does not accept observed"),
    ],
)
def test_incompatible_options(tmp_path, kwargs, message):
    surrogate = load(tmp_path, make_checkpoint("mlp"))
    with pytest.raises(ValueError, match=message):
        surrogate.predict(torch.zeros(1, 2), **kwargs)


def test_conditioning_and_input_errors(tmp_path):
    surrogate = load(tmp_path, make_checkpoint("itarflow", conditioned=True))
    with pytest.raises(ValueError, match="requires observed.*'c', 'a'"):
        surrogate.predict(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="observed must have shape"):
        surrogate.predict(torch.zeros(2, 2), observed=torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match="inputs must have shape.*'x', 'y'"):
        surrogate.predict(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="finite"):
        surrogate.predict(torch.full((2, 2), float("nan")))


def test_flow_matching_requires_settings_and_accepts_overrides(tmp_path):
    checkpoint = make_checkpoint("flow_matching")
    checkpoint["config"]["model"].update(num_steps=None, solver=None)
    surrogate = load(tmp_path, checkpoint)
    inputs = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="requires num_solver_steps"):
        surrogate.predict(inputs)
    with pytest.raises(ValueError, match="requires solver"):
        surrogate.predict(inputs, num_solver_steps=1)
    assert surrogate.predict(inputs, num_solver_steps=1, solver="euler").shape == (
        1,
        1,
        4,
        4,
    )


@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing_stats", "missing 'normalisation'"),
        ("missing_std", "normalisation.targets.mean and .std"),
        ("wrong_shape", "tensor of shape"),
        ("zero_std", "strictly positive"),
        ("wrong_timesteps", "dataset requires"),
    ],
)
def test_invalid_checkpoints(tmp_path, defect, message):
    checkpoint = make_checkpoint("mlp")
    if defect == "missing_stats":
        del checkpoint["normalisation"]
    elif defect == "missing_std":
        del checkpoint["normalisation"]["targets"]["std"]
    elif defect == "wrong_shape":
        checkpoint["normalisation"]["inputs"]["mean"] = torch.zeros(2)
    elif defect == "zero_std":
        checkpoint["normalisation"]["targets"]["std"][0, 0] = 0
    else:
        checkpoint["config"]["dataset"]["num_timesteps"] = 5
    with pytest.raises(ValueError, match=message):
        load(tmp_path, checkpoint)
