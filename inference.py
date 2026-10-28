import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models.itarflow import ImprovedTARFlow
from models.mlp import ResidualMLP
from models.sit import InterpolantTransformer
from models.transformer import Transformer


class Surrogate:
    """Checkpoint-backed predictions in unstandardized dataset units."""

    def __init__(self, model, config, normalisation):
        self.model = model.eval()
        self.config = config
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        try:
            ds = config.dataset
            self.input_names = list(ds.surrogate_input_variables)
            output_names = list(ds.output_variables)
            self.num_timesteps = ds.num_timesteps
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"Checkpoint is missing dataset metadata: {exc}") from exc
        self.observed_names = list(ds.get("observed_output_variables") or [])
        if not set(self.observed_names).issubset(output_names):
            raise ValueError("Observed variables must be present in output_variables")
        for names in (self.input_names, output_names, self.observed_names):
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Checkpoint contains duplicate variable names: {names}"
                )
        self.prediction_names = [
            n for n in output_names if n not in self.observed_names
        ]
        if not self.prediction_names or self.num_timesteps <= 0:
            raise ValueError(
                "Checkpoint requires predicted variables and a positive num_timesteps"
            )

        self._validate_model()
        input_mean, input_std = self._stats(
            normalisation, "inputs", len(self.input_names)
        )
        output_mean, output_std = self._stats(
            normalisation, "targets", len(output_names)
        )
        obs_idx = [output_names.index(n) for n in self.observed_names]
        pred_idx = [output_names.index(n) for n in self.prediction_names]
        self.input_mean, self.input_std = input_mean[:, 0], input_std[:, 0]
        self.observed_mean, self.observed_std = (
            output_mean[obs_idx].T,
            output_std[obs_idx].T,
        )
        self.output_mean, self.output_std = (
            output_mean[pred_idx].T,
            output_std[pred_idx].T,
        )

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("config", "model", "normalisation"):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint {path} is missing '{key}'")
        config = OmegaConf.create(checkpoint["config"])
        if "model" not in config:
            raise ValueError(f"Checkpoint {path} is missing config.model")
        model = instantiate(config.model)
        model.load_state_dict(checkpoint["model"])
        return cls(model.to(device), config, checkpoint["normalisation"])

    def _validate_model(self):
        c = self.config.model
        n_in, n_out, t = (
            len(self.input_names),
            len(self.prediction_names),
            self.num_timesteps,
        )
        cross_attn = c.get("cross_attn")
        if isinstance(self.model, ResidualMLP):
            expected = {"input_dim": n_in, "output_dim": t * n_out}
        elif isinstance(self.model, Transformer):
            expected = {
                "static_input_dim": n_in,
                "x_input_dim": n_out,
                "sequence_len": t + 1,
            }
        elif isinstance(self.model, InterpolantTransformer):
            c = c.sit
            cross_attn = c.get("cross_attn")
            expected = {"cond_dim": n_in, "data_dim": n_out, "sequence_len": t}
        elif isinstance(self.model, ImprovedTARFlow):
            expected = {"static_in": n_in, "out_channels": n_out, "num_timesteps": t}
        else:
            raise TypeError(f"Unsupported model type: {type(self.model).__name__}")
        for key, value in expected.items():
            if c.get(key) != value:
                raise ValueError(
                    f"Checkpoint model {key}={c.get(key)}; dataset requires {value}"
                )
        if bool(self.observed_names) != (cross_attn is not None):
            raise ValueError(
                "Checkpoint observed variables and model cross-attention configuration disagree"
            )
        if cross_attn is not None and cross_attn.cond_dim != len(self.observed_names):
            raise ValueError(
                "Checkpoint cross-attention cond_dim does not match observed variables"
            )

    def _stats(self, normalisation, name, channels):
        try:
            mean, std = normalisation[name]["mean"], normalisation[name]["std"]
        except KeyError as exc:
            raise ValueError(
                f"Checkpoint requires normalisation.{name}.mean and .std"
            ) from exc
        shape = (channels, self.num_timesteps)
        for key, value in (("mean", mean), ("std", std)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(
                    f"normalisation.{name}.{key} must be a tensor of shape {shape}"
                )
            if not torch.isfinite(value).all() or (key == "std" and (value <= 0).any()):
                raise ValueError(
                    f"normalisation.{name}.{key} must be finite, with strictly positive stds"
                )
        return mean.to(device=self.device, dtype=self.dtype), std.to(
            device=self.device, dtype=self.dtype
        )

    def _tensor(self, value, name, shape, names):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = (
                tuple(value.shape)
                if isinstance(value, torch.Tensor)
                else type(value).__name__
            )
            raise ValueError(
                f"{name} must have shape {shape}; received {actual}. Variable order: {names}"
            )
        value = value.to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    @torch.inference_mode()
    def predict(
        self,
        inputs: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        num_solver_steps: int | None = None,
        solver: str | None = None,
        use_kv_cache: bool | None = None,
    ) -> torch.Tensor:
        """Accept raw [B, inputs] and optional [B, T, observed]; return [B, samples, T, predicted]."""
        stochastic = isinstance(self.model, (InterpolantTransformer, ImprovedTARFlow))
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if not stochastic and (num_samples != 1 or seed is not None):
            raise ValueError(
                f"{type(self.model).__name__} is deterministic; use num_samples=1 and omit seed"
            )
        if not isinstance(self.model, InterpolantTransformer) and (
            num_solver_steps is not None or solver is not None
        ):
            raise ValueError(
                "num_solver_steps and solver are supported only by interpolant models"
            )
        if not isinstance(self.model, Transformer) and use_kv_cache is not None:
            raise ValueError(
                "use_kv_cache is supported only by autoregressive transformers"
            )
        if isinstance(self.model, InterpolantTransformer):
            num_solver_steps = (
                self.config.model.get("num_steps")
                if num_solver_steps is None
                else num_solver_steps
            )
            solver = self.config.model.get("solver") if solver is None else solver
            if type(num_solver_steps) is not int or num_solver_steps < 1:
                raise ValueError(
                    "Interpolant inference requires num_solver_steps=<positive integer>, supplied explicitly or saved as model.num_steps in the checkpoint"
                )
            if solver not in ("euler", "heun"):
                raise ValueError(
                    "Interpolant inference requires solver='euler' or 'heun', supplied explicitly or saved in the checkpoint"
                )
        if isinstance(self.model, Transformer):
            if use_kv_cache is None:
                use_kv_cache = self.config.get("sampling", {}).get(
                    "use_kv_cache", False
                )
            if type(use_kv_cache) is not bool:
                raise ValueError("use_kv_cache must be a boolean")

        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 2
            or inputs.shape[0] == 0
        ):
            raise ValueError(
                f"inputs must have shape [batch, {len(self.input_names)}] with a nonempty batch. Variable order: {self.input_names}"
            )
        batch = inputs.shape[0]
        inputs = self._tensor(
            inputs, "inputs", (batch, len(self.input_names)), self.input_names
        )
        inputs = (inputs - self.input_mean) / self.input_std
        if self.observed_names:
            shape = (batch, self.num_timesteps, len(self.observed_names))
            if observed is None:
                raise ValueError(
                    f"This checkpoint requires observed with shape {shape}, in unstandardized units. Variable order: {self.observed_names}"
                )
            observed = self._tensor(observed, "observed", shape, self.observed_names)
            observed = (observed - self.observed_mean) / self.observed_std
        elif observed is not None:
            raise ValueError("This checkpoint does not accept observed sequences")

        shape = (batch * num_samples, self.num_timesteps, len(self.prediction_names))
        if isinstance(self.model, ResidualMLP):
            output = self.model(inputs).reshape(shape)
        elif isinstance(self.model, Transformer):
            output = torch.cat(
                list(
                    self.model.generate(
                        inputs,
                        max_output_steps=self.num_timesteps,
                        use_kv_cache=use_kv_cache,
                        conditioning_sequence=observed,
                    )
                ),
                dim=1,
            )
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(
                shape, device=self.device, dtype=self.dtype, generator=generator
            )
            inputs = inputs.repeat_interleave(num_samples, dim=0)
            if observed is not None:
                observed = observed.repeat_interleave(num_samples, dim=0)
            if isinstance(self.model, InterpolantTransformer):
                output, _ = self.model.sample(
                    noise,
                    inputs,
                    num_steps=num_solver_steps,
                    method=solver,
                    cond_seq=observed,
                )
            else:
                output, _ = self.model.reverse(noise, inputs, cond_seq=observed)
        output = output * self.output_std + self.output_mean
        return output.reshape(
            batch, num_samples, self.num_timesteps, len(self.prediction_names)
        )
