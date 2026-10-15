import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import hydra
import hydra.core.config_store
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import wandb
from dataloader import (
    DatasetConfig,
    DatasetMeansAndStds,
    get_data_iterator,
    get_random_batch,
    make_tensors,
    prepare_splits,
)
from models.itarflow import ImprovedTARFlow, iTarFlowConfig
from models.mlp import ResidualMLP, ResidualMLPConfig
from models.sit import InterpolantTransformer, InterpolantTransformerConfig
from models.transformer import Transformer, TransformerConfig
from models.utils.deterministic import make_deterministic
from models.utils.precision import PrecisionMode, configure_precision, training_autocast


@dataclass
class SamplingConfig:
    generate_samples: bool
    samples_name: str | None
    subsample: int | None
    use_kv_cache: bool = False
    keep_greater_than_zero: bool = False


@dataclass
class LoggerConfig:
    use_wandb: bool
    project: str
    tags: list[str]


@dataclass
class CheckpointerConfig:
    save_every: int | str | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class TrainingLoss(nn.Module):
    """Expose the models' common loss API through a DDP-compatible forward."""

    def __init__(self, model: Any, pred_and_obs_idx):
        super().__init__()
        self.model: Any = model
        self.pred_and_obs_idx = pred_and_obs_idx

    def forward(self, static_input, x, targets, mask=None):
        return self.model.compute_loss(
            static_input=static_input,
            x=x,
            targets=targets,
            mask=mask,
            pred_and_obs_idx=self.pred_and_obs_idx,
        )


def step_seed(seed: int, step: int, runtime: DistributedRuntime, stream: int) -> int:
    """A seed unique to (stream, step, rank).

    Reseeding from this every step makes the RNG a pure function of the step, so
    resuming from a checkpoint needs no saved RNG state. `stream` (0 or 1) keeps
    the two consumers, batch sampling and model noise, on distinct Philox streams.
    """
    return (2 * seed + stream) * 2**32 + step * runtime.world_size + runtime.rank


def save_checkpoint(
    path, cfg, step, model, optimiser, train_losses, val_losses, val_steps
):
    checkpoint = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "train_losses": torch.tensor(train_losses, dtype=torch.float32),
        "val_losses": val_losses,
        "val_steps": val_steps,
    }
    # Write to a temporary file then rename, so an interrupted save never leaves a
    # half-written best.pt/last.pt behind
    checkpoint_dir = os.path.dirname(path) or "."
    fd, temporary_path = tempfile.mkstemp(
        prefix="checkpoint_", suffix=".pt", dir=checkpoint_dir
    )
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


@torch.no_grad()
def evaluate(loss_module, static_input_data, x_data, batch_size, runtime) -> float:
    """Mean loss over the validation set, of which each rank holds its own shard."""
    # Accumulate (loss * valid samples, valid samples) so that padded batches and
    # uneven shards are weighted correctly
    stats = torch.zeros(2, dtype=torch.float64, device=runtime.device)
    for static_input, x, targets, mask in get_data_iterator(
        static_input_data, x_data, batch_size
    ):
        n_valid = mask.sum().double()
        batch_loss = loss_module(static_input, x, targets, mask).double()
        stats += torch.stack((batch_loss * n_valid, n_valid))
    if runtime.distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if stats[1].item() == 0:
        raise ValueError("Validation set is empty")
    return (stats[0] / stats[1]).item()


@dataclass
class TrainConfig:
    max_steps: int
    eval_every: int
    peak_lr: float
    lr_warmup_frac: float
    batch_size: int
    num_gpus: int
    precision: str
    deterministic: bool
    seed: int
    sampling: SamplingConfig
    plot_loss_curves: bool
    model: Any
    dataset: DatasetConfig
    logger: LoggerConfig
    checkpointer: CheckpointerConfig


# Register the intended structure and types for these different configs
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
cs.store(group="model", name="mlp_schema", node=ResidualMLPConfig)
cs.store(group="model", name="ar_transformer_schema", node=TransformerConfig)
cs.store(
    group="model",
    name="interpolant_transformer_schema",
    node=InterpolantTransformerConfig,
)
cs.store(group="model", name="itarflow_schema", node=iTarFlowConfig)
cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)


def validate_config(cfg: TrainConfig):
    """Fail fast on config errors, before any worker is spawned or data is loaded."""
    if not 1 <= cfg.num_gpus <= torch.cuda.device_count():
        raise ValueError(
            f"num_gpus={cfg.num_gpus} but {torch.cuda.device_count()} GPUs are visible"
        )
    if cfg.batch_size <= 0 or cfg.batch_size % cfg.num_gpus != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be positive and divisible by "
            f"num_gpus ({cfg.num_gpus})"
        )
    save_every = cfg.checkpointer.save_every
    if save_every != "never" and (
        isinstance(save_every, bool)
        or not isinstance(save_every, int)
        or save_every <= 0
    ):
        raise ValueError(
            "checkpointer.save_every must be 'never' or a positive integer"
        )


# TODO: Implement different schedules
def get_lr(step, learning_rate, min_lr, max_steps, warmup_steps):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def generate_comparison_plot(
    cfg: TrainConfig,
    model: ResidualMLP | Transformer | InterpolantTransformer | ImprovedTARFlow,
    df: pl.DataFrame,
    candidates: pl.DataFrame,
    data_inputs_means_and_stds: DatasetMeansAndStds,
    data_targets_means_and_stds: DatasetMeansAndStds,
    subsample: int | None = 10,
    keep_greater_than_zero: bool = False,
    num_steps: int | None = None,
    solver: str | None = None,
    multi_mode: str | None = None,
    seed: int = 0,
    device: str = "cuda",
):
    if subsample is not None:
        candidates = candidates.sample(n=subsample, seed=seed)
    selected_df = df.filter(pl.col("param_index").is_in(candidates["param_index"]))

    # TODO: Ensure equal number of reps
    selected_df = selected_df.sort(["param_index", "rep", "t"])

    T = cfg.dataset.num_timesteps
    n_cols_in = len(cfg.dataset.surrogate_input_variables)
    n_cols_out = len(cfg.dataset.output_variables)
    obs_vars = cfg.dataset.observed_output_variables or []
    if obs_vars:
        all_out = cfg.dataset.output_variables
        pred_vars = [v for v in all_out if v not in obs_vars]
        obs_idx = [all_out.index(v) for v in obs_vars]
        pred_idx = [all_out.index(v) for v in pred_vars]
        n_pred = len(pred_vars)
        plot_vars = pred_vars
        n_plot = n_pred
    else:
        obs_idx = None
        pred_idx = None
        n_pred = n_cols_out
        plot_vars = list(cfg.dataset.output_variables)
        n_plot = n_cols_out
    n_reps = candidates["rep_count"].item(0)
    n_param_grps = len(selected_df) // (T * n_reps)

    inputs_np = (
        selected_df.select(cfg.dataset.surrogate_input_variables)
        .cast(pl.Float32)
        .to_numpy()
    )
    outputs_np = (
        selected_df.select(cfg.dataset.output_variables).cast(pl.Float32).to_numpy()
    )

    inputs_tensor = (
        torch.from_numpy(inputs_np)
        .view(n_param_grps, n_reps, T, n_cols_in)
        .permute(0, 1, 3, 2)
    )  # (param_groups, reps, input_cols, T)
    outputs_tensor = (
        torch.from_numpy(outputs_np)
        .view(n_param_grps, n_reps, T, n_cols_out)
        .permute(0, 1, 3, 2)
    )  # (param_groups, reps, output_cols, T)

    # Calculate the median of inputs, since within a param group certain columns may still vary
    median_inputs_tensor = inputs_tensor.quantile(q=0.5, dim=1, keepdim=True)

    # Z-score
    surrogate_input = (
        median_inputs_tensor.to(device) - data_inputs_means_and_stds.means.to(device)
    ) / data_inputs_means_and_stds.stds.to(device)

    assert surrogate_input.size(2) == len(
        cfg.dataset.surrogate_input_variables
    ) and outputs_tensor.size(2) == len(cfg.dataset.output_variables)

    surrogate_input = surrogate_input[..., 0]
    # Vmap since our surrogate_input has an extra n_reps (=1) dim
    # surrogate_input, _ = torch.vmap(
    #     ensure_consistent_data_dim, in_dims=(1, None), out_dims=(1, None)
    # )(surrogate_input, outputs_tensor[:, 0, ...])

    multiple_realisations = False
    if "residual_mlp" in cfg.model._target_:
        surrogate_input = surrogate_input[:, 0, :n_cols_in]
        generated = model(surrogate_input).reshape(-1, T, n_cols_out)

        generated = (
            (
                generated
                * data_targets_means_and_stds.stds.transpose(1, 0)[None, ...].to(device)
                + data_targets_means_and_stds.means.transpose(1, 0)[None, ...].to(
                    device
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
    elif "autoregressive_transformer" in cfg.model._target_:
        assert isinstance(model, Transformer)
        # Build conditioning sequence for cross-attention from observed columns
        # TODO: Investigate alternatives to using median — e.g. per-rep conditioning
        # For now we take the median observed sequence across reps as a single
        # conditioning input, z-score it, and feed it to generate.
        cond_seq = None
        if obs_idx is not None:
            # outputs_tensor: (param_groups, reps, output_cols, T)
            # Collapse reps to median → (param_groups, output_cols, T)
            median_outputs = outputs_tensor.quantile(q=0.5, dim=1)
            # Extract only the observed columns and z-score them
            obs_raw = median_outputs[:, obs_idx, :]  # (param_groups, n_obs, T)
            obs_means = data_targets_means_and_stds.means[obs_idx, :]
            obs_stds = data_targets_means_and_stds.stds[obs_idx, :]
            # Permute to (param_groups, T, n_obs) to match model's expected cond shape
            cond_seq = (
                (obs_raw.to(device) - obs_means.to(device)) / obs_stds.to(device)
            ).permute(0, 2, 1)
        stream = model.generate(
            surrogate_input[
                :, 0, :
            ],  # (param_groups,1,input_cols)->(param_groups,input_cols)
            None,
            T,
            use_kv_cache=cfg.sampling.use_kv_cache,
            conditioning_sequence=cond_seq,
        )
        # Model is deterministic, one output for each param group
        generated = torch.empty((n_param_grps, 0, n_pred), device=device)
        for pred in stream:
            generated = torch.cat((generated, pred), dim=1)
        # Un-z-score using only the predicted columns' stats
        if pred_idx is not None:
            pred_means = data_targets_means_and_stds.means[pred_idx, :]
            pred_stds = data_targets_means_and_stds.stds[pred_idx, :]
        else:
            pred_means = data_targets_means_and_stds.means
            pred_stds = data_targets_means_and_stds.stds

        generated = (
            (
                generated * pred_stds.transpose(1, 0)[None, ...].to(device)
                + pred_means.transpose(1, 0)[None, ...].to(device)
            )
            .detach()
            .cpu()
            .numpy()
        )
    elif "interpolant_transformer" in cfg.model._target_:
        assert isinstance(model, InterpolantTransformer)
        # num_steps is always supplied for interpolant models (from cfg.model.num_steps)
        assert num_steps is not None
        assert solver is not None
        multiple_realisations = True
        cond_seq = None
        generated = []
        if multi_mode == "median":
            if obs_idx is not None:
                # outputs_tensor: (param_groups, reps, output_cols, T)
                # Collapse reps to median → (param_groups, output_cols, T)
                median_outputs = outputs_tensor.quantile(q=0.5, dim=1)
                # Extract only the observed columns and z-score them
                obs_raw = median_outputs[:, obs_idx, :]  # (param_groups, n_obs, T)
                obs_means = data_targets_means_and_stds.means[obs_idx, :]
                obs_stds = data_targets_means_and_stds.stds[obs_idx, :]
                # Permute to (param_groups, T, n_obs) to match model's expected cond shape
                cond_seq = (
                    (obs_raw.to(device) - obs_means.to(device)) / obs_stds.to(device)
                ).permute(0, 2, 1)
                # print(cond_seq.shape)
                # print(outputs_tensor.shape)

            if pred_idx is not None:
                pred_means = data_targets_means_and_stds.means[pred_idx, :]
                pred_stds = data_targets_means_and_stds.stds[pred_idx, :]
            else:
                pred_means = data_targets_means_and_stds.means
                pred_stds = data_targets_means_and_stds.stds

            assert cfg.dataset.reps_for_comparison_plot is not None
            for i in range(
                cfg.dataset.reps_for_comparison_plot
            ):  # match number of reps to param group TODO: make configurable
                single_realisation_generated, _x0 = model.sample(
                    torch.randn(
                        (
                            outputs_tensor.shape[0],  # N param groups
                            outputs_tensor.shape[3],  # T
                            n_pred,  # predicted output cols
                        ),
                        device=model.get_device(),
                    ),
                    surrogate_input[:, 0, :],
                    num_steps=num_steps,
                    method=solver,
                    cond_seq=cond_seq,
                )
                single_realisation_generated = (
                    (
                        single_realisation_generated
                        * pred_stds.transpose(1, 0)[None, ...].to(device)
                        + pred_means.transpose(1, 0)[None, ...].to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                generated.append(single_realisation_generated)
            generated = np.stack(
                generated, axis=1
            )  # (param_groups, 10 ,T, output_cols)
        elif (
            multi_mode == "entire_param_group"
        ):  # HRP and Magenta are bizzare in that realisations within the same param group can have different static inputs
            # NOTE: Slow, maybe do some kind of batching
            if obs_idx is not None:
                raise NotImplementedError(
                    "Cross attn support not yet implemented for entire param group sampling"
                )
            generated = []
            for param_group_inputs in inputs_tensor:
                param_group_inputs = (
                    param_group_inputs.to(device)
                    - data_inputs_means_and_stds.means.to(device)
                ) / data_inputs_means_and_stds.stds.to(device)

                param_group_inputs = param_group_inputs[
                    ..., 0
                ]  # We only need the first timestep
                # param_group_inputs=param_group_inputs[:,:,:n_cols_in]
                # surrogate_input = surrogate_input[:, 0, :n_cols_in]

                param_group_generated, _x0 = model.sample(
                    torch.randn(
                        (
                            outputs_tensor.shape[1],  # num reps
                            outputs_tensor.shape[3],  # T
                            outputs_tensor.shape[2],  # input_cols
                        ),
                        device=model.get_device(),
                    ),
                    param_group_inputs,
                    num_steps=num_steps,
                    method=solver,
                )
                param_group_generated = (
                    (
                        param_group_generated
                        * data_targets_means_and_stds.stds.transpose(1, 0)[
                            None, ...
                        ].to(device)
                        + data_targets_means_and_stds.means.transpose(1, 0)[
                            None, ...
                        ].to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                generated.append(param_group_generated)
            generated = np.stack(generated)  # (param_groups, n_reps ,T, output_cols)
        else:
            raise ValueError(
                "Unrecognised multi_mode for stochastic interpolant transformer"
            )
    elif "itarflow" in cfg.model._target_:
        assert isinstance(model, ImprovedTARFlow)
        generated, _ = model.reverse(
            torch.randn(
                (
                    outputs_tensor.shape[0],  # N param groups
                    outputs_tensor.shape[3],  # T
                    n_pred,  # predicted output cols
                ),
                device=model.get_device(),
            ),
            surrogate_input[:, 0, :],
        )
        generated = (
            (
                generated
                * data_targets_means_and_stds.stds.transpose(1, 0)[None, ...].to(device)
                + data_targets_means_and_stds.means.transpose(1, 0)[None, ...].to(
                    device
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(f"{cfg.model._target_} not a recognised model")

    fig, axarr = plt.subplots(
        generated.shape[0],
        n_plot,
        figsize=(n_plot * 8, 5 * generated.shape[0]),
    )
    if keep_greater_than_zero:
        generated = np.maximum(0, generated)

    for i, param_index in zip(range(generated.shape[0]), candidates["param_index"]):
        shrtn = {"rdt.nonadherence": "rdt.nonad", "microscopy.use": "micro.use"}
        for j, output_col in zip(range(n_plot), plot_vars):
            # j indexes into generated (predicted-only cols), col_idx indexes into
            # outputs_tensor (all cols) so we plot the right ground-truth column
            col_idx = list(cfg.dataset.output_variables).index(output_col)
            if j % 2 == 0:
                axarr[i, j].set_title(
                    " ".join(
                        [
                            f"{k if k not in ['rdt.nonadherence', 'microscopy.use'] else shrtn[k]}: {v:.2f}"
                            for k, v in zip(
                                cfg.dataset.surrogate_input_variables,
                                median_inputs_tensor[
                                    i, 0, :, 0
                                ],  # Get the un-zscored median input
                            )
                        ]
                    )
                )

            # Plot all reps in the param group
            axarr[i, j].plot(
                outputs_tensor[i, :, col_idx, ...].T, color="orange", linestyle="--"
            )
            axarr[i, j].plot(
                torch.mean(outputs_tensor[i, :, col_idx, ...], dim=0),
                color="red",
                linestyle="-",
            )
            # Plot generated realisation(s)
            if multiple_realisations:
                # (param_groups, n_reps ,T, output_cols)
                axarr[i, j].plot(
                    generated[i, :, :, j].T,
                    color="deepskyblue",
                    linestyle="dotted",
                    alpha=0.5,
                )
                mean_realisation = np.mean(generated[i, :, :, j], axis=0)
                axarr[i, j].plot(mean_realisation, linewidth=1.5)
            else:
                axarr[i, j].plot(generated[i, :, j], linewidth=1.5)
            axarr[i, j].set_ylabel(output_col)
    fig.tight_layout()
    assert cfg.sampling.samples_name is not None
    fig.savefig(cfg.sampling.samples_name)


RENDEZVOUS_FILENAME = ".rendezvous"  # Workers meet through this file in the output dir


def run_training(
    cfg: TrainConfig, runtime: DistributedRuntime, precision_mode, output_dir: str
) -> float:
    if runtime.distributed:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{os.path.join(output_dir, RENDEZVOUS_FILENAME)}",
            rank=runtime.rank,
            world_size=runtime.world_size,
            device_id=runtime.device,
        )
    if runtime.is_main:
        print(OmegaConf.to_yaml(cfg))

    if runtime.is_main and cfg.logger.use_wandb:
        wandb.init(
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),  # ty: ignore[invalid-argument-type]
            tags=cfg.logger.tags,
        )
        # Full unprocessed cfg uploaded to wandb
        yaml_str = OmegaConf.to_yaml(cfg)
        with tempfile.NamedTemporaryFile(
            mode="w+t",
            prefix="train_config",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(yaml_str)
            temp_file.flush()
            wandb.save(
                temp_file_path, base_path=os.path.dirname(temp_file_path), policy="now"
            )

    # Allow overriding
    OmegaConf.set_struct(cfg, False)  # ty: ignore[invalid-argument-type]
    obs_vars = cfg.dataset.observed_output_variables or []

    if cfg.dataset.exclude_output_variables:
        excluded = set(cfg.dataset.exclude_output_variables)

        obs_and_excl_intersection = excluded.intersection(set(obs_vars))
        assert len(obs_and_excl_intersection) == 0, (
            f"Observed and excluded variables intersect: {obs_and_excl_intersection}"
        )

        cfg.dataset.output_variables = [
            v for v in cfg.dataset.output_variables if v not in excluded
        ]
    if "residual_mlp" in cfg.model._target_:
        cfg.model.input_dim = len(cfg.dataset.surrogate_input_variables)
        cfg.model.output_dim = cfg.dataset.num_timesteps * len(
            cfg.dataset.output_variables
        )
        pred_and_obs_idx = None
    elif "autoregressive_transformer" in cfg.model._target_:
        # +1 since we are prepending the static input
        cfg.model.sequence_len = cfg.dataset.num_timesteps + 1
        all_out = cfg.dataset.output_variables

        if obs_vars:
            assert cfg.model.cross_attn is not None, (
                "Attempting to train on observed sequences without a cross-attention module"
            )
            pred_vars = [v for v in all_out if v not in obs_vars]
            obs_idx = [all_out.index(v) for v in obs_vars]
            pred_idx = [all_out.index(v) for v in pred_vars]
            pred_and_obs_idx = (pred_idx, obs_idx)

            # Our x-input dim is the number of variables we're trying to predict
            cfg.model.x_input_dim = len(pred_idx)
            # Cross-attn input dim is the number of observed sequence variables
            cfg.model.cross_attn.cond_dim = len(obs_idx)
            # Static input dim is the same as before
            cfg.model.static_input_dim = len(cfg.dataset.surrogate_input_variables)

        else:
            cfg.model.x_input_dim = len(all_out)
            cfg.model.static_input_dim = len(cfg.dataset.surrogate_input_variables)
            pred_and_obs_idx = None

    elif "interpolant_transformer" in cfg.model._target_:
        all_out = cfg.dataset.output_variables
        cfg.model.sit.sequence_len = cfg.dataset.num_timesteps
        cfg.model.sit.cond_dim = len(cfg.dataset.surrogate_input_variables)

        if obs_vars:
            assert cfg.model.sit.cross_attn is not None, (
                "Attempting to train on observed sequences without a cross-attention module"
            )
            pred_vars = [v for v in all_out if v not in obs_vars]
            obs_idx = [all_out.index(v) for v in obs_vars]
            pred_idx = [all_out.index(v) for v in pred_vars]
            pred_and_obs_idx = (pred_idx, obs_idx)

            # Cross-attn input dim is the number of observed sequence variables
            cfg.model.sit.cross_attn.cond_dim = len(obs_idx)
            # Our x data dim is the number of variables we're trying to predict
            cfg.model.sit.data_dim = len(pred_idx)
        else:
            cfg.model.sit.data_dim = len(cfg.dataset.output_variables)
            pred_and_obs_idx = None
    elif "itarflow" in cfg.model._target_:
        # NOTE: obs_idx (sequence cond) unsupported for now
        if obs_vars:
            raise NotImplementedError(
                "Itarflow model does not support sequence conditioning"
            )
        cfg.model.out_channels = len(cfg.dataset.output_variables)
        cfg.model.num_timesteps = cfg.dataset.num_timesteps
        cfg.model.static_in = len(cfg.dataset.surrogate_input_variables)
        pred_and_obs_idx = None

    else:
        raise ValueError(f"Cannot identify model with target {cfg.model._target_}")
    OmegaConf.set_struct(cfg, True)  # ty: ignore[invalid-argument-type]
    model: ResidualMLP | Transformer | InterpolantTransformer | ImprovedTARFlow = (
        hydra.utils.instantiate(cfg.model)
    )
    model.to(runtime.device)

    # NOTE: val split and split seed hardcoded for now
    train_df, val_df, input_means_and_stds, target_means_and_stds = prepare_splits(
        cfg.dataset.path,
        0.1,
        42,
        cfg.dataset.grouping_variables,
        cfg.dataset.surrogate_input_variables,
        cfg.dataset.output_variables,
        cfg.dataset.num_timesteps,
        cfg.dataset.trunc_until,
        cfg.dataset.input_col_to_source_col,
    )
    train_static_input_data, train_x_data, _, _ = make_tensors(
        train_df,
        cfg.dataset.surrogate_input_variables,
        cfg.dataset.output_variables,
        cfg.dataset.num_timesteps,
        model.get_device(),
        z_score=True,
        means_and_stds=(input_means_and_stds, target_means_and_stds),
    )
    val_static_input_data, val_x_data, _, _ = make_tensors(
        val_df,
        cfg.dataset.surrogate_input_variables,
        cfg.dataset.output_variables,
        cfg.dataset.num_timesteps,
        z_score=True,
        means_and_stds=(input_means_and_stds, target_means_and_stds),
    )
    # Each rank validates its own shard of the validation set; shard on the CPU so
    # that only the shard is copied to the GPU
    val_static_input_data = val_static_input_data[
        runtime.rank :: runtime.world_size
    ].to(runtime.device)
    val_x_data = val_x_data[runtime.rank :: runtime.world_size].to(runtime.device)
    save_every = cfg.checkpointer.save_every
    checkpointing = save_every != "never"
    if checkpointing:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if runtime.is_main:
            os.makedirs(ckpt_dir, exist_ok=True)

    # batch_size is the global batch size, split evenly across GPUs
    local_batch_size = cfg.batch_size // runtime.world_size
    gen = torch.Generator(model.get_device())  # Reseeded every step, see step_seed
    # NOTE: Weight decay is currently hardcoded
    optimiser = model.setup_optimiser(0.001, cfg.peak_lr)
    # TODO: Print num. params

    start_step = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_steps: list[int] = []
    if cfg.checkpointer.resume_from is not None:
        ckpt = torch.load(
            cfg.checkpointer.resume_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        train_losses = ckpt["train_losses"].tolist()
        val_losses = ckpt["val_losses"]
        val_steps = ckpt["val_steps"]
        start_step = ckpt["step"] + 1
        # Resuming reproduces an uninterrupted run only if the sampling and the
        # learning-rate schedule are unchanged
        changed = [
            key
            for key in (
                "seed",
                "num_gpus",
                "batch_size",
                "max_steps",
                "peak_lr",
                "lr_warmup_frac",
            )
            if ckpt["config"][key] != getattr(cfg, key)
        ]
        if runtime.is_main:
            print(f"\nResumed from {cfg.checkpointer.resume_from}")
            if changed:
                print(f"Note: {changed} differ from the checkpoint's config")

    # model: optimiser, checkpoints and sample generation
    # loss_module: evaluation (compiled)
    # train_loss_module: training forward and backward (DDP-wrapped)
    loss_module: nn.Module = TrainingLoss(model, pred_and_obs_idx)
    if not cfg.deterministic:
        # Compile before wrapping in DDP: dynamo cannot trace DDP.forward with
        # fullgraph=True, and compiling the inner module is what lets DDP overlap
        # the gradient all-reduce with the backward pass. Training and evaluation
        # calls differ in their guards (train/eval mode, grad, mask), so they get
        # separate graphs on this one object
        loss_module = torch.compile(  # ty: ignore[invalid-assignment]
            loss_module,
            mode="reduce-overhead",  # Stricter than default torch but can reduce python overhead that dominates when batches are small
            fullgraph=True,  # crash if there is a graph break
        )
    elif runtime.is_main:
        print("Deterministic mode: torch.compile disabled")
    train_loss_module = loss_module
    if runtime.distributed:
        train_loss_module = DistributedDataParallel(
            loss_module,
            device_ids=[runtime.device],
            broadcast_buffers=False,  # The only buffers are constant RoPE tables
        )
    lr_warmup = int(cfg.lr_warmup_frac * cfg.max_steps)

    model.train()
    t0 = time.monotonic()
    max_steps = cfg.max_steps
    final_val_loss = float("inf")
    pbar = tqdm(
        range(start_step, max_steps),
        total=max_steps,
        initial=start_step,
        desc="step  | loss  | lr  | dt ",
        disable=not runtime.is_main,
    )
    for step in pbar:
        # NOTE: Minimum lr currently hardcoded to 1e-6
        lr = get_lr(step, cfg.peak_lr, 1e-6, max_steps, lr_warmup)
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr

        # Reseed per step so the batch schedule (and, in deterministic mode, the
        # model's noise) is a pure function of the step and resuming is exact
        gen.manual_seed(step_seed(cfg.seed, step, runtime, stream=0))
        if cfg.deterministic:
            torch.cuda.manual_seed(step_seed(cfg.seed, step, runtime, stream=1))
        static_input, x_input, targets = get_random_batch(
            gen, train_static_input_data, train_x_data, local_batch_size
        )
        optimiser.zero_grad()
        with training_autocast(precision_mode, runtime.device.type):
            loss = train_loss_module(static_input, x_input, targets)

        loss.backward()
        optimiser.step()

        # Clone: compiled outputs live in cudagraph-managed memory, never mutate them in place
        global_loss = loss.detach().clone()
        if runtime.distributed:
            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)

        # Logging
        if runtime.is_main:
            global_loss_value = global_loss.item()
            train_losses.append(global_loss_value)
            torch.cuda.synchronize(runtime.device)
            t1 = time.monotonic()
            dt = (t1 - t0) * 1000  # ms per step
            t0 = t1
            if step % 10 == 0 or step == max_steps - 1:
                pbar.set_description(
                    f"step {step:3d} | loss {global_loss_value:.6f} | "
                    f"lr {lr:.4e} | dt {dt:.2f}ms"
                )
            if cfg.logger.use_wandb:
                wandb.log(
                    {
                        "train/loss": global_loss_value,
                        "lr": lr,
                        "train/dt": dt,
                        "step": step,
                    }
                )
        should_evaluate = (
            step > 0 and step % cfg.eval_every == 0
        ) or step == max_steps - 1
        save_best = False
        if should_evaluate:
            model.eval()
            val_t0 = time.monotonic()
            val_loss = evaluate(
                loss_module,
                val_static_input_data,
                val_x_data,
                local_batch_size,
                runtime,
            )
            model.train()
            save_best = val_loss < (min(val_losses) if val_losses else float("inf"))
            val_losses.append(val_loss)
            val_steps.append(step)
            val_dt = (time.monotonic() - val_t0) * 1000
            if runtime.is_main:
                print(
                    f"\nValidation step {step:3d} | loss {val_loss:.6f} | "
                    f"dt {val_dt:.2f}ms\n"
                )
                if cfg.logger.use_wandb:
                    wandb.log({"val/loss": val_loss, "val/dt": val_dt, "step": step})
                t0 = time.monotonic()
            if step == max_steps - 1:
                final_val_loss = val_loss

        if checkpointing and runtime.is_main:
            checkpoint_paths = []
            if save_best:
                checkpoint_paths.append(os.path.join(ckpt_dir, "best.pt"))
            if step % save_every == 0 or step == max_steps - 1:
                checkpoint_paths.append(
                    os.path.join(
                        ckpt_dir,
                        "last.pt" if step == max_steps - 1 else f"ckpt_{step}.pt",
                    )
                )
            for save_path in checkpoint_paths:
                save_checkpoint(
                    save_path,
                    cfg,
                    step,
                    model,
                    optimiser,
                    train_losses,
                    val_losses,
                    val_steps,
                )
                print(f"Saved checkpoint to {save_path}")
    if runtime.distributed:
        # All ranks leave the process group together here (NCCL communicator shutdown
        # needs every rank to take part within the timeout); only rank 0 has work
        # left. Deliberately no try/finally: if a worker fails, exiting lets mp.spawn
        # terminate the others at once, whereas a destroy from the failing worker
        # would wait on peers that are still inside a collective
        dist.destroy_process_group()

    if not runtime.is_main:
        return final_val_loss

    if cfg.plot_loss_curves:
        _fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
        ax.plot(val_steps, val_losses, label="Validation Loss")
        plt.legend()
        plt.show()

    if cfg.sampling.generate_samples:
        assert cfg.sampling.samples_name is not None, (
            "samples_name must be specified if generate_samples is True."
        )
        # Identify indices with exact unique reps
        candidates = (
            val_df.group_by("param_index")
            .agg(pl.col("rep").n_unique().alias("rep_count"))
            .filter(
                pl.col("rep_count") == cfg.dataset.reps_for_comparison_plot
                # pl.col("rep_count") == 5  # TODO: This should be in the dataset conf
                # if "model_sims" in cfg.dataset.path
                # else pl.col("rep_count") == 10
            )
            # group_by returns hash-order (non-deterministic), so sort for reproducible sampling
            .sort("param_index")
        )
        generate_comparison_plot(
            cfg,
            model,
            val_df,
            candidates,
            input_means_and_stds,
            target_means_and_stds,
            subsample=cfg.sampling.subsample,
            keep_greater_than_zero=cfg.sampling.keep_greater_than_zero,
            num_steps=cfg.model.num_steps
            if "interpolant_transformer" in cfg.model._target_
            else None,
            solver=cfg.model.solver
            if "interpolant_transformer" in cfg.model._target_
            else None,
            multi_mode=cfg.model.multi_mode
            if "interpolant_transformer" in cfg.model._target_
            else None,
            device=str(runtime.device),
        )
    if cfg.logger.use_wandb:
        assert wandb.run is not None
        wandb.run.summary["final_val_loss"] = final_val_loss
        wandb.finish(quiet=True)
    return final_val_loss


def train_worker(
    rank: int, world_size: int, cfg: TrainConfig, output_dir: str, result_queue=None
) -> float:
    """Set up this process's GPU and train; with num_gpus > 1 there is one per GPU."""
    runtime = DistributedRuntime(rank, world_size, torch.device("cuda", rank))
    torch.cuda.set_device(runtime.device)
    # Rank 0 keeps cfg.seed so its model init matches a single-GPU run; DDP then
    # broadcasts rank 0's weights to the other ranks
    seed = cfg.seed + rank
    if cfg.deterministic:
        make_deterministic(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    precision_mode = PrecisionMode(cfg.precision)
    configure_precision(precision_mode)
    final_val_loss = run_training(cfg, runtime, precision_mode, output_dir)
    if result_queue is not None and runtime.is_main:
        result_queue.put(final_val_loss)
    return final_val_loss


@hydra.main(config_path="configs/train", config_name=None, version_base=None)
def main(cfg: TrainConfig) -> float:
    validate_config(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.num_gpus == 1:
        return train_worker(0, 1, cfg, output_dir)
    # One process per GPU inside this single Hydra run, so sweeps (--multirun,
    # wandb agents) work exactly as they do on one GPU
    rendezvous_file = os.path.join(output_dir, RENDEZVOUS_FILENAME)
    if os.path.exists(rendezvous_file):
        os.remove(
            rendezvous_file
        )  # A stale one from an earlier run would deadlock the workers
    result_queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        train_worker,
        args=(cfg.num_gpus, cfg, output_dir, result_queue),
        nprocs=cfg.num_gpus,
    )
    return result_queue.get()


if __name__ == "__main__":
    main()
