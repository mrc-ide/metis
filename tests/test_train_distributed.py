import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from models.mlp import ResidualMLP, ResidualMLPConfig
from train import (
    DistributedRuntime,
    TrainingLoss,
    evaluate,
    save_checkpoint,
    step_seed,
)

SINGLE_PROCESS = DistributedRuntime(0, 1, torch.device("cpu"))


def test_step_seed_is_unique_per_stream_step_and_rank():
    runtimes = [DistributedRuntime(rank, 2, torch.device("cpu")) for rank in range(2)]
    seeds = [
        step_seed(42, step, runtime, stream)
        for step in range(3)
        for runtime in runtimes
        for stream in (0, 1)
    ]
    assert len(set(seeds)) == len(seeds)


def test_training_loss_delegates_to_model(mlp_model, tiny_mlp_config):
    wrapper = TrainingLoss(mlp_model, pred_and_obs_idx=None)
    static_input = torch.randn(2, tiny_mlp_config.input_dim)
    targets = torch.randn(2, 2, tiny_mlp_config.output_dim // 2)

    loss = wrapper(static_input, torch.empty(2, 0), targets)
    loss.backward()

    assert loss.ndim == 0
    assert mlp_model.input_layer.weight.grad is not None


def test_evaluate_weights_padded_final_batch(mlp_model, tiny_mlp_config):
    loss_module = TrainingLoss(mlp_model, pred_and_obs_idx=None)
    static_input = torch.randn(3, tiny_mlp_config.input_dim)
    x_data = torch.randn(3, 2, tiny_mlp_config.output_dim // 2)  # (N, channels, T)

    with torch.no_grad():
        # batch_size 2: the second batch is one real sample plus one padding sample
        val_loss = evaluate(loss_module, static_input, x_data, 2, SINGLE_PROCESS)
        targets = x_data.permute(0, 2, 1)
        expected = loss_module(static_input, targets[:, :-1], targets).item()

    assert val_loss == pytest.approx(expected)
    with pytest.raises(ValueError, match="empty"):
        evaluate(loss_module, static_input[:0], x_data[:0], 2, SINGLE_PROCESS)


def test_checkpoint_round_trips_and_leaves_no_temporary_files(tmp_path, mlp_model):
    optimiser = torch.optim.AdamW(mlp_model.parameters())
    cfg = OmegaConf.create({"seed": 1})
    path = tmp_path / "checkpoint.pt"

    for step in (3, 4):  # The second save overwrites the first
        save_checkpoint(
            path, cfg, step, mlp_model, optimiser, [1.0, 0.5], [0.75], [step]
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["step"] == 4
    assert checkpoint["config"] == {"seed": 1}
    assert checkpoint["val_steps"] == [4]
    torch.testing.assert_close(checkpoint["train_losses"], torch.tensor([1.0, 0.5]))
    mlp_model.load_state_dict(checkpoint["model"])
    optimiser.load_state_dict(checkpoint["optimiser"])
    assert list(tmp_path.iterdir()) == [path]


def _run_ddp_test(rank, rendezvous_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(5)
        config = ResidualMLPConfig(
            input_dim=2,
            output_dim=2,
            num_blocks=1,
            hidden_dim=4,
            num_layers_per_block=2,
        )
        model = ResidualMLP(config)
        reference = copy.deepcopy(model)
        # Compile before wrapping in DDP, as train.py does: the reverse order fails
        # with fullgraph=True because dynamo cannot trace DDP.forward
        distributed_model = DistributedDataParallel(
            torch.compile(TrainingLoss(model, None), backend="eager", fullgraph=True)
        )
        optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
        reference_optimiser = torch.optim.SGD(reference.parameters(), lr=0.05)

        all_static = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
        all_targets = torch.tensor(
            [[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 2.0]], [[2.0, 1.0]]]
        )
        local_slice = slice(rank * 2, (rank + 1) * 2)

        # Two DDP steps on half batches must match two steps on the full batch
        for _ in range(2):
            optimiser.zero_grad()
            loss = distributed_model(
                all_static[local_slice], torch.empty(2, 0), all_targets[local_slice]
            )
            loss.backward()
            optimiser.step()

            reference_optimiser.zero_grad()
            reference_loss = TrainingLoss(reference, None)(
                all_static, torch.empty(4, 0), all_targets
            )
            reference_loss.backward()
            reference_optimiser.step()

        for parameter, reference_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(parameter, reference_parameter)

        # Uneven validation shards (3 samples on rank 0, 1 on rank 1, batch size 2)
        # must still give the mean over all four samples
        loss_module = TrainingLoss(reference, None)
        runtime = DistributedRuntime(rank, 2, torch.device("cpu"))
        shard = slice(0, 3) if rank == 0 else slice(3, 4)
        all_x_data = all_targets.permute(0, 2, 1)  # (N, channels, T)
        with torch.no_grad():
            val_loss = evaluate(
                loss_module, all_static[shard], all_x_data[shard], 2, runtime
            )
            expected = loss_module(all_static, torch.empty(4, 0), all_targets).item()
        assert val_loss == pytest.approx(expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_ddp_matches_single_process(tmp_path):
    rendezvous_path = tmp_path / "ddp_rendezvous"
    mp.spawn(_run_ddp_test, args=(str(rendezvous_path),), nprocs=2, join=True)
