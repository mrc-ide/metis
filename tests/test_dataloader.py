import polars as pl
import pytest
import torch

from dataloader import (
    clean_df,
    get_data_iterator,
    get_random_batch,
    left_truncate_and_recalculate_static_inputs,
    make_tensors,
    prepare_splits,
)


@pytest.mark.parametrize("config_name", ["sirds"])
def test_dataset_configs_load(config_name):
    """Very basic test to check that we can parse all the configs"""
    import hydra

    with hydra.initialize(version_base=None, config_path="../configs/dataset"):
        conf = hydra.compose(config_name=config_name)

    assert isinstance(conf.num_timesteps, int)
    assert conf.num_timesteps > 0
    assert len(conf.surrogate_input_variables) > 0
    assert len(conf.output_variables) > 0
    assert hasattr(conf, "grouping_variables")


@pytest.fixture
def dummy_data_tensor():
    return torch.zeros((50, 8, 10))


def test_data_iterator_no_padding(dummy_data_tensor):
    iterator = get_data_iterator(torch.zeros((50, 6)), dummy_data_tensor, 50)
    unpadded_static, unpadded_x_inputs, unpadded_x_targets, mask = next(iterator)

    assert torch.equal(mask, torch.ones((50,))), "Incorrect mask for unpadded output"
    assert torch.equal(unpadded_static, torch.zeros((50, 6))), (
        "Unpadded static is incorrect"
    )
    assert torch.equal(unpadded_x_inputs, torch.zeros((50, 9, 8))), (
        "Unpadded inputs are incorrect."
    )
    assert torch.equal(unpadded_x_targets, torch.zeros((50, 10, 8))), (
        "Unpadded targets are incorrect."
    )


def test_data_iterator_padding(dummy_data_tensor):
    # Batch size is bigger than tensor so will pad to ensure batch size maintained
    iterator = get_data_iterator(torch.zeros((50, 6)), dummy_data_tensor, 100)
    padded_static, padded_x_inputs, padded_x_targets, mask = next(iterator)

    assert torch.equal(mask, torch.cat([torch.ones((50,)), torch.zeros((50,))])), (
        "Incorrect mask for padded output"
    )
    assert torch.equal(
        padded_static, torch.cat([torch.zeros(50, 6), -1 * torch.ones(50, 6)])
    ), "Static inputs not padded correctly"

    assert torch.equal(
        padded_x_inputs,
        torch.cat([torch.zeros((50, 9, 8)), -1 * torch.ones(50, 9, 8)]),
    ), "x-inputs not padded correctly"
    assert torch.equal(
        padded_x_targets,
        torch.cat([torch.zeros((50, 10, 8)), -1 * torch.ones(50, 10, 8)]),
    ), "Targets not padded correctly"


def test_clean_df_removes_duplicates_without_interleaving():
    """clean_df should keep exactly one complete timeseries per (group, rep),
    with no interleaving of rows from different source timeseries.

    The key trick: duplicate timeseries have values whose sort order *crosses*
    between timesteps. A naive sort-all-columns + unique approach would pick
    from series A at one timestep and series B at the next.
    """
    grouping_cols = ["a"]

    # Series A and B for rep=1 have values that cross:
    #   A: t=1 → value=1.0,  t=2 → value=5.0
    #   B: t=1 → value=2.0,  t=2 → value=3.0
    # Sorting by value at t=1 picks A, but at t=2 picks B → interleaving.
    # Contiguous blocks (as in real parquet data): A first, then B.
    df = pl.DataFrame(
        [
            # Series A (first contiguous block)
            {"a": 0.1, "rep": 1, "t": 1.0, "value": 1.0},
            {"a": 0.1, "rep": 1, "t": 2.0, "value": 5.0},
            # Series B (second contiguous block)
            {"a": 0.1, "rep": 1, "t": 1.0, "value": 2.0},
            {"a": 0.1, "rep": 1, "t": 2.0, "value": 3.0},
            # Rep 2 — no duplicates
            {"a": 0.1, "rep": 2, "t": 1.0, "value": 10.0},
            {"a": 0.1, "rep": 2, "t": 2.0, "value": 20.0},
        ]
    )

    cleaned = clean_df(df, grouping_cols)
    cleaned_sorted = cleaned.sort(grouping_cols + ["rep", "t"])

    # 1. Correct row count: rep=1 deduped from 4→2, rep=2 unchanged at 2
    assert cleaned_sorted.height == 4, (
        f"Expected 4 rows after dedup, got {cleaned_sorted.height}"
    )

    # 2. No interleaving: rep=1's values must all come from the same series
    rep1 = cleaned_sorted.filter(pl.col("rep") == 1).sort("t")
    vals = rep1["value"].to_list()
    from_a = vals == [1.0, 5.0]
    from_b = vals == [2.0, 3.0]
    assert from_a or from_b, (
        f"Interleaving detected: rep=1 values {vals} mix rows from series A and B"
    )

    # 3. Rep 2 (no duplicates) should be untouched
    rep2 = cleaned_sorted.filter(pl.col("rep") == 2)
    assert rep2.height == 2

    # 4. Deterministic: same input order → same output
    cleaned_again = clean_df(df, grouping_cols).sort(grouping_cols + ["rep", "t"])
    assert cleaned_sorted.equals(cleaned_again), "clean_df is not deterministic"


def test_prepare_splits_deterministic(tmp_path):
    """Calling prepare_splits twice with the same seed should produce identical splits."""
    grouping_cols = ["g1", "g2"]
    surrogate_input_cols = ["g1", "g2", "input1"]
    output_cols = ["out1", "out2", "out3"]
    num_timesteps = 3
    n_params = 20
    n_reps = 5

    rows = []
    for i in range(n_params):
        for rep in range(n_reps):
            for t in range(num_timesteps):
                rows.append(
                    {
                        "g1": float(i),
                        "g2": float(i * 2),
                        "rep": rep,
                        "t": float(t),
                        "input1": float(i + rep) * 0.1,
                        "out1": float(t) * 0.5,
                        "out2": float(t) * 0.3,
                        "out3": float(t) * 0.1,
                    }
                )
    df = pl.DataFrame(rows)

    path = tmp_path / "test.parquet"
    df.write_parquet(path)

    kwargs = dict(
        val_prop=0.3,
        seed=42,
        grouping_cols=grouping_cols,
        surrogate_input_cols=surrogate_input_cols,
        output_cols=output_cols,
        num_timesteps=num_timesteps,
        trunc_until=None,
        input_col_to_source_col_dict=None,
    )
    train_a, val_a, _, _ = prepare_splits(path, **kwargs)
    train_b, val_b, _, _ = prepare_splits(path, **kwargs)

    assert train_a.equals(train_b), "Train splits differ across calls with same seed"
    assert val_a.equals(val_b), "Val splits differ across calls with same seed"


def test_exclude_output_variables(tmp_path):
    """Excluding output columns before calling prepare_splits should produce
    the same normalisation stats for the remaining columns."""
    grouping_cols = ["g1", "g2"]
    surrogate_input_cols = ["g1", "g2", "input1"]
    all_output_cols = ["out1", "out2", "out3"]
    num_timesteps = 3
    n_params = 20
    n_reps = 5

    rows = []
    for i in range(n_params):
        for rep in range(n_reps):
            for t in range(num_timesteps):
                rows.append(
                    {
                        "g1": float(i),
                        "g2": float(i * 2),
                        "rep": rep,
                        "t": float(t),
                        "input1": float(i + rep) * 0.1,
                        "out1": float(t) * 0.5,
                        "out2": float(t) * 999.0,  # distinct values so stats differ
                        "out3": float(t) * 0.1,
                    }
                )
    df = pl.DataFrame(rows)
    path = tmp_path / "test.parquet"
    df.write_parquet(path)

    common = dict(
        val_prop=0.3,
        seed=42,
        grouping_cols=grouping_cols,
        surrogate_input_cols=surrogate_input_cols,
        num_timesteps=num_timesteps,
        trunc_until=None,
        input_col_to_source_col_dict=None,
    )

    # Baseline: all output cols
    _, _, _, baseline_stats = prepare_splits(
        path, output_cols=all_output_cols, **common
    )

    # Exclude "out2" before calling the pipeline (mirrors train.py logic)
    excluded = {"out2"}
    filtered_cols = [v for v in all_output_cols if v not in excluded]
    _, _, _, filtered_stats = prepare_splits(path, output_cols=filtered_cols, **common)

    # Filtered stats should match the baseline stats at the kept indices
    kept_idx = [all_output_cols.index(v) for v in filtered_cols]
    assert torch.allclose(filtered_stats.means, baseline_stats.means[kept_idx])
    assert torch.allclose(filtered_stats.stds, baseline_stats.stds[kept_idx])


def test_left_truncate_and_recalculate_static_inputs():
    """Test that truncation drops rows before the boundary and recalculates
    static input columns from the source column at the new first timestep."""
    # Two (param_index, rep) groups, three timesteps each.
    # "static_input" is the column that should be recalculated,
    # "source" is the column whose value at the new t_min provides the replacement.
    df = pl.DataFrame(
        {
            "param_index": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "rep": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "static_input": [
                10.0,  # param index 0, rep 1 — initially source@t=0
                10.0,
                10.0,
                20.0,  # param index 0, rep 2
                20.0,
                20.0,
                ####
                30.0,  # param index 1, rep 1
                30.0,
                30.0,
                40.0,  # param index 1, rep 2
                40.0,
                40.0,
            ],
            "source": [
                10.0,  # param 0, rep 1 — source changes over time
                11.0,
                12.0,
                20.0,  # param 0, rep 2
                21.0,
                22.0,
                ####
                30.0,  # param 1, rep 1
                31.0,
                32.0,
                40.0,  # param 1, rep 2
                41.0,
                42.0,
            ],
            "other_col": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        }
    )

    result = left_truncate_and_recalculate_static_inputs(
        df,
        incl_start_t_boundary=1.0,
        input_col_to_source_col_dict={"static_input": "source"},
    )

    # 1. All rows before t=1.0 should be gone
    assert result.filter(pl.col("t") < 1.0).height == 0, (
        "Rows before boundary were not removed"
    )
    # 2. Should keep only t=1.0 and t=2.0 → 2 timesteps × 4 groups = 8 rows
    assert result.height == 8

    # 3. static_input should now equal source's value at t=1.0 for each group
    for param_idx in [0, 1]:
        for rep in [1, 2]:
            group = result.filter(
                (pl.col("param_index") == param_idx) & (pl.col("rep") == rep)
            )
            expected_static = group.filter(pl.col("t") == 1.0)["source"][0]
            actual_statics = group["static_input"].to_list()
            assert all(v == expected_static for v in actual_statics), (
                f"static_input not recalculated correctly for param_index={param_idx}, rep={rep}: "
                f"expected all {expected_static}, got {actual_statics}"
            )

    # 4. Other columns should be unaffected
    expected_other = result.filter((pl.col("param_index") == 0) & (pl.col("rep") == 1))[
        "other_col"
    ].to_list()
    assert expected_other == [2.0, 3.0], (
        f"Non-static columns were unexpectedly modified: {expected_other}"
    )


def test_make_tensors_shapes_and_z_scoring():
    """make_tensors should return inputs of shape (N, n_in), outputs of shape
    (N, n_out, T), and z-score both so mean==0 and std==1 per channel."""
    surrogate_input_cols = ["in1", "in2"]
    output_cols = ["out1", "out2", "out3"]
    num_timesteps = 4
    n_params = 6
    n_reps = 5
    n_samples = n_params * n_reps

    rows = []
    for p in range(n_params):
        for rep in range(n_reps):
            # Inputs are constant over t within each (param, rep)
            in1 = float(p) + 0.1 * rep
            in2 = float(p) * 2.0 - rep
            for t in range(num_timesteps):
                rows.append(
                    {
                        "param_index": p,
                        "rep": rep,
                        "t": float(t),
                        "in1": in1,
                        "in2": in2,
                        "out1": float(p) + float(t) * 0.5 + 0.01 * rep,
                        "out2": float(p) * 3.0 - float(t) + rep,
                        "out3": float(t) * 0.1 + float(p) * 0.2 + rep * 0.05,
                    }
                )
    df = pl.DataFrame(rows)

    inputs, outputs, in_stats, out_stats = make_tensors(
        df,
        surrogate_input_cols=surrogate_input_cols,
        output_cols=output_cols,
        num_timesteps=num_timesteps,
        z_score=True,
    )

    # Inputs are sliced to the first timestep; outputs keep the full time dim
    assert inputs.shape == (n_samples, len(surrogate_input_cols))
    assert outputs.shape == (n_samples, len(output_cols), num_timesteps)

    # Stats are computed before the first-timestep slice, so they keep T
    assert in_stats.means.shape == (len(surrogate_input_cols), num_timesteps)
    assert in_stats.stds.shape == (len(surrogate_input_cols), num_timesteps)
    assert out_stats.means.shape == (len(output_cols), num_timesteps)
    assert out_stats.stds.shape == (len(output_cols), num_timesteps)

    # Outputs: per (channel, timestep) z-scoring across the sample dim
    out_mean = outputs.mean(dim=0)
    out_std = outputs.std(dim=0)
    assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5)
    assert torch.allclose(out_std, torch.ones_like(out_std), atol=1e-5)

    # Inputs are constant over t, so the first-timestep slice is still
    # zero-mean / unit-std per channel across samples
    in_mean = inputs.mean(dim=0)
    in_std = inputs.std(dim=0)
    assert torch.allclose(in_mean, torch.zeros_like(in_mean), atol=1e-5)
    assert torch.allclose(in_std, torch.ones_like(in_std), atol=1e-5)


def test_get_random_batch_shapes_and_offset():
    """get_random_batch should return x_inputs (B, T-1, C), static_inputs
    (B, n_in), x_targets (B, T, C); inputs and targets are aligned by index
    with the static inputs, x_inputs is x_targets minus the last timestep,
    and it is deterministic given a seeded torch.Generator."""
    N, C, T, n_in = 30, 4, 8, 3
    batch_size = 12

    # Encode the source index in both tensors so we can verify alignment:
    # static_input_data[i, 0] == i, and x_data[i, 0, 0] == i.
    static_input_data = torch.zeros((N, n_in), dtype=torch.float32)
    static_input_data[:, 0] = torch.arange(N, dtype=torch.float32)
    static_input_data[:, 1:] = torch.randn((N, n_in - 1))

    x_data = torch.arange(N * C * T, dtype=torch.float32).view(N, C, T)
    x_data[:, 0, 0] = torch.arange(N, dtype=torch.float32)

    gen_a = torch.Generator(device="cpu").manual_seed(123)
    gen_b = torch.Generator(device="cpu").manual_seed(123)
    gen_c = torch.Generator(device="cpu").manual_seed(456)

    static_inputs, x_inputs, x_targets = get_random_batch(
        gen_a, static_input_data, x_data, batch_size
    )

    # Shapes
    assert static_inputs.shape == (batch_size, n_in)
    assert x_inputs.shape == (batch_size, T - 1, C)
    assert x_targets.shape == (batch_size, T, C)

    # x_inputs is x_targets without the final timestep
    assert torch.equal(x_inputs, x_targets[:, :-1, :])

    # Per-row alignment: static and x must come from the same source idx.
    # static_inputs[b, 0] holds the source index; x_targets[b, 0, 0] should match.
    src_idx_static = static_inputs[:, 0]
    src_idx_x = x_targets[:, 0, 0]
    assert torch.equal(src_idx_static, src_idx_x)
    # And those indices must be valid samples
    assert (src_idx_static >= 0).all() and (src_idx_static < N).all()

    # Determinism: same seed → identical 3-tuple
    static_inputs_b, x_inputs_b, x_targets_b = get_random_batch(
        gen_b, static_input_data, x_data, batch_size
    )
    assert torch.equal(static_inputs, static_inputs_b)
    assert torch.equal(x_inputs, x_inputs_b)
    assert torch.equal(x_targets, x_targets_b)

    # Different seed → (very likely) different selection
    static_inputs_c, _, _ = get_random_batch(
        gen_c, static_input_data, x_data, batch_size
    )
    assert not torch.equal(static_inputs, static_inputs_c)
