import polars as pl
import torch

from dataclasses import dataclass
from colorama import Fore, init
import omegaconf
from collections.abc import Iterator

# autorest=True automatically resets color after each print statement
init(autoreset=True)


@dataclass
class DatasetConfig:
    surrogate_input_variables: list[str] = omegaconf.MISSING
    grouping_variables: list[str] = omegaconf.MISSING
    output_variables: list[str] = omegaconf.MISSING
    num_timesteps: int = omegaconf.MISSING
    path: str = omegaconf.MISSING
    reps_for_comparison_plot: int | None = None
    trunc_until: int | None = None
    input_col_to_source_col: dict[str, str] | None = None
    observed_output_variables: list[str] | None = None
    exclude_output_variables: list[str] | None = None


@dataclass
class DatasetMeansAndStds:
    means: torch.Tensor
    stds: torch.Tensor


def clean_df(df: pl.DataFrame, grouping_cols):
    """Remove duplicate timeseries that share the same (group, rep) identity.

    Some parameter groups contain multiple complete timeseries with the same rep
    id. Within each (group, rep) sorted by original row order, we count
    first-timestep boundaries to assign a series id, then keep only series 1.
    """
    assert "rep" in df.columns and "t" in df.columns

    df = df.with_row_index("__orig_order")

    # Check if duplicates exist at the first timestep
    df_first_t = df.filter(pl.col("t") == pl.col("t").min().over(grouping_cols))
    has_dupes = (
        df_first_t.group_by(grouping_cols + ["rep"]).len().filter(pl.col("len") > 1)
    )

    if has_dupes.height > 0:
        print(
            f"{Fore.RED}[Warning] Duplicated rep ids detected at the first timestep "
            f"in {has_dupes.height} cases, only keeping first occurrence."
        )

        # Within each (group, rep), duplicate timeseries appear as contiguous
        # blocks where t resets back to the start value.
        # e.g. t = [10, 11, 12, ..., 59, 10, 11, 12, ..., 59]
        #                              ^^^ reset here
        # We count occurrences of t == group_min_t via cumsum to assign an
        # incrementing series id per block, then keep only series 1.
        df = df.sort(grouping_cols + ["rep", "__orig_order"])
        df = df.with_columns(
            pl.col("t")
            .eq(pl.col("t").min().over(grouping_cols))
            .cum_sum()
            .over(grouping_cols + ["rep"])
            .alias("__series_id")
        )
        df = df.filter(pl.col("__series_id") == 1).drop("__series_id")

    return df.drop("__orig_order")


def left_truncate_and_recalculate_static_inputs(
    df: pl.DataFrame,
    incl_start_t_boundary: float,
    input_col_to_source_col_dict: dict[str, str],
):
    # Discard all entries strictly before the starting boundary time
    df = df.filter(pl.col("t") >= incl_start_t_boundary)
    # Select the source_col's value at the first timestep for each param_index and rep
    # and assign it to every entry of the input_col
    df = df.with_columns(
        [
            pl.col(v)
            .filter(pl.col("t") == pl.col("t").min())
            .first()
            .over("param_index", "rep")
            .alias(k)
            for k, v in input_col_to_source_col_dict.items()
        ]
    )
    print(
        f"{Fore.BLUE}[INFO] Truncated up to {incl_start_t_boundary} and recalculated static inputs: {list(input_col_to_source_col_dict.keys())}"
    )
    return df


def prepare_splits(
    fp,
    val_prop,
    seed,
    grouping_cols,
    surrogate_input_cols,
    output_cols,
    num_timesteps,
    trunc_until,
    input_col_to_source_col_dict,
):
    ######### Build param_index, clean and truncate ####################
    df = pl.read_parquet(fp).with_columns(
        (pl.struct(grouping_cols).rank("dense") - 1).alias("param_index")
    )
    df = clean_df(df, grouping_cols)
    if trunc_until is not None:
        df = left_truncate_and_recalculate_static_inputs(
            df, trunc_until, input_col_to_source_col_dict
        )
    ######### Calculate dataset mean and stds ####################

    df_sorted = df.sort(["rep", "param_index", "t"])
    T = num_timesteps
    n_cols_in = len(surrogate_input_cols)
    n_cols_out = len(output_cols)
    n_samples = len(df_sorted) // T
    inputs_np = df_sorted.select(surrogate_input_cols).cast(pl.Float32).to_numpy()
    outputs_np = df_sorted.select(output_cols).cast(pl.Float32).to_numpy()
    inputs_tensor = (
        torch.from_numpy(inputs_np).view(n_samples, T, n_cols_in).permute(0, 2, 1)
    )
    outputs_tensor = (
        torch.from_numpy(outputs_np).view(n_samples, T, n_cols_out).permute(0, 2, 1)
    )

    input_means = inputs_tensor.mean((0))
    input_stds = inputs_tensor.std((0))

    output_means = outputs_tensor.mean((0))
    output_stds = outputs_tensor.std((0))

    ######### Split data ####################

    # We want to split on unique param_index so that we don't have some realisations
    # with the same inputs in both train and val
    # .unique() returns hash-order (non-deterministic), so sort first for reproducible shuffling
    unique_param_indices = df["param_index"].unique().sort().shuffle(seed)
    n_val = int(len(unique_param_indices) * val_prop)
    val_indices = unique_param_indices.head(n_val)
    train_indices = unique_param_indices.tail(-n_val)
    # Need to convert indices to list so that polars knows we are performing a set membership check
    train_df = df.filter(pl.col("param_index").is_in(train_indices.to_list()))
    val_df = df.filter(pl.col("param_index").is_in(val_indices.to_list()))

    return (
        train_df,
        val_df,
        DatasetMeansAndStds(input_means, input_stds),
        DatasetMeansAndStds(output_means, output_stds),
    )


def make_tensors(
    df: pl.DataFrame,
    surrogate_input_cols,
    output_cols,
    num_timesteps,
    device="cpu",
    z_score=True,
    means_and_stds=None,
):
    # 1. Global sort ensures data is contiguous for reshaping
    df_sorted = df.sort(["rep", "param_index", "t"])

    # 2. Determine dimensions
    T = num_timesteps
    n_cols_in = len(surrogate_input_cols)
    n_cols_out = len(output_cols)
    n_samples = len(df_sorted) // T

    # 3. Vectorized extraction
    # Select columns, cast to Float32, and convert to numpy
    # as a single block.
    inputs_np = df_sorted.select(surrogate_input_cols).cast(pl.Float32).to_numpy()
    outputs_np = df_sorted.select(output_cols).cast(pl.Float32).to_numpy()

    # 4. Reshape and Permute
    # Current shape: (N_samples * T, Channels)
    # Target shape: (N_samples, Channels, T)

    # View as (N_samples, T, Channels) then swap axes to (N_samples, Channels, T)
    inputs_tensor = (
        torch.from_numpy(inputs_np).view(n_samples, T, n_cols_in).permute(0, 2, 1)
    )
    outputs_tensor = (
        torch.from_numpy(outputs_np).view(n_samples, T, n_cols_out).permute(0, 2, 1)
    )
    # TODO: Test that this behaviour works correctly
    if means_and_stds is None:
        input_means = inputs_tensor.mean((0))
        input_stds = inputs_tensor.std((0))

        output_means = outputs_tensor.mean((0))
        output_stds = outputs_tensor.std((0))
    else:
        assert isinstance(means_and_stds, tuple)
        input_means_and_stds, output_means_and_stds = means_and_stds
        input_means = input_means_and_stds.means
        input_stds = input_means_and_stds.stds

        output_means = output_means_and_stds.means
        output_stds = output_means_and_stds.stds

    if z_score:
        inputs_tensor = (inputs_tensor - input_means[None, ...]) / input_stds[None, ...]
        outputs_tensor = (outputs_tensor - output_means[None, ...]) / output_stds[
            None, ...
        ]

    assert inputs_tensor.size(1) == n_cols_in and outputs_tensor.size(1) == n_cols_out

    # NOTE: Inputs are constant over T, the df contains values for each T but we don't need that
    # treat it as a first timestep,
    inputs_tensor = inputs_tensor[:, :, 0]

    return (
        inputs_tensor.to(device),  # (N_samples, static_input_dim)
        outputs_tensor.to(device),  # (N_samples, x_input_dim, T)
        DatasetMeansAndStds(input_means, input_stds),
        DatasetMeansAndStds(output_means, output_stds),
    )


def get_random_batch(
    gen: torch.Generator,
    static_input_data: torch.Tensor,
    x_data: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch a random batch from a tensor dataset."""
    assert x_data.size(0) == static_input_data.size(0)
    # Select realisation
    idx = torch.randint(x_data.size(0), (batch_size,), generator=gen, device=gen.device)

    # x-inputs exclude last timestep

    # (batch_size, T-1, n_output_cols)
    x_inputs = x_data[idx, ..., :-1].permute(0, 2, 1)

    static_inputs = static_input_data[idx]

    # (batch_size, T, n_output_cols)
    x_targets = x_data[idx, ...].permute(0, 2, 1)

    return static_inputs, x_inputs, x_targets


def get_data_iterator(
    static_input_data: torch.Tensor, x_data: torch.Tensor, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Returns an iterator over a tensor dataset.
    Batches consist of inputs, targets and a mask which indicates where
    zero-padding of the batch has occurred to maintain a constant batch size."""
    assert x_data.size(0) == static_input_data.size(0)
    n_samples = x_data.size(0)
    for idx in range(0, n_samples, batch_size):
        x_chunk = x_data[idx : idx + batch_size].permute(0, 2, 1)
        static_chunk = static_input_data[idx : idx + batch_size]

        padding_amount = batch_size - x_chunk.size(0)
        x_chunk = torch.concat(
            [
                x_chunk,
                -1
                * torch.ones(
                    (padding_amount, *x_chunk.shape[1:]),
                    dtype=x_chunk.dtype,
                    device=x_chunk.device,
                ),
            ]
        )
        static_chunk = torch.concat(
            [
                static_chunk,
                -1
                * torch.ones(
                    (padding_amount, *static_chunk.shape[1:]),
                    dtype=static_chunk.dtype,
                    device=static_chunk.device,
                ),
            ]
        )
        mask = torch.concat(
            [
                torch.ones(
                    (batch_size - padding_amount,),
                    dtype=x_chunk.dtype,
                    device=x_chunk.device,
                ),
                torch.zeros(
                    (padding_amount,), dtype=x_chunk.dtype, device=x_chunk.device
                ),
            ]
        )
        x_inputs = x_chunk[:, :-1, ...]
        x_targets = x_chunk
        yield static_chunk, x_inputs, x_targets, mask
