#!/usr/bin/env python
"""Time-varying R_t for South Africa: simulate -> train surrogate -> MCMC -> posterior predictive.

Script version of time_varying_inference.ipynb. Run from the repository root:

    uv run time_varying_inference.py runs/sa --n-params 20000 --max-steps 10000 --num-samples 500

The output directory gets a date-time suffix (runs/sa_2026-10-19_14-30-00) and contains:
    args.json                  the CLI arguments used
    prior_predictive.png       prior draws against the WHO series (the notebook's prior predictive cell)
    data/covid_sa_data.parquet metis-format training set
    configs/dataset/           dataset config handed to train.py via --config-dir
    surrogate/                 hydra run dir of train.py (checkpoints/last.pt, .hydra/config.yaml)
    surrogate_realisations.png train.py's sampling plot: surrogate draws against held-out simulations
    mcmc/samples.pt            posterior draws, shape (chains, draws, ...) per site
    mcmc/summary.txt           per-site summary with ESS and split R-hat
    ppc_surrogate.png          posterior predictive through the flow's reverse pass
    ppc_simulator.png          posterior predictive through the simulator (also effective R_t)
    ppc_coverage.json          share of observed weeks inside the 95% predictive bands

Entry points:
    --dataset PATH      use an existing metis-format parquet instead of simulating one
    --checkpoint PATH   use a trained checkpoint (its config and dataset path are stored inside it)
                        and skip simulation and training; --dataset then overrides the dataset used
                        for the normalisation statistics
    --surrogate-dir DIR alias for --checkpoint DIR/checkpoints/last.pt
    --combine RUN ...   stack the chains of finished runs instead of sampling (cross-chain R-hat, pooled
                        predictive checks); needs --checkpoint and the same window and IFR as those runs
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve
from scipy.stats import gamma as gamma_dist

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# ----------------------------------------------------------------------------
# Simulator (identical to the notebook)
# ----------------------------------------------------------------------------


def delay_pmf(K, mean, sd):
    """Discretised gamma pmf; index k is the weight for a lag of k+1 days."""
    shape, scale = (mean / sd) ** 2, sd**2 / mean
    k = np.arange(K)
    w = gamma_dist.cdf(k + 1, shape, scale=scale) - gamma_dist.cdf(
        k, shape, scale=scale
    )
    return w / w.sum()


GEN = delay_pmf(40, mean=6.5, sd=6.5 * 0.62)  # generation interval (Flaxman et al.)
PI_C = delay_pmf(30, mean=8.0, sd=4.0)  # infection -> case report
PI_D = delay_pmf(70, mean=22.9, sd=9.1)  # infection -> death (Flaxman et al.)

N = 60e6  # population of South Africa
BURN_IN_WEEKS = 3  # simulated before the first observed week, using R_0 and ifr_0
SEED_DAYS = 6  # seed i0 infections/day for this long (Flaxman et al.)
IMPORTS = 1.0  # constant daily importation

STATIC = ["log_omega", "log_rho", "log_ifr", "log_i0", "log_phi_c", "log_phi_d"]
OBSERVED = ["log_R_t"]
PREDICTED = ["log_cases", "log_deaths"]


def simulate_covid(theta, n_weeks, seed=None, deterministic=False):
    """theta: dict with log_R of shape (B, n_weeks) and the STATIC scalars of shape (B,).
    Returns weekly arrays of shape (B, n_weeks)."""
    rng = np.random.default_rng(seed)
    log_R = theta["log_R"]
    B = log_R.shape[0]
    T = (BURN_IN_WEEKS + n_weeks) * 7

    def daily(log_path):  # weekly paths -> daily, burn-in weeks held at the first value
        full = np.concatenate(
            [np.repeat(log_path[:, :1], BURN_IN_WEEKS, axis=1), log_path], axis=1
        )
        return np.exp(np.repeat(full, 7, axis=1))

    Rt = daily(log_R)
    ifr = np.exp(theta["log_ifr"])[:, None]  # constant over time
    omega, rho = np.exp(theta["log_omega"]), np.exp(theta["log_rho"])

    inf, R_eff = np.zeros((B, T)), np.zeros((B, T))
    S = np.full(B, N)
    seed_rate = np.broadcast_to(np.exp(theta["log_i0"])[:, None], (B, SEED_DAYS))
    inf[:, :SEED_DAYS] = seed_rate if deterministic else rng.poisson(seed_rate)
    R_eff[:, :SEED_DAYS] = Rt[:, :SEED_DAYS]
    S = S - inf[:, :SEED_DAYS].sum(axis=1)
    L = len(GEN)
    for t in range(SEED_DAYS, T):
        s = max(0, t - L)
        R_eff[:, t] = Rt[:, t] * S / N
        lam = R_eff[:, t] * (inf[:, s:t] @ GEN[: t - s][::-1]) + IMPORTS
        i_t = np.minimum(lam if deterministic else rng.poisson(lam), S)
        inf[:, t] = i_t
        S = S - i_t + omega * (N - S)

    mu_c = rho[:, None] * fftconvolve(inf, PI_C[None, :], axes=1)[:, :T]
    mu_d = ifr * fftconvolve(inf, PI_D[None, :], axes=1)[:, :T]

    def weekly(x):
        return x[:, BURN_IN_WEEKS * 7 :].reshape(B, n_weeks, 7).sum(axis=2)

    mu_c_w, mu_d_w = weekly(mu_c), weekly(mu_d)
    if deterministic:
        cases, deaths = mu_c_w, mu_d_w
    else:
        phi_c, phi_d = (
            np.exp(theta["log_phi_c"])[:, None],
            np.exp(theta["log_phi_d"])[:, None],
        )
        cases = rng.negative_binomial(phi_c, phi_c / (phi_c + mu_c_w))
        deaths = rng.negative_binomial(phi_d, phi_d / (phi_d + mu_d_w))
    return {
        "cases": cases,
        "deaths": deaths,
        "infections": weekly(inf),
        "R_eff_t": weekly(R_eff) / 7,
    }


# ----------------------------------------------------------------------------
# Prior (identical to the notebook)
# ----------------------------------------------------------------------------

PRIOR = dict(
    R0=(np.log(3.0), 0.3),
    sigma_R=0.2,
    log_ifr=(
        np.log(0.005),
        0.6,
    ),  # constant IFR: reported deaths per infection, log-normal
    log_omega=(np.log(1 / 730), np.log(1 / 90)),
    log_rho=(np.log(0.02), np.log(0.4)),
    seed_rate=0.03,
    log_phi=(np.log(5.0), np.log(200.0)),
)


def random_walk(rng, B, n_weeks, start, sigma):
    """numpy: log x_0 ~ N(start), log x_{w+1} | x_w ~ N(log x_w, sigma^2)."""
    x = np.empty((B, n_weeks))
    x[:, 0] = rng.normal(*start, B)
    for w in range(1, n_weeks):
        x[:, w] = x[:, w - 1] + rng.normal(0.0, sigma)
    return x


def sample_prior(B, n_weeks, seed=None):
    rng = np.random.default_rng(seed)
    sigma_R = np.abs(rng.normal(0.0, PRIOR["sigma_R"], B))
    tau = rng.exponential(1 / PRIOR["seed_rate"], B)
    return {
        "log_R": random_walk(rng, B, n_weeks, PRIOR["R0"], sigma_R),
        "log_omega": rng.uniform(*PRIOR["log_omega"], B),
        "log_rho": rng.uniform(*PRIOR["log_rho"], B),
        "log_ifr": rng.normal(*PRIOR["log_ifr"], B),
        "log_i0": np.log(rng.exponential(tau)),
        "log_phi_c": rng.uniform(*PRIOR["log_phi"], B),
        "log_phi_d": rng.uniform(*PRIOR["log_phi"], B),
        "sigma_R": sigma_R,
    }


# ----------------------------------------------------------------------------
# Data and training set
# ----------------------------------------------------------------------------


def load_who(csv, start, end):
    who = pd.read_csv(csv, parse_dates=["Date_reported"])
    c = who[who.Country == "South Africa"].sort_values("Date_reported")
    c = c[(c.Date_reported >= start) & (c.Date_reported <= end)]
    print(
        f"WHO data: {len(c)} weeks, {c.Date_reported.min().date()} -> {c.Date_reported.max().date()}"
    )
    return c.New_cases.clip(lower=0).to_numpy(), c.New_deaths.clip(lower=0).to_numpy()


def prior_predictive(obs_cases, obs_deaths, n_weeks, path, n_draws=5000, seed=0):
    """The notebook's prior predictive check: prior draws through the simulator against the data."""
    theta = sample_prior(n_draws, n_weeks, seed=seed)
    out = simulate_covid(theta, n_weeks, seed=seed + 1)
    print(
        f"prior predictive: share of draws with fewer than 1000 cases in every week: {(out['cases'].max(axis=1) < 1e3).mean():.1%}"
    )
    tw = np.arange(n_weeks)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(tw, out["R_eff_t"][:15].T, color="#2E86AB", alpha=0.4, lw=1)
    axes[0].axhline(1, color="grey", ls=":")
    axes[0].set(
        xlabel="week",
        ylabel="effective R_t",
        yscale="log",
        title="effective R_t prior draws (15)",
    )
    for ax, key, obs in [
        (axes[1], "cases", obs_cases),
        (axes[2], "deaths", obs_deaths),
    ]:
        q = np.percentile(out[key], [2.5, 25, 50, 75, 97.5], axis=0)
        ax.fill_between(tw, q[0], q[-1], color="#2E86AB", alpha=0.2, label="95%")
        ax.fill_between(tw, q[1], q[-2], color="#2E86AB", alpha=0.4, label="50%")
        ax.plot(tw, obs, "k", lw=1.5, label="WHO")
        ax.set(
            xlabel="week",
            ylabel=f"weekly {key}",
            yscale="log",
            title=f"{key}: prior predictive vs data",
        )
        ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_dataset(n_params, n_reps, n_weeks, seed=0):
    rng = np.random.default_rng(seed)
    theta = sample_prior(n_params, n_weeks, seed=rng.integers(2**31))
    pid, t = (
        np.repeat(np.arange(n_params), n_weeks),
        np.tile(np.arange(n_weeks), n_params),
    )
    frames = []
    for rep in range(n_reps):
        out = simulate_covid(theta, n_weeks, seed=rng.integers(2**31))
        frame = {
            "param_id": pid,
            "rep": rep,
            "t": t,
            "cases": out["cases"].reshape(-1),
            "deaths": out["deaths"].reshape(-1),
            "log_cases": np.log1p(out["cases"].reshape(-1)),
            "log_deaths": np.log1p(out["deaths"].reshape(-1)),
            "log_R_t": theta["log_R"].reshape(-1),
            "R_eff_t": out["R_eff_t"].reshape(-1),
        }
        frame |= {k: theta[k][pid] for k in STATIC + ["sigma_R"]}
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def simulate_dataset(run_dir, args, n_weeks):
    """Simulate the training set from the prior and save it under run_dir/data."""
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "covid_sa_data.parquet"
    t0 = time.time()
    df = make_dataset(args.n_params, args.n_reps, n_weeks, seed=args.seed)
    df.to_parquet(parquet)
    print(
        f"training set: {args.n_params} parameter sets x {args.n_reps} reps = {len(df)} rows ({time.time() - t0:.0f}s)"
    )
    return parquet


def check_dataset(parquet, n_weeks):
    """Validate an existing parquet against the observation window; returns its number of reps."""
    parquet = Path(parquet)
    if not parquet.is_file():
        raise FileNotFoundError(f"dataset not found: {parquet}")
    cols = pd.read_parquet(parquet, columns=["t", "rep"])
    weeks = int(cols["t"].max()) + 1
    if weeks != n_weeks:
        raise ValueError(
            f"dataset has {weeks} weeks per series, the observation window has {n_weeks}"
        )
    n_reps = int(cols["rep"].nunique())
    print(
        f"using existing dataset: {parquet} ({len(cols) // (weeks * n_reps)} parameter sets x {n_reps} reps)"
    )
    return n_reps


def write_dataset_config(run_dir, parquet, n_weeks, n_reps):
    """Write the dataset config that train.py picks up through --config-dir."""
    cfg_dir = run_dir / "configs" / "dataset"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "covid_sa.yaml").write_text(
        f"surrogate_input_variables: {STATIC}\n"
        f"grouping_variables: [param_id]\n"
        f"output_variables: {PREDICTED + OBSERVED}\n"
        f"observed_output_variables: {OBSERVED}\n"
        f"num_timesteps: {n_weeks}\n"
        f"path: {Path(parquet).resolve()}\n"
        f"reps_for_comparison_plot: {n_reps}\n"  # train.py only plots parameter sets with exactly this many reps
        "trunc_until: null\n"
        "input_col_to_source_col: null\n"
        "exclude_output_variables: null\n"
    )
    return run_dir / "configs"


# ----------------------------------------------------------------------------
# Surrogate: train and load
# ----------------------------------------------------------------------------


def train_surrogate(run_dir, config_dir, args):
    surrogate_dir = (run_dir / "surrogate").resolve()
    sampling = (
        []
        if args.no_samples_plot
        else [
            "sampling.generate_samples=True",
            f"sampling.samples_name={(run_dir / 'surrogate_realisations.png').resolve()}",
            f"sampling.subsample={args.subsample}",
        ]
    )
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "--config-name",
        "default",
        "--config-dir",
        str(config_dir.resolve()),
        "dataset=covid_sa",
        "model=itarflow_cross_attn",
        f"model.channels={args.channels}",
        f"model.num_layers={args.num_layers}",
        f"model.num_blocks={args.num_blocks}",
        f"model.num_heads={args.num_heads}",
        f"peak_lr={args.lr}",
        f"max_steps={args.max_steps}",
        f"eval_every={args.eval_every}",
        f"batch_size={args.batch_size}",
        f"num_gpus={args.num_gpus}",
        f"checkpointer.save_every={args.ckpt_every}",
        f"seed={args.seed}",
        f"hydra.run.dir={surrogate_dir}",
        *sampling,
    ]
    print("training surrogate:\n  " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=REPO, check=True)
    return surrogate_dir


class Surrogate:
    """Trained flow plus the normalisation statistics the training pipeline used."""

    def __init__(self, checkpoint, device, compile_model=True, dataset=None):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from dataloader import prepare_splits

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])  # train.py stores the config it ran with
        ds = cfg.dataset
        if dataset is not None:
            ds.path = str(Path(dataset).resolve())
        elif not Path(ds.path).is_absolute():
            ds.path = str(
                (REPO / ds.path).resolve()
            )  # train.py was run from the repo root
        if not Path(ds.path).is_file():
            raise FileNotFoundError(
                f"dataset for normalisation statistics not found: {ds.path}"
            )
        print(
            f"surrogate: {checkpoint} (step {ckpt['step']}), statistics from {ds.path}"
        )
        if (
            list(ds.observed_output_variables) != OBSERVED
            or list(ds.surrogate_input_variables) != STATIC
        ):
            raise ValueError(
                f"checkpoint was trained with observed={list(ds.observed_output_variables)} and "
                f"static={list(ds.surrogate_input_variables)}, but this script expects observed={OBSERVED} "
                f"and static={STATIC} (constant IFR). Retrain the surrogate for this model."
            )
        all_out = list(ds.output_variables)
        self.pred_idx = [i for i, v in enumerate(all_out) if v not in OBSERVED]
        self.obs_idx = [all_out.index(v) for v in OBSERVED]
        self.T = ds.num_timesteps
        self.device = device

        # runtime overrides train.py applies to the saved (pre-override) config
        cfg.model.out_channels = len(self.pred_idx)
        cfg.model.num_timesteps = self.T
        cfg.model.static_in = len(STATIC)
        cfg.model.cross_attn.cond_dim = len(self.obs_idx)
        nf = instantiate(cfg.model).to(device)
        nf.load_state_dict(ckpt["model"])
        nf.eval()
        nf.requires_grad_(False)
        self.nf = nf
        self.nf_c = (
            torch.compile(nf, mode="reduce-overhead", dynamic=False)
            if compile_model
            else nf
        )
        self.compiled = compile_model

        _, _, in_stats, out_stats = prepare_splits(  # same arguments as train.py
            ds.path,
            0.1,
            42,
            ds.grouping_variables,
            ds.surrogate_input_variables,
            ds.output_variables,
            ds.num_timesteps,
            ds.trunc_until,
            ds.input_col_to_source_col,
        )
        f32 = lambda x: torch.as_tensor(
            np.asarray(x), dtype=torch.float32, device=device
        )
        self.pred_mean, self.pred_std = (
            f32(out_stats.means[self.pred_idx].T),
            f32(out_stats.stds[self.pred_idx].T),
        )
        self.cond_mean, self.cond_std = (
            f32(out_stats.means[self.obs_idx].T),
            f32(out_stats.stds[self.obs_idx].T),
        )
        self.static_mean, self.static_std = (
            f32(in_stats.means[:, 0]),
            f32(in_stats.stds[:, 0]),
        )
        self.data_dim = self.T * len(self.pred_idx)

    def normalise_obs(self, cases, deaths):
        y = torch.as_tensor(
            np.stack([np.log1p(cases), np.log1p(deaths)], 1),
            dtype=torch.float32,
            device=self.device,
        )
        return ((y - self.pred_mean) / self.pred_std).unsqueeze(0)

    def loglik(self, paths, static, y):
        """paths: (B, T, 1) raw log_R_t; static: (B, 6) raw in STATIC order; y: (1, T, 2) normalised."""
        cond_seq = (paths - self.cond_mean) / self.cond_std
        static_n = (static - self.static_mean) / self.static_std
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        z, logdet_mean = self.nf_c(y.expand(paths.shape[0], -1, -1), static_n, cond_seq)
        if self.compiled:
            z, logdet_mean = z.clone(), logdet_mean.clone()
        log_pz = -0.5 * z.reshape(z.shape[0], -1).pow(2).sum(
            1
        ) - 0.5 * self.data_dim * math.log(2 * math.pi)
        return log_pz + self.data_dim * logdet_mean

    @torch.no_grad()
    def sample(self, paths, static, seed=0):
        """Draw weekly (cases, deaths) from the surrogate; returns (B, T, 2) on the count scale."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(
            paths.shape[0], self.T, len(self.pred_idx), device=self.device, generator=g
        )
        x, _ = self.nf.reverse(
            z,
            (static - self.static_mean) / self.static_std,
            (paths - self.cond_mean) / self.cond_std,
        )
        return torch.expm1(x * self.pred_std + self.pred_mean).clamp(min=0)


# ----------------------------------------------------------------------------
# MCMC with Pyro
# ----------------------------------------------------------------------------


def pyro_random_walk(name, T, start_loc, start_scale, sigma_scale, device):
    """Non-centred random walk: log x_{w+1} = log x_w + sigma * eps_w, eps ~ N(0, 1)."""
    import pyro
    import pyro.distributions as pdist

    t = lambda v: torch.tensor(float(v), device=device)
    x0 = pyro.sample(f"{name}0", pdist.Normal(t(start_loc), t(start_scale)))
    sigma = pyro.sample(f"sigma_{name}", pdist.HalfNormal(t(sigma_scale)))
    eps = pyro.sample(
        f"eps_{name}", pdist.Normal(t(0.0), t(1.0)).expand([T - 1]).to_event(1)
    )
    return torch.cat([x0[None], x0 + torch.cumsum(sigma * eps, 0)])


def pyro_centred_walk(name, T, start_loc, start_scale, sigma_scale, device):
    """Centred random walk: the path itself is the latent, with the walk prior added as a factor.
    Better geometry than the non-centred form when the data pin the path down week by week."""
    import pyro
    import pyro.distributions as pdist

    t = lambda v: torch.tensor(float(v), device=device)
    sigma = pyro.sample(f"sigma_{name}", pdist.HalfNormal(t(sigma_scale)))
    x = pyro.sample(
        name, pdist.Normal(t(0.0), t(1.0)).expand([T]).to_event(1).mask(False)
    )
    pyro.factor(
        f"{name}_prior",
        pdist.Normal(t(start_loc), t(start_scale)).log_prob(x[0])
        + pdist.Normal(x[:-1], sigma).log_prob(x[1:]).sum(),
    )
    return x


def paths_from_samples(s, name):
    """Rebuild (draws, T) paths from the sites pyro_random_walk creates."""
    x0, sigma, eps = s[f"{name}0"], s[f"sigma_{name}"], s[f"eps_{name}"]
    return torch.cat(
        [x0[:, None], x0[:, None] + torch.cumsum(sigma[:, None] * eps, 1)], 1
    )


def make_posterior_model(sur, y_obs, parametrisation="noncentred", ifr=None):
    """ifr: fixed infection fatality ratio (not inferred); None draws it from the prior instead."""
    import pyro
    import pyro.distributions as pdist

    T, device = sur.T, sur.device
    t = lambda v: torch.tensor(float(v), device=device)
    walk = pyro_centred_walk if parametrisation == "centred" else pyro_random_walk

    def posterior_model():
        log_R = walk("log_R", T, *PRIOR["R0"], PRIOR["sigma_R"], device)
        tau = pyro.sample("tau", pdist.Exponential(t(PRIOR["seed_rate"])))
        u = pyro.sample("u_i0", pdist.Exponential(t(1.0)))  # non-centred: i0 = tau * u
        statics = {
            "log_omega": pyro.sample(
                "log_omega",
                pdist.Uniform(t(PRIOR["log_omega"][0]), t(PRIOR["log_omega"][1])),
            ),
            "log_rho": pyro.sample(
                "log_rho", pdist.Uniform(t(PRIOR["log_rho"][0]), t(PRIOR["log_rho"][1]))
            ),
            "log_ifr": t(math.log(ifr))
            if ifr is not None
            else pyro.sample(
                "log_ifr", pdist.Normal(t(PRIOR["log_ifr"][0]), t(PRIOR["log_ifr"][1]))
            ),
            "log_i0": torch.log(tau * u),
            "log_phi_c": pyro.sample(
                "log_phi_c",
                pdist.Uniform(t(PRIOR["log_phi"][0]), t(PRIOR["log_phi"][1])),
            ),
            "log_phi_d": pyro.sample(
                "log_phi_d",
                pdist.Uniform(t(PRIOR["log_phi"][0]), t(PRIOR["log_phi"][1])),
            ),
        }
        paths = log_R[None, :, None]  # (1, T, 1): the only observed sequence
        static = torch.stack([statics[k] for k in STATIC])[None]
        pyro.factor("surrogate_likelihood", sur.loglik(paths, static, y_obs)[0])

    return posterior_model


def run_mcmc(sur, y_obs, args, out_dir):
    """Chains run one after another (Pyro's multiprocess chains do not combine with a compiled CUDA model).
    Returns a dict of tensors with shape (chains, draws, ...)."""
    import pyro
    from pyro.infer import MCMC, NUTS
    from pyro.infer.autoguide.initialization import init_to_median

    ifr = None if args.infer_ifr else args.ifr
    print(
        "IFR:",
        "inferred from its prior"
        if ifr is None
        else f"fixed at {ifr:.4%} (reported deaths per infection)",
    )
    model = make_posterior_model(sur, y_obs, args.parametrisation, ifr=ifr)
    chains, diagnostics = [], {}
    for c in range(args.num_chains):
        pyro.set_rng_seed(args.seed + c)
        kernel = NUTS(
            model,
            init_strategy=init_to_median(),
            max_tree_depth=args.max_tree_depth,
            target_accept_prob=args.target_accept_prob,
            full_mass=args.full_mass,
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=args.warmup_samples,
            num_samples=args.num_samples,
            num_chains=1,
        )
        t0 = time.time()
        mcmc.run()
        d = mcmc.diagnostics()
        diagnostics[f"chain {c}"] = {
            "seconds": round(time.time() - t0, 1),
            "divergences": len(d["divergences"]["chain 0"]),
        }
        print(f"chain {c}: {diagnostics[f'chain {c}']}")
        chains.append(mcmc.get_samples())
    samples = {k: torch.stack([ch[k] for ch in chains]) for k in chains[0]}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in samples.items()}, out_dir / "samples.pt")
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    write_summary(samples, out_dir / "summary.txt")
    return samples


def combine_chains(run_dirs, out_dir, args):
    """Stack the chains of finished runs (one samples.pt each) and write the combined samples, diagnostics and
    summary. The runs must share checkpoint, window, IFR and parametrisation, and match this run's settings."""
    chains, diagnostics, settings = [], {}, []
    keys = ("checkpoint", "start", "end", "ifr", "infer_ifr", "parametrisation")
    for r in map(Path, run_dirs):
        chains.append(torch.load(r / "mcmc/samples.pt", map_location="cpu"))
        diag_file, args_file = r / "mcmc/diagnostics.json", r / "args.json"
        if diag_file.exists():
            diagnostics |= {
                f"{r.name}: {k}": v
                for k, v in json.loads(diag_file.read_text()).items()
            }
        prev = json.loads(args_file.read_text()) if args_file.exists() else {}
        settings.append({k: prev.get(k) for k in keys})
        print(
            f"loaded {r / 'mcmc/samples.pt'}: {chains[-1][next(iter(chains[-1]))].shape[0]} chain(s)"
        )
    mine = {k: getattr(args, k) for k in keys}
    for k in keys:
        values = {
            str(Path(s[k]).resolve()) if k == "checkpoint" and s[k] else str(s[k])
            for s in settings
        }
        current = (
            str(Path(mine[k]).resolve())
            if k == "checkpoint" and mine[k]
            else str(mine[k])
        )
        if len(values) > 1 or (values and values != {current}):
            raise ValueError(f"runs disagree on {k}: {values} (this run: {current})")
    if any(set(c) != set(chains[0]) for c in chains):
        raise ValueError("runs have different sample sites")
    samples = {k: torch.cat([c[k] for c in chains], dim=0) for k in chains[0]}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_dir / "samples.pt")
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"combined {samples[next(iter(samples))].shape[0]} chains")
    write_summary(samples, out_dir / "summary.txt")
    return samples


def write_summary(samples, path):
    from pyro.ops.stats import effective_sample_size, split_gelman_rubin

    lines = [
        f"{'site':16s} {'mean':>10s} {'sd':>10s} {'5%':>10s} {'95%':>10s} {'min ESS':>8s} {'max Rhat':>9s}"
    ]
    for name, x in samples.items():
        flat = x.reshape(x.shape[0], x.shape[1], -1).float()  # (chains, draws, K)
        try:
            ess = effective_sample_size(flat, chain_dim=0, sample_dim=1)
            rhat = split_gelman_rubin(flat, chain_dim=0, sample_dim=1)
            ess_s, rhat_s = f"{ess.min():8.0f}", f"{rhat.max():9.3f}"
            if (
                ess.min() < 0 or x.shape[1] < 50
            ):  # estimators are unreliable on short chains
                ess_s = f"{'n/a':>8s}"
        except Exception:  # too few draws for the estimators
            ess_s, rhat_s = f"{'n/a':>8s}", f"{'n/a':>9s}"
        v = flat.reshape(-1, flat.shape[-1])
        if v.shape[1] == 1:
            q = torch.quantile(v[:, 0], torch.tensor([0.05, 0.95], device=v.device))
            lines.append(
                f"{name:16s} {v.mean():10.4f} {v.std():10.4f} {q[0]:10.4f} {q[1]:10.4f} {ess_s} {rhat_s}"
            )
        else:
            lines.append(
                f"{name:16s} {'vector of ' + str(v.shape[1]):>43s} {ess_s} {rhat_s}"
            )
    text = "\n".join(lines)
    print(text)
    Path(path).write_text(text + "\n")


def posterior_draws(samples, n_pp, seed=0, ifr=None):
    """Flatten chains and pick n_pp draws: log R path (n_pp, T) and statics dict of (n_pp,) tensors."""
    s = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples.items()}
    g = torch.Generator().manual_seed(seed)
    log_R = (
        s["log_R"] if "log_R" in s else paths_from_samples(s, "log_R")
    )  # centred or non-centred
    sel = torch.randperm(log_R.shape[0], generator=g)[:n_pp].to(log_R.device)
    log_R = log_R[sel]
    log_ifr = (
        s["log_ifr"][sel]
        if "log_ifr" in s
        else torch.full((len(sel),), math.log(ifr), device=log_R.device)
    )
    statics = {
        "log_omega": s["log_omega"][sel],
        "log_rho": s["log_rho"][sel],
        "log_ifr": log_ifr,
        "log_i0": (s["tau"][sel] * s["u_i0"][sel]).log(),
        "log_phi_c": s["log_phi_c"][sel],
        "log_phi_d": s["log_phi_d"][sel],
    }
    return log_R, statics


# ----------------------------------------------------------------------------
# Posterior predictive checks
# ----------------------------------------------------------------------------


def _band(ax, tw, arr, color, label):
    q = np.percentile(arr, [2.5, 25, 50, 75, 97.5], axis=0)
    ax.fill_between(tw, q[0], q[4], color=color, alpha=0.2)
    ax.fill_between(tw, q[1], q[3], color=color, alpha=0.4, label=label)
    ax.plot(tw, q[2], color=color, lw=1)


def _coverage(arr, obs):
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return float(np.mean((obs >= lo) & (obs <= hi)))


def _counts_panels(axes, tw, cases, deaths, obs_cases, obs_deaths, color, label):
    for ax, arr, obs, name in [
        (axes[0], cases, obs_cases, "cases"),
        (axes[1], deaths, obs_deaths, "deaths"),
    ]:
        _band(ax, tw, arr, color, label)
        ax.plot(tw, obs, "k", lw=1.5, label="WHO")
        ax.set(
            yscale="log",
            xlabel="week",
            ylabel=f"weekly {name}",
            title=f"{label}: {name}",
        )
        ax.legend()


def ppc_surrogate(sur, log_R, statics, obs_cases, obs_deaths, path, seed=0):
    """Posterior predictive through the flow's reverse pass. Returns 95%-band coverage."""
    paths = log_R[:, :, None]
    static = torch.stack([statics[k] for k in STATIC], 1)
    pp = sur.sample(paths, static, seed=seed).cpu().numpy()  # (n_pp, T, 2)
    tw = np.arange(sur.T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _counts_panels(
        axes[:2],
        tw,
        pp[:, :, 0],
        pp[:, :, 1],
        obs_cases,
        obs_deaths,
        "#2E86AB",
        "surrogate",
    )
    _band(axes[2], tw, np.exp(log_R.cpu().numpy()), "#2E86AB", "basic R_t (posterior)")
    axes[2].axhline(1, color="grey", ls=":")
    axes[2].set(xlabel="week", ylabel="R_t", yscale="log", title="posterior basic R_t")
    axes[2].legend()
    fig.suptitle("Posterior predictive: surrogate")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {
        "cases": _coverage(pp[:, :, 0], obs_cases),
        "deaths": _coverage(pp[:, :, 1], obs_deaths),
    }


def ppc_simulator(log_R, statics, obs_cases, obs_deaths, n_weeks, path, seed=0):
    """Posterior predictive through the simulator. Returns 95%-band coverage."""
    theta = {"log_R": log_R.cpu().numpy()} | {
        k: v.cpu().numpy() for k, v in statics.items()
    }
    sim = simulate_covid(theta, n_weeks, seed=seed)
    tw = np.arange(n_weeks)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _counts_panels(
        axes[:2],
        tw,
        sim["cases"],
        sim["deaths"],
        obs_cases,
        obs_deaths,
        "#C73E1D",
        "simulator",
    )
    _band(axes[2], tw, sim["R_eff_t"], "#C73E1D", "effective R_t")
    axes[2].axhline(1, color="grey", ls=":")
    axes[2].set(
        xlabel="week",
        ylabel="R_t",
        yscale="log",
        title="posterior effective R_t (simulator)",
    )
    axes[2].legend()
    fig.suptitle("Posterior predictive: simulator")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {
        "cases": _coverage(sim["cases"], obs_cases),
        "deaths": _coverage(sim["deaths"], obs_deaths),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "out_dir", type=Path, help="output directory; a date-time suffix is appended"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="existing metis-format parquet: skips simulating the training set",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="trained checkpoint (.pt): skips simulation and training, goes straight to inference",
    )
    p.add_argument(
        "--surrogate-dir",
        type=Path,
        default=None,
        help="alias for --checkpoint DIR/checkpoints/last.pt",
    )
    p.add_argument(
        "--combine",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        default=None,
        help="instead of sampling, stack the chains of these finished runs (same checkpoint, window, "
        "IFR and parametrisation) and run the summary and predictive checks on the pooled draws; "
        "needs --checkpoint",
    )
    p.add_argument(
        "--who-csv", type=Path, default=Path("data/WHO-COVID-19-global-data.csv")
    )
    p.add_argument("--start", default="2020-03-29")
    p.add_argument("--end", default="2022-07-31")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="do not torch.compile the surrogate likelihood",
    )
    p.add_argument(
        "--n-prior-pred",
        type=int,
        default=5000,
        help="prior draws in the prior predictive plot",
    )
    d = p.add_argument_group("dataset")
    d.add_argument("--n-params", type=int, default=20_000)
    d.add_argument("--n-reps", type=int, default=2)
    m = p.add_argument_group("model (itarflow_cross_attn)")
    m.add_argument("--channels", type=int, default=128)
    m.add_argument("--num-layers", type=int, default=4)
    m.add_argument("--num-blocks", type=int, default=4)
    m.add_argument("--num-heads", type=int, default=4)
    tr = p.add_argument_group("training")
    tr.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    tr.add_argument("--max-steps", type=int, default=10_000)
    tr.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="global training batch size (train.py default 128)",
    )
    tr.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPUs for training; train.py splits the global batch across them",
    )
    tr.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="validation interval in steps (default: equal to --max-steps, i.e. validate once at the end)",
    )
    tr.add_argument("--ckpt-every", type=int, default=5_000)
    tr.add_argument(
        "--subsample",
        type=int,
        default=5,
        help="validation parameter sets drawn in train.py's realisations plot (tutorial default)",
    )
    tr.add_argument(
        "--no-samples-plot",
        action="store_true",
        help="skip train.py's realisations plot",
    )
    mc = p.add_argument_group("MCMC (NUTS)")
    mc.add_argument("--max-tree-depth", type=int, default=6)
    mc.add_argument("--target-accept-prob", type=float, default=0.8)
    mc.add_argument("--warmup-samples", type=int, default=300)
    mc.add_argument("--num-samples", type=int, default=300)
    mc.add_argument(
        "--num-chains", type=int, default=1, help="chains are run one after another"
    )
    mc.add_argument(
        "--ifr",
        type=float,
        default=0.0025,
        help="fixed infection fatality ratio, reported deaths per infection (default 0.25%%)",
    )
    mc.add_argument(
        "--infer-ifr",
        action="store_true",
        help="infer the IFR from its prior instead of fixing it",
    )
    mc.add_argument(
        "--full-mass",
        action="store_true",
        help="adapt a dense mass matrix; captures the ascertainment/IFR/seeding/path ridge",
    )
    mc.add_argument(
        "--parametrisation",
        choices=["noncentred", "centred"],
        default="noncentred",
        help="R path as unit-normal increments (noncentred) or as the path itself (centred)",
    )
    mc.add_argument(
        "--n-ppc",
        type=int,
        default=200,
        help="posterior draws used for the predictive checks",
    )
    args = p.parse_args()
    if args.eval_every is None:
        args.eval_every = args.max_steps
    return args


def main():
    sys.stdout.reconfigure(
        line_buffering=True
    )  # keep our prints in order with train.py's output
    args = parse_args()
    run_dir = (
        args.out_dir.parent / f"{args.out_dir.name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )  # default=str handles Path values
    print(f"run directory: {run_dir}")

    obs_cases, obs_deaths = load_who(args.who_csv, args.start, args.end)
    n_weeks = len(obs_cases)
    prior_predictive(
        obs_cases,
        obs_deaths,
        n_weeks,
        run_dir / "prior_predictive.png",
        n_draws=args.n_prior_pred,
        seed=args.seed,
    )

    checkpoint = args.checkpoint or (
        args.surrogate_dir / "checkpoints/last.pt" if args.surrogate_dir else None
    )
    if checkpoint is None:
        if args.dataset is None:
            parquet, n_reps = simulate_dataset(run_dir, args, n_weeks), args.n_reps
        else:
            parquet, n_reps = args.dataset, check_dataset(args.dataset, n_weeks)
        config_dir = write_dataset_config(run_dir, parquet, n_weeks, n_reps)
        checkpoint = train_surrogate(run_dir, config_dir, args) / "checkpoints/last.pt"

    sur = Surrogate(
        checkpoint, args.device, compile_model=not args.no_compile, dataset=args.dataset
    )
    assert sur.T == n_weeks, (
        f"surrogate was trained on {sur.T} weeks, data has {n_weeks}"
    )
    y_obs = sur.normalise_obs(obs_cases, obs_deaths)

    if args.combine:
        samples = combine_chains(args.combine, run_dir / "mcmc", args)
    else:
        samples = run_mcmc(sur, y_obs, args, run_dir / "mcmc")
    samples = {k: v.to(sur.device) for k, v in samples.items()}

    log_R, statics = posterior_draws(
        samples, args.n_ppc, seed=args.seed, ifr=None if args.infer_ifr else args.ifr
    )
    coverage = {
        "surrogate": ppc_surrogate(
            sur,
            log_R,
            statics,
            obs_cases,
            obs_deaths,
            run_dir / "ppc_surrogate.png",
            seed=args.seed,
        ),
        "simulator": ppc_simulator(
            log_R,
            statics,
            obs_cases,
            obs_deaths,
            n_weeks,
            run_dir / "ppc_simulator.png",
            seed=args.seed,
        ),
    }
    (run_dir / "ppc_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(
        "share of observed weeks inside the 95% posterior predictive band:",
        json.dumps(coverage),
    )
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
