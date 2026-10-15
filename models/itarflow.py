# Modified version of improved TarFlow (https://github.com/Lyy-iiis/BiFlow/tree/itarflow) with improvements
# inspired by https://github.com/Lyy-iiis/imeanflow
# Transformer-based autoregressive normalising flow

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.transformer import norm


@dataclass
class iTarFlowConfig:
    _target_: str = "models.itarflow.create_itarflow"
    num_timesteps: int = 32
    out_channels: int = 4
    channels: int = 384
    patch_size: int = 1
    num_layers: int = 8
    num_blocks: int = 8
    num_heads: int = 4
    static_in: int = 10
    num_static_tokens: int = 8
    mlp_ratio: float = 8 / 3
    weight_init_constant: float = 0.32
    # dtype: Any = torch.float32
    clip_range: float = 1.0


def create_itarflow(**kwargs):
    return ImprovedTARFlow(iTarFlowConfig(**kwargs))


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute real-valued cosine and sine RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (seq_len, dim/2)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def apply_rotary_pos_emb(x, rope_freqs):
    """Apply rotary position embeddings to x.

    Args:
        x: (B, num_heads, seq_len, head_dim) — real-valued
        rope_freqs: (seq_len, head_dim/2, 2) — cosine and sine values
    Returns:
        x with RoPE applied, same shape and dtype as input
    """
    orig_dtype = x.dtype
    x_float = x.float()
    cos, sin = rope_freqs.unbind(dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_even, x_odd = x_float[..., ::2], x_float[..., 1::2]
    x_out = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    ).flatten(start_dim=-2)
    return x_out.to(orig_dtype)


class PermutationFlip(nn.Module):
    def __init__(self, flip: bool = True):
        super().__init__()
        self.flip = flip

    def forward(self, x: torch.Tensor):
        if self.flip:
            return x.flip(dims=[1])
        return x.clone()


class iMFLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias: bool = True, init_constant: float = 1.0
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.normal_(
            self.linear.weight, 0.0, std=init_constant / math.sqrt(in_features)
        )
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class SwiGLUMLP(nn.Module):
    def __init__(self, in_features, hidden_features, weight_init_constant: float = 1.0):
        super().__init__()
        self.w1 = iMFLinear(
            in_features, hidden_features, bias=False, init_constant=weight_init_constant
        )
        self.w2 = iMFLinear(
            hidden_features,
            in_features,
            bias=False,
            init_constant=weight_init_constant,
        )
        self.w3 = iMFLinear(
            in_features, hidden_features, bias=False, init_constant=weight_init_constant
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StaticEmbedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            iMFLinear(input_dim, embedding_dim),
            nn.ReLU(),
            iMFLinear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ConditionalRoPEAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        weight_init_constant: float,
        dtype=torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads

        init_kwargs = dict(
            in_features=channels,
            out_features=channels,
            bias=False,
            init_constant=weight_init_constant,
        )
        self.q_proj = iMFLinear(**init_kwargs)
        self.k_proj = iMFLinear(**init_kwargs)
        self.v_proj = iMFLinear(**init_kwargs)
        self.out_proj = iMFLinear(**init_kwargs)

        self.q_norm = norm  # RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = norm  # RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        seq,
        rope_freqs,
        mask=None,
        k_cache=None,
        v_cache=None,
    ):
        num_heads = self.num_heads
        B, T_seq, C = seq.shape
        activation_dtype = seq.dtype

        q, k, v = self.q_proj(seq), self.k_proj(seq), self.v_proj(seq)

        q = q.reshape(B, T_seq, num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T_seq, num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T_seq, num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)  # self.q_norm(q.to(torch.float32)).to(activation_dtype)
        k = self.k_norm(k)  # self.k_norm(k.to(torch.float32)).to(activation_dtype)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        if k_cache is not None:
            idx = k_cache["idx"]
            k_cache["cache"][:, :, idx : idx + T_seq, :] = k
            v_cache["cache"][:, :, idx : idx + T_seq, :] = v
            k = k_cache["cache"][:, :, : idx + T_seq, :]
            v = v_cache["cache"][:, :, : idx + T_seq, :]
            attn_mask = None
        else:
            attn_mask = mask

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, T_seq, C)
        attn = self.out_proj(attn)
        return attn, k_cache, v_cache


class InContextLayer(nn.Module):
    def __init__(
        self,
        channels,
        dtype,
        num_heads,
        mlp_ratio,
        weight_init_constant,
        device: str = "cuda",
    ):
        super().__init__()
        self.norm1 = norm  # RMSNorm(channels, eps=1e-6)
        self.attn = ConditionalRoPEAttention(
            channels, num_heads, weight_init_constant, dtype=dtype, device=device
        )
        self.norm2 = norm  # RMSNorm(channels, eps=1e-6)
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(channels * mlp_ratio)
        self.weight_init_constant = weight_init_constant
        self.mlp = SwiGLUMLP(
            channels, self.mlp_hidden_dim, weight_init_constant=weight_init_constant
        )
        self.attn_scale = nn.Parameter(torch.zeros(channels))
        self.mlp_scale = nn.Parameter(torch.zeros(channels))

    def forward(
        self,
        x,
        rope_freqs,
        mask=None,
        k_cache=None,
        v_cache=None,
    ):
        attn, k_cache, v_cache = self.attn(
            self.norm1(x),
            rope_freqs,
            mask=mask,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        x = x + attn * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x, k_cache, v_cache


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_channels,
        dtype,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.norm = norm  # RMSNorm(hidden_dim, eps=1e-6)
        self.proj_out = nn.Linear(self.hidden_dim, 2 * self.out_channels, dtype=dtype)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        return self.proj_out(self.norm(x))


class InContextMetaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_heads,
        num_patches,
        num_layers,
        static_in,
        permutation,
        dtype,
        num_static_tokens: int = 8,
        mlp_ratio: int = 8 / 3,
        weight_init_constant: float = 0.32,
        clip_range: float = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.permutation = permutation
        self.static_in = static_in
        self.num_static_tokens = num_static_tokens
        self.mlp_ratio = mlp_ratio
        self.weight_init_constant = weight_init_constant
        self.clip_range = clip_range

        self.proj_in = nn.Linear(in_channels, channels, dtype=dtype)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

        self.static_embedder = StaticEmbedder(static_in, channels)

        std = 1 / math.sqrt(channels)
        self.static_tokens = nn.Parameter(
            torch.randn(self.num_static_tokens, channels) * std
        )

        self.prefix_tokens = self.num_static_tokens
        self.total_tokens = self.num_patches + self.prefix_tokens

        self.head_dim = self.channels // self.num_heads
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, self.total_tokens),
        )

        self.blocks = nn.ModuleList(
            [
                InContextLayer(
                    channels=channels,
                    dtype=dtype,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    weight_init_constant=weight_init_constant,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(channels, in_channels, dtype=dtype)

    def build_attn_mask(self, device):
        T_c = self.prefix_tokens
        T = self.num_patches
        total = T_c + T
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:T_c, :T_c] = True  # cond <-> cond
        mask[T_c:, :T_c] = True  # patches -> cond
        mask[T_c:, T_c:] = torch.tril(  # patches -> patches (causal)
            torch.ones(T, T, dtype=torch.bool, device=device)
        )
        return mask

    def _build_c_attn(self, label):
        static_embed = self.static_embedder(label)
        static_tokens = self.static_tokens.unsqueeze(0) + static_embed.unsqueeze(1)
        return static_tokens

    def forward(self, x, static):
        x_in = self.permutation(x)
        x = self.proj_in(x_in)
        c_attn = self._build_c_attn(static)
        seq = torch.cat([c_attn, x], dim=1)
        mask = self.build_attn_mask(x.device)
        for block in self.blocks:
            seq, _, _ = block(
                seq,
                self.rope_freqs,
                mask=mask,
            )
        seq = seq[:, self.prefix_tokens :]
        x = self.final_layer(seq)  # [B,T,2*C]
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        if self.clip_range is not None:
            x = torch.clamp(x, min=-self.clip_range, max=self.clip_range)
        alpha, mu = x.chunk(2, dim=-1)

        x_new = (x_in - mu) * torch.exp(-alpha)  # [B,T,C_in]
        log_jacob = -alpha.mean(axis=(1, 2))
        x_new = self.permutation(x_new)
        return x_new, log_jacob, alpha, mu

    def reverse(self, x, static):
        x_recon = self.permutation(x).clone()
        B, T, C = x_recon.shape
        T_c = self.prefix_tokens

        def _make_cache():
            return [
                {
                    "cache": torch.zeros(
                        B,
                        self.num_heads,
                        T_c + T,
                        self.head_dim,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    "idx": 0,
                }
                for _ in range(len(self.blocks))
            ]

        k_cache = _make_cache()
        v_cache = _make_cache()

        # Phase 1: Conditioning cache warmup
        cond_rope = self.rope_freqs[:T_c]
        c_attn = self._build_c_attn(static)
        for bi, block in enumerate(self.blocks):
            k_cache[bi]["idx"] = 0
            v_cache[bi]["idx"] = 0
            c_attn, k_cache[bi], v_cache[bi] = block(
                c_attn,
                cond_rope,
                k_cache=k_cache[bi],
                v_cache=v_cache[bi],
            )

        # Phase 2: Autoregressive patch decoding
        alphas = []
        for i in range(T - 1):
            x_step = self.proj_in(x_recon[:, i : i + 1, :])
            patch_rope = self.rope_freqs[T_c + i : T_c + i + 1]

            for bi, block in enumerate(self.blocks):
                k_cache[bi]["idx"] = T_c + i
                v_cache[bi]["idx"] = T_c + i
                x_step, k_cache[bi], v_cache[bi] = block(
                    x_step,
                    patch_rope,
                    k_cache=k_cache[bi],
                    v_cache=v_cache[bi],
                )

            out = self.final_layer(x_step)
            if self.clip_range is not None:
                out = torch.clamp(out, min=-self.clip_range, max=self.clip_range)

            alpha, mu = out.chunk(2, dim=-1)
            alphas.append(alpha)
            x_recon[:, i + 1 : i + 2, :] = (
                x_recon[:, i + 1 : i + 2, :] * torch.exp(alpha) + mu
            )

        # Reverse log-det: pad with zero for token 0 (untransformed)
        alphas_tensor = torch.cat([torch.zeros_like(alphas[0]), *alphas], dim=1)
        reverse_logdet = alphas_tensor.mean(dim=(1, 2))

        return self.permutation(x_recon), reverse_logdet, alphas_tensor


class ImprovedTARFlow(nn.Module):
    def __init__(
        self,
        config: iTarFlowConfig,
    ):
        super().__init__()

        self.config = config
        self.num_timesteps = config.num_timesteps
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        patch_num = self.num_timesteps // (self.patch_size)
        self.num_patches = patch_num
        self.in_channels = self.out_channels * self.patch_size
        assert self.patch_size * patch_num == self.num_timesteps, (
            "num_timesteps must be divisible by patch_size"
        )

        self.blocks = nn.ModuleList(
            [
                InContextMetaBlock(
                    in_channels=self.in_channels,
                    channels=self.config.channels,
                    num_heads=self.config.num_heads,
                    num_patches=self.num_patches,
                    num_layers=self.config.num_layers,
                    static_in=self.config.static_in,
                    permutation=PermutationFlip(i % 2 == 1),
                    dtype=torch.float32,  # self.config.dtype,
                    num_static_tokens=self.config.num_static_tokens,
                    mlp_ratio=self.config.mlp_ratio,
                    weight_init_constant=self.config.weight_init_constant,
                    clip_range=self.config.clip_range,
                )
                for i in range(self.config.num_blocks)
            ]
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.out_channels, (
            f"Input channels must match out_channels: expected {self.out_channels}, got {C}"
        )
        u = x.reshape(B, T // self.patch_size, C * self.patch_size)

        return u

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, patched_T, patched_C = x.shape

        u = x.reshape(B, patched_T * self.patch_size, patched_C // self.patch_size)
        return u

    def forward(self, x: torch.Tensor, static_conditioning: torch.Tensor):
        x = self.patchify(x)  # [B, T, C * patch_size^2]

        total_logdet = 0.0
        for block in self.blocks:
            x, logdet, alpha, mu = block(x, static_conditioning)
            total_logdet += logdet
        x = self.unpatchify(x)
        return x, total_logdet

    def reverse(self, x: torch.Tensor, static_conditioning: torch.Tensor):
        x = self.patchify(x)
        total_reverse_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for block in reversed(self.blocks):
            x, block_logdet, _ = block.reverse(x, static_conditioning)
            total_reverse_logdet += block_logdet
        x = self.unpatchify(x)
        return x, total_reverse_logdet

    def setup_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> torch.optim.Optimizer:
        param_dict = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        decay_params = [param for param in param_dict.values() if param.dim() >= 2]
        nodecay_params = [param for param in param_dict.values() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        print(f"Using {'fused' if use_fused else ''} AdamW")
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def compute_loss(
        self,
        static_input: torch.Tensor,
        x: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        pred_and_obs_idx: tuple[list[int], list[int]] | None = None,
    ) -> torch.Tensor:
        if pred_and_obs_idx:
            raise ValueError(
                "Improved TarFlow does not yet support sequence conditioning"
            )
        # NOTE: TarFlow models mean reduce their log-dets
        z, total_logdet = self.forward(targets, static_input)

        # Maximum Likelihood Estimation via instantaneous change of variables
        log_prior = 0.5 * (z**2)

        per_sample_loss = log_prior.mean(dim=(1, 2)) - total_logdet
        if mask is not None:
            return (per_sample_loss * mask).sum() / mask.sum()
        return per_sample_loss.mean()
