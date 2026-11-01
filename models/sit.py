# Linear Flow Matching with a non-causal transformer adapted to sequences.
# https://github.com/willisma/SiT
# https://github.com/LTH14/JiT/blob/main/model_jit.py
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.transformer import apply_rotary_emb, norm
from models.utils.cross_attention import (
    CrossAttention,
    CrossAttentionConfig,
)


@dataclass
class FlowMatchingTransformerConfig:
    _target_: str = "models.sit.create_flow_matching_transformer"
    num_steps: int | None = 50
    solver: str | None = "heun"
    multi_mode: str = "median"
    sequence_len: int = 256
    data_dim: int = 8
    cond_dim: int = 8
    n_blocks: int = 12
    n_head: int = 8  # number of query heads
    n_kv_head: int = 8  # number of key/value heads (GQA)
    qkv_bias: bool = True
    embd_dim: int = 64
    cond_dropout_prob: float = 0.0
    # Controls data/conditioning input projections; the final layer always has a bias.
    in_and_out_biases: bool = True
    cross_attn: CrossAttentionConfig | None = None


def create_flow_matching_transformer(**kwargs):
    if isinstance(kwargs.get("cross_attn"), Mapping):
        kwargs["cross_attn"] = CrossAttentionConfig(**kwargs["cross_attn"])
    return FlowMatchingTransformer(FlowMatchingTransformerConfig(**kwargs))


class Mlp(nn.Module):
    """Transformer feed-forward layer with a fixed expansion ratio of four."""

    def __init__(self, embd_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, 4 * embd_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(4 * embd_dim, embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply conditioning-dependent shift and scale."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# https://github.com/willisma/SiT/blob/main/models.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditioningEmbedder(nn.Module):
    """
    Embeds continuous vector conditioning into hidden representations.
    Handles conditioning dropout for classifier-free guidance.
    """

    def __init__(self, cond_dim: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_size)
        self.dropout_prob = dropout_prob
        # NOTE: Fixed for now
        self.null_embedding = nn.Parameter(
            torch.zeros(hidden_size), requires_grad=False
        )

    def forward(
        self,
        cond: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ):
        embeddings = self.proj(cond)  # (B, hidden_size)

        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_ids = (
                    torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
                )
            else:
                drop_ids = force_drop_ids == 1

            # Replace dropped embeddings with the fixed null embedding.
            embeddings = torch.where(
                drop_ids[:, None], self.null_embedding[None, :], embeddings
            )
        return embeddings


# Self-attention with RoPE, GQA support and QK-prenorm
class SelfAttention(nn.Module):
    def __init__(self, config: FlowMatchingTransformerConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.embd_dim = config.embd_dim
        self.head_dim = self.embd_dim // self.n_head

        # Ensure embedding dim is divisible by n_head
        assert self.embd_dim % self.n_head == 0
        # Ensure number of kv heads is correct, this allows us to do MHA or GQA
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(
            self.embd_dim, self.n_head * self.head_dim, bias=config.qkv_bias
        )
        self.c_k = nn.Linear(
            self.embd_dim, self.n_kv_head * self.head_dim, bias=config.qkv_bias
        )
        self.c_v = nn.Linear(
            self.embd_dim, self.n_kv_head * self.head_dim, bias=config.qkv_bias
        )
        # Residual stream projection
        # TODO: Configurable. I think JiT uses bias
        self.c_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        enable_gqa = q.size(1) != k.size(1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        )
        # Transpose back to (B, T, H, D) and merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, config: FlowMatchingTransformerConfig, layer_idx: int):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = Mlp(config.embd_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.embd_dim, 6 * config.embd_dim, bias=True)
        )

        self.has_cross_attn = config.cross_attn is not None
        if self.has_cross_attn:
            # Patch the embedding dim through
            config.n_embd = config.embd_dim  # ty: ignore[unresolved-attribute]
            self.cross_attn = CrossAttention(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos_sin: torch.Tensor,
        cond_seq: torch.Tensor | None = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(norm(x), shift_msa, scale_msa), cos_sin
        )
        if self.has_cross_attn:
            x = x + self.cross_attn(norm(x), cond_seq, cos_sin, cos_sin, None, None)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(norm(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, config: FlowMatchingTransformerConfig):
        super().__init__()
        self.linear = nn.Linear(config.embd_dim, config.data_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.embd_dim, 2 * config.embd_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(norm(x), shift, scale)
        x = self.linear(x)
        return x


class FlowMatchingTransformer(nn.Module):
    """Learn a noise-to-data velocity field along a fixed linear path."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        config: FlowMatchingTransformerConfig,
    ):
        super().__init__()
        self.data_dim = config.data_dim
        self.cond_dim = config.cond_dim
        self.n_head = config.n_head
        self.embd_dim = config.embd_dim
        self.head_dim = self.embd_dim // self.n_head
        self.sequence_len = config.sequence_len
        self.cond_dropout_prob = config.cond_dropout_prob

        self.input_projection = nn.Linear(
            self.data_dim,
            self.embd_dim,
            bias=config.in_and_out_biases,
        )

        self.t_embedder = TimestepEmbedder(self.embd_dim)
        self.cond_embedder = ConditioningEmbedder(
            self.cond_dim, self.embd_dim, self.cond_dropout_prob
        )
        self.has_cross_attn = config.cross_attn is not None
        if config.cross_attn is not None:
            self.cond_sequence_projection = nn.Linear(
                config.cross_attn.cond_dim, self.embd_dim, bias=config.in_and_out_biases
            )

        # RoPE for positional embedding
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.head_dim
        )
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            [SiTBlock(config, layer_idx) for layer_idx in range(config.n_blocks)]
        )

        self.final_layer = FinalLayer(config)
        self.initialize_weights()

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000, device=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.input_projection.weight.device

        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.has_cross_attn:
            torch.nn.init.normal_(
                self.cond_sequence_projection.weight, mean=0.0, std=1.0
            )
            if self.cond_sequence_projection.bias is not None:
                torch.nn.init.zeros_(self.cond_sequence_projection.bias)

        # Initialise conditioning embedding MLP:
        assert isinstance(self.cond_embedder.proj, nn.Linear)
        nn.init.normal_(self.cond_embedder.proj.weight, std=0.02)

        # Initialize timestep embedding MLP:
        t_mlp_0 = self.t_embedder.mlp[0]
        assert isinstance(t_mlp_0, nn.Linear)
        nn.init.normal_(t_mlp_0.weight, std=0.02)
        t_mlp_2 = self.t_embedder.mlp[2]
        assert isinstance(t_mlp_2, nn.Linear)
        nn.init.normal_(t_mlp_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            assert isinstance(block, SiTBlock)
            block_mod = block.adaLN_modulation[-1]
            assert isinstance(block_mod, nn.Linear)
            nn.init.constant_(block_mod.weight, 0)
            assert block_mod.bias is not None
            nn.init.constant_(block_mod.bias, 0)

        # Zero-out output layers:
        final_mod = self.final_layer.adaLN_modulation[-1]
        assert isinstance(final_mod, nn.Linear)
        nn.init.constant_(final_mod.weight, 0)
        assert final_mod.bias is not None
        nn.init.constant_(final_mod.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cond_seq: torch.Tensor | None = None,
    ):
        """
        Predict the velocity along the noise-to-data path.
        x: (B, T, data_dim) tensor of interpolated sequence data
        t: (B,) tensor of flow times, from zero (noise) to one (data)
        y: (B, cond_dim) tensor of conditioning inputs
        """
        B, T, _ = x.size()
        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert x.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {x.device} != {self.cos.device}"
        )
        if self.has_cross_attn:
            assert cond_seq is not None
            B_cond_seq, T_cond_seq, _ = cond_seq.size()
            assert B_cond_seq == B, (
                f"Conditioning sequence batch size ({B_cond_seq}) does not match x batch size ({B})"
            )
            assert T_cond_seq == self.sequence_len, (
                "Currently only full sequence length conditioning is supported"
            )
            cond_seq = self.cond_sequence_projection(cond_seq)
        else:
            cond_seq = None

        cos_sin = (
            self.cos[:, :T],
            self.sin[:, :T],
        )
        x = self.input_projection(x)
        t = self.t_embedder(t)  # (N, D)
        y = self.cond_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        x = norm(x)
        for block in self.blocks:
            x = block(x, c, cos_sin, cond_seq=cond_seq)

        # NOTE: final layer contains norm
        x = self.final_layer(x, c)
        return x

    def setup_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),  # Standard AdamW
    ) -> torch.optim.Optimizer:
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate into decay and no_decay
        # Afaik this is so we don't use decay biases
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if available
        use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames

        optimiser = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )
        print(f"Using {'fused' if use_fused else ''} AdamW")
        return optimiser

    def get_device(self):
        return self.input_projection.weight.device

    @torch.no_grad()
    def sample(
        self,
        x_0: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 50,
        method: str = "euler",
        cond_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate from noise at t=0 to data at t=1."""
        if method not in ("euler", "heun"):
            raise ValueError("Solver method must be one of [euler, heun]")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")

        ts = torch.linspace(0, 1, num_steps + 1, device=self.get_device())
        x_t = x_0
        for i in range(num_steps):
            dt = ts[i + 1] - ts[i]
            velocity = self(x_t, ts[i][None], y, cond_seq=cond_seq)
            x_euler = x_t + velocity * dt
            # Preserve the original sampler: Heun uses Euler on the final step.
            if method == "heun" and i < num_steps - 1:
                correction = self(x_euler, ts[i + 1][None], y, cond_seq=cond_seq)
                x_t = x_t + (dt / 2) * (velocity + correction)
            else:
                x_t = x_euler
        return x_t

    def compute_loss(
        self,
        static_input: torch.Tensor,
        x: torch.Tensor | None,  # unused; shared training API across models
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        pred_and_obs_idx: tuple[list[int], list[int]] | None = None,
    ) -> torch.Tensor:
        if pred_and_obs_idx is not None:  # We are doing sequence conditioning
            assert self.has_cross_attn, (
                "Attempting to do training with sequence conditioning without a cross attention module"
            )
            pred_idx, obs_idx = pred_and_obs_idx
            # Condition on the entire observed sequence.
            conditioning_seq = targets[:, :, obs_idx]
            targets = targets[:, :, pred_idx]
        else:
            conditioning_seq = None

        # NOTE: Unfortunately we cannot compile compute_loss and pass in a
        # generator due to Torch's limitations so this
        # is the cleanest option, don't think it matters
        # too much
        x_0 = torch.randn_like(targets, device=self.get_device())
        t = torch.rand((targets.shape[0],), device=self.get_device())
        t_broadcast = t[:, None, None]
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * targets
        velocity = self(x_t, t, static_input, cond_seq=conditioning_seq)
        target_velocity = targets - x_0
        loss = F.mse_loss(velocity, target_velocity, reduction="none")
        if mask is not None:
            loss = loss * mask[:, None, None]
            return loss.sum() / (mask.sum() * (x_0.size(1) * x_0.size(2)))
        return loss.mean()
