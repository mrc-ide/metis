from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: int = 10000, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    # (1, seq_len, 1, head_dim//2) for broadcasting with (B, T, H, D)
    return cos[None, :, None, :], sin[None, :, None, :]


# TODO: How would we incorporate classifier-free guidance
@dataclass
class CrossAttentionConfig:
    n_head: int = 8
    cond_dim: int = 1  # will be inferred from data
    n_kv_head: int = 8
    qkv_bias: bool = True


class CrossAttnKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # We fill the entire cross-attn kv cache once as we hit a layer for the first time
        self.populated = [False] * num_layers

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


# Cross-attention with RoPE, GQA support, QK-prenorm and KV cache
class CrossAttention(nn.Module):
    def __init__(
        self,
        config,  # CrossAttentionConfig, or a model config for existing callers
        layer_idx: int,
        *,
        n_embd: int | None = None,
    ):
        super().__init__()
        if isinstance(config, CrossAttentionConfig):
            if n_embd is None:
                raise ValueError("n_embd is required with CrossAttentionConfig")
            attention_config = config
        else:
            attention_config = config.cross_attn
        self.n_head = attention_config.n_head
        self.n_kv_head = attention_config.n_kv_head
        self.embd_dim = config.n_embd if n_embd is None else n_embd
        self.head_dim = self.embd_dim // self.n_head
        self.layer_idx = layer_idx

        # Ensure embedding dim is divisible by n_head
        assert self.embd_dim % self.n_head == 0
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"
        # Ensure number of kv heads is correct, this allows us to do MHA or GQA
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(
            self.embd_dim, self.n_head * self.head_dim, bias=attention_config.qkv_bias
        )
        self.c_k = nn.Linear(
            self.embd_dim,
            self.n_kv_head * self.head_dim,
            bias=attention_config.qkv_bias,
        )
        self.c_v = nn.Linear(
            self.embd_dim,
            self.n_kv_head * self.head_dim,
            bias=attention_config.qkv_bias,
        )
        # TODO: Do we want bias=False?
        self.c_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        # Zero-init output projection so cross-attention starts as a no-op
        nn.init.zeros_(self.c_proj.weight)

    def prepare_kv(self, cond, cos_sin_cond, kv_cache: CrossAttnKVCache):
        """Populate fixed observed K/V independently of the first query position."""
        k, v = self._project_kv(cond, cos_sin_cond)
        kv_cache.k_cache[self.layer_idx] = k
        kv_cache.v_cache[self.layer_idx] = v
        kv_cache.populated[self.layer_idx] = True

    def _project_kv(self, cond, cos_sin_cond):
        batch, length, _ = cond.shape
        k = self.c_k(cond).view(batch, length, self.n_kv_head, self.head_dim)
        v = self.c_v(cond).view(batch, length, self.n_kv_head, self.head_dim)
        return norm(apply_rotary_emb(k, *cos_sin_cond)), v

    # TODO: Support sparse conditioning sequences
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos_sin_x: torch.Tensor,
        cos_sin_cond: torch.Tensor | None = None,
        kv_cache: CrossAttnKVCache | None = None,
        current_x_pos: int | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()

        if cond is not None:
            assert B == cond.shape[0], (
                "Batches between x-sequence and the conditioning sequence must match"
            )

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

        # Shared RoPE so that conditioning and x tokens share
        # the same positional representation
        # e.g. a vector at timestep t=2 in the cond sequence should have
        # the same position as a vector at timestep t=2 in the x sequence
        q = apply_rotary_emb(q, *cos_sin_x)
        q = norm(q)

        if kv_cache is None or not kv_cache.populated[self.layer_idx]:
            assert cos_sin_cond is not None, (
                "Need cos_sin for conditioning sequence when kv cache is none or unpopulated"
            )
            k, v = self._project_kv(cond, cos_sin_cond)
            use_causual = True  # Ensure that do not allow x to attend to the future

            if (
                kv_cache is not None
            ):  # We only need to cache once and then we can just read
                kv_cache.k_cache[self.layer_idx] = k
                kv_cache.v_cache[self.layer_idx] = v
                kv_cache.populated[self.layer_idx] = True
        else:
            assert current_x_pos is not None, (
                "Need current x pos when using cross-attention KV-cache to correctly slice."
            )
            k, v = kv_cache.get_layer_cache(self.layer_idx)

            # Slice up to and including current x_pos
            k = k[:, : current_x_pos + 1]
            v = v[:, : current_x_pos + 1]

            # T==1 (autoregressive): slicing alone enforces causality, so a
            # causal mask of T_q=1 over T_k>1 would incorrectly drop all but K_0.
            # T>1 (parallel/FM-style): we still need the causal mask so each
            # query Q_i only attends to K_0..K_i.
            use_causual = T > 1

        # (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        enable_gqa = self.n_head != self.n_kv_head

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=use_causual, enable_gqa=enable_gqa
        )

        # Same length as x
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
