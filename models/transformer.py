# Adapted from https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator
from models.utils.cross_attention import (
    CrossAttention,
    CrossAttnKVCache,
    CrossAttentionConfig,
)


@dataclass
class TransformerConfig:
    _target_: str = "models.transformer.create_autoregressive_transformer"
    sequence_len: int = 256
    x_input_dim: int = 8
    static_input_dim: int = 8
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    rope_base: int = 10000
    n_embd: int = 64
    in_and_out_biases: bool = True
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"  # "SSSL"
    cross_attn: CrossAttentionConfig | None = None


def create_autoregressive_transformer(**kwargs):
    return Transformer(TransformerConfig(**kwargs))


class KVCache:
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
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
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
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        # TODO: Shuld we also zero-out k and v for safety?

    def get_pos(self) -> int:
        """Get current position (assumes all batch elements at same position)."""
        return int(self.cache_seqlens[0].item())

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    enable_gqa: bool,
) -> torch.Tensor:
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        )

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


def sdpa_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int] = (-1, -1),
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    SPDA training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """

    if k_cache is not None:
        assert v_cache is not None and cache_seqlens is not None
        B, T_new, H, D = q.shape
        pos = cache_seqlens[0].item()  # Assume uniform pos across batch
        # Modify cache in-place
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v
        # Get full cache up to current position + new tokens
        k_full = k_cache[:, : pos + T_new, :]
        v_full = v_cache[:, : pos + T_new, :]
    else:
        k_full = k
        v_full = v

    q = q.transpose(1, 2)
    k_full = k_full.transpose(1, 2)
    v_full = v_full.transpose(1, 2)
    enable_gqa = q.size(1) != k_full.size(1)
    y = _sdpa_attention(q, k_full, v_full, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def norm(x: torch.Tensor) -> torch.Tensor:
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # (,)
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        # Ensure embedding dim is divisible by n_head
        assert self.n_embd % self.n_head == 0
        # Ensure number of kv heads is correct, this allows us to do MHA or GQA
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Residual stream projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout,
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = sdpa_attn_func(q, k, v, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = sdpa_attn_func(
                q,
                k,
                v,
                window_size=window_size,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=kv_cache.cache_seqlens,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.has_cross_attn = config.cross_attn is not None
        if self.has_cross_attn:
            self.cross_attn = CrossAttention(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin_x: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None = None,
        cross_kv_cache: CrossAttnKVCache | None = None,
        cond: torch.Tensor | None = None,
        cos_sin_cond: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.has_cross_attn:
            if cross_kv_cache is not None:
                # A populated cross-attn cache only exists during cached
                # generation, which always supplies the self-attention cache too.
                assert kv_cache is not None
                B, T, C = x.size()
                current_x_pos = (
                    kv_cache.get_pos() + T - 1
                )  # Current 0-indexed position of last x-token
            else:
                current_x_pos = None

        x = x + self.attn(norm(x), cos_sin_x, window_size, kv_cache)
        if self.has_cross_attn:
            x = x + self.cross_attn(
                norm(x), cond, cos_sin_x, cos_sin_cond, cross_kv_cache, current_x_pos
            )
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    # TODO: Should I put everything else here?
    # Need to declare here otherwise ty can't figure out their types with the register buffer stuff
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: TransformerConfig):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        self.x_input_proj = nn.Linear(
            config.x_input_dim,
            config.n_embd,
            bias=config.in_and_out_biases,
        )
        self.static_input_proj = nn.Linear(
            config.static_input_dim,
            config.n_embd,
            bias=config.in_and_out_biases,
        )

        if self.config.cross_attn is not None:
            self.cond_proj = nn.Linear(
                self.config.cross_attn.cond_dim,
                self.config.n_embd,
                bias=config.in_and_out_biases,
            )
        self.h = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.output_proj = nn.Linear(
            config.n_embd, config.x_input_dim, bias=config.in_and_out_biases
        )
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim, self.config.rope_base
        )
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        """
        Initialize the full model in this one function for maximum clarity.

        input_proj:    normal, std=1.0
        output_proj: normal, std=0.001
        for each block:
            attn.c_q:            uniform, std=1/sqrt(n_embd)
            attn.c_k:            uniform, std=1/sqrt(n_embd)
            attn.c_v:            uniform, std=1/sqrt(n_embd)
            attn.c_proj:         zeros
            mlp.c_fc:            uniform, std=1/sqrt(n_embd)
            mlp.c_proj:          zeros
        """
        # Input projections
        torch.nn.init.normal_(self.x_input_proj.weight, mean=0.0, std=1.0)
        if self.x_input_proj.bias is not None:
            torch.nn.init.zeros_(self.x_input_proj.bias)

        torch.nn.init.normal_(self.static_input_proj.weight, mean=0.0, std=1.0)
        if self.static_input_proj.bias is not None:
            torch.nn.init.zeros_(self.static_input_proj.bias)

        # conditioning projection
        if self.config.cross_attn is not None:
            torch.nn.init.normal_(self.cond_proj.weight, mean=0.0, std=1.0)
            if self.cond_proj.bias is not None:
                torch.nn.init.zeros_(self.cond_proj.bias)

        # Output projection
        torch.nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.001)
        if self.output_proj.bias is not None:
            torch.nn.init.zeros_(self.output_proj.bias)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = (
            3**0.5 * n_embd**-0.5
        )  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.h:
            assert isinstance(block, Block)
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)

            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            # Initialise residual projects to zeroes so we start with identity
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

            if block.has_cross_attn:
                torch.nn.init.uniform_(block.cross_attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.cross_attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.cross_attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(
                    block.cross_attn.c_proj.weight
                )  # Cross-attn proj already has zero init...

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 10000,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.x_input_proj.weight.device

        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()

        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config: TransformerConfig) -> list[tuple[int, int]]:
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.x_input_proj.weight.device

    # Inspired by https://github.com/karpathy/nanoGPT/blob/master/model.py
    # NanoChat uses Muon which we omit for simplicity
    def setup_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.8, 0.95),
    ) -> torch.optim.Optimizer:
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate into decay and no_decay
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

    def forward(
        self,
        x: torch.Tensor | None = None,
        static_input: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor
        | None = None,  # Allows us to handle padded batches so we can keep batch sizes constant
        cond: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        cross_kv_cache: CrossAttnKVCache | None = None,
    ) -> torch.Tensor:

        if static_input is not None:  # Can be none if we're using a kv-cache
            assert static_input.ndim == 2
            static_in = self.static_input_proj(static_input)
        else:
            assert x is not None, "Static input and x are both None."
            assert kv_cache is not None, (
                "Static input not supplied and KV-cache is None."
            )
            assert kv_cache.get_pos() > 0, (
                "Static input not supplied and KV-cache is empty"
            )

        if x is not None:  # Should only be none when generating from scratch
            assert x.ndim == 3
            x = self.x_input_proj(x)

            if static_input is not None:
                B, T, C = x.size()
                B_static, C_static = static_input.size()
                assert B_static == B, (
                    f"Static input batch {B_static} does not match x input batch {B}"
                )
                x = torch.cat(
                    [static_in[:, None, :], x], dim=1
                )  # Prepend in the time dimension
        else:  # Generating from scratch
            x = static_in[:, None, :]

        B, T, C = x.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert x.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {x.device} != {self.cos.device}"
        )

        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        if kv_cache is None:
            T0 = 0
        else:
            T0 = kv_cache.get_pos()
        cos_sin_x = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        if self.config.cross_attn is not None:
            if cross_kv_cache is None or not all(cross_kv_cache.populated):
                assert cond is not None
                assert static_in is not None
                cond_seq = self.cond_proj(cond)
                # Preprend onto conditioning sequence
                cond_seq = torch.cat([static_in[:, None, :], cond_seq], dim=1)
                T_COND = cond_seq.size(1)
                assert T_COND == self.config.sequence_len, (
                    "Currently only full sequence length conditioning is supported"
                )
                # Since we have prepended the static input vector, the rope positions line-up
                cos_sin_cond = (
                    self.cos[:, :T_COND],
                    self.sin[:, :T_COND],
                )

            else:  # we have a cross-attn kv-cache and it is fully populated
                assert kv_cache is not None  # We need a self-attention cache
                cond_seq = None
                cos_sin_cond = None

        else:
            assert cond is None
            cond_seq = None
            cos_sin_cond = None
        x = norm(x)
        for i, block in enumerate(self.h):
            x = block(
                x,
                cos_sin_x,
                self.window_sizes[i],
                kv_cache=kv_cache,
                cross_kv_cache=cross_kv_cache,
                cond=cond_seq,
                cos_sin_cond=cos_sin_cond,
            )
        x = norm(x)
        x = self.output_proj(x)
        if targets is not None:
            # training
            # (B,T, input_dim)
            loss = F.mse_loss(x, targets, reduction="none")
            if mask is not None:
                loss = loss * mask[:, None, None]
                return loss.sum() / (
                    mask.sum() * x.size(1) * x.size(2)
                )  # i.e. average over all unmasked entries

            return loss.mean()
        else:
            assert mask is None, (
                "Using mask without training targets is currently not supported"
            )
            return x

    def compute_loss(
        self,
        static_input: torch.Tensor,
        x: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        pred_and_obs_idx: tuple[list[int], list[int]] | None = None,
    ) -> torch.Tensor:
        if pred_and_obs_idx is not None:  # We are doing sequence conditioning
            assert self.config.cross_attn is not None, (
                "Attempting to do training with sequence conditioning without a cross attention module"
            )
            pred_idx, obs_idx = pred_and_obs_idx

            # x-sequence inputs and targets are only at the predicted idx
            x = x[:, :, pred_idx]

            # we causually condition on the entire sequence of observed idx
            conditioning_seq = targets[:, :, obs_idx]
            targets = targets[:, :, pred_idx]

        else:
            conditioning_seq = None
        return self.forward(
            x, static_input, targets=targets, mask=mask, cond=conditioning_seq
        )

    @torch.inference_mode()
    def generate(
        self,
        static_input: torch.Tensor,
        inputs: torch.Tensor | None = None,
        max_output_steps: int = 1,
        use_kv_cache: bool = False,
        conditioning_sequence: torch.Tensor | None = None,
    ) -> Iterator[torch.Tensor]:
        device = self.get_device()

        if inputs is not None:
            inputs = inputs.to(device)
        static_input = static_input.to(device)

        if self.config.cross_attn is not None:
            assert conditioning_sequence is not None

        if not use_kv_cache:
            for _ in range(max_output_steps):
                outputs = self.forward(
                    inputs, static_input, cond=conditioning_sequence
                )[:, -1, :].unsqueeze(1)  # (B,t,C) -> (B,1,C)
                if inputs is not None:
                    inputs = torch.cat(
                        (inputs, outputs), dim=1
                    )  # dim and axis are identical, dim has the typing ty needs
                else:
                    inputs = outputs
                yield outputs
        else:
            if inputs is not None:
                B, T_xin, C = inputs.size()
                T_sin = static_input.size(1)
                T_in = T_xin + T_sin
            else:
                T_in = 1
                B = static_input.size(0)
            kv_cache = KVCache(
                B,
                num_heads=self.config.n_kv_head,
                seq_len=T_in + max_output_steps,
                head_dim=self.config.n_embd // self.config.n_head,
                num_layers=self.config.n_layer,
                device=device,
                dtype=static_input.dtype,
            )
            if self.config.cross_attn is not None:
                cross_kv_cache = CrossAttnKVCache(
                    B,
                    num_heads=self.config.cross_attn.n_kv_head,
                    seq_len=self.config.sequence_len,
                    head_dim=self.config.n_embd // self.config.cross_attn.n_head,
                    num_layers=self.config.n_layer,
                    device=device,
                    dtype=static_input.dtype,
                )
            else:
                cross_kv_cache = None
            # Prefill static inputs, already supplied x inputs and conditioning sequences
            # Forward pass builds cache up to T_in and fills entire crossattncache
            x = self.forward(
                inputs,
                static_input,
                kv_cache=kv_cache,
                cond=conditioning_sequence,
                cross_kv_cache=cross_kv_cache,
            )
            next_input = x[:, -1:, :]  # first generated (B, 1, C)
            yield next_input

            # Instead of computing forward over a growing sequence and appending the last output, thereby wasting compute,
            # we only compute on the most recent token and iteratively build up the cache to save compute
            for _ in range(max_output_steps - 1):
                next_input = self.forward(
                    next_input, kv_cache=kv_cache, cross_kv_cache=cross_kv_cache
                )  # (B,1,C)
                yield next_input

    def estimate_flops(self):
        raise NotImplementedError()

    # We don't have embedding here
    def num_scaling_params(self) -> dict[str, int]:
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        input_layer = sum(p.numel() for p in self.x_input_proj.parameters()) + sum(
            p.numel() for p in self.static_input_proj.parameters()
        )

        output_layer = sum(p.numel() for p in self.output_proj.parameters())

        transformer_matrices = sum(p.numel() for p in self.h.parameters())
        total = input_layer + output_layer + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), (
            "Parameter count mismatch"
        )
        return {
            "input_layer": input_layer,
            "output_layer": output_layer,
            "transformer_matrices": transformer_matrices,
            "total": total,
        }
