import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.utils.cross_attention import (
    CrossAttention,
    CrossAttnKVCache,
    CrossAttentionConfig,
    precompute_rotary_embeddings,
)
from models.transformer import norm
import pytest


@dataclass
class ToyModuleConfig:
    n_embd: int
    seq_len: int
    cross_attn: CrossAttentionConfig


class ToyModule(nn.Module):
    def __init__(self, cfg: ToyModuleConfig):
        super().__init__()
        self.x_in_proj = nn.Linear(1, cfg.n_embd, bias=True)
        self.con_in_proj = nn.Linear(1, cfg.n_embd, bias=True)
        self.mlp = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        torch.nn.init.zeros_(
            self.mlp.weight
        )  # Everything in the residual stream should start with weight 0

        self.cross_attn = CrossAttention(cfg, layer_idx=0)
        self.x_out_proj = nn.Linear(cfg.n_embd, 1, bias=True)

        self.head_dim = self.cross_attn.embd_dim // self.cross_attn.n_head
        cos, sin = precompute_rotary_embeddings(cfg.seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cond):
        B, T, C = x.size()
        x = self.x_in_proj(x)
        cond = self.con_in_proj(cond)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = x + self.cross_attn(norm(x), cond, cos_sin, cos_sin)
        x = x + self.mlp(norm(x))
        return self.x_out_proj(x)


def test_forward_pass_shape_same_length():
    batch_size = 8
    n_embd = 64
    seq_len = 32
    n_head = 4
    n_kv_head = 4
    head_dim = n_embd // n_head

    cross_attn = CrossAttention(
        config=ToyModuleConfig(
            n_embd=n_embd,
            seq_len=seq_len,  # Unused by cross attn module itself
            cross_attn=CrossAttentionConfig(
                n_head=n_head,
                cond_dim=1,  # Unused by cross-attn module itself
                n_kv_head=n_kv_head,
            ),
        ),
        layer_idx=0,
    )
    x_seq = torch.randn((batch_size, seq_len, n_embd))
    cond_seq = torch.randn((batch_size, seq_len, n_embd))
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim)
    output = cross_attn(
        x_seq, cond_seq, (cos, sin), (cos, sin), kv_cache=None, current_x_pos=None
    )
    assert output.shape == (batch_size, seq_len, n_embd)


def test_gqa_shape_same_length():
    torch.manual_seed(5)
    batch_size = 8
    n_embd = 64
    seq_len = 32
    n_head = 4
    n_kv_head = 2
    head_dim = n_embd // n_head
    cross_attn = CrossAttention(
        config=ToyModuleConfig(
            n_embd=n_embd,
            seq_len=seq_len,  # Unused by cross attn module itself
            cross_attn=CrossAttentionConfig(
                n_head=n_head,
                cond_dim=1,  # Unused by cross-attn module itself
                n_kv_head=n_kv_head,
            ),
        ),
        layer_idx=0,
    )
    x_seq = torch.randn((batch_size, seq_len, n_embd))
    cond_seq = torch.randn((batch_size, seq_len, n_embd))
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim)
    output = cross_attn(
        x_seq, cond_seq, (cos, sin), (cos, sin), kv_cache=None, current_x_pos=None
    )
    assert output.shape == (batch_size, seq_len, n_embd)


def test_causality_same_length():
    """Changing a future token must not affect past outputs"""
    torch.manual_seed(4)
    batch_size = 8
    n_embd = 64
    seq_len = 32
    n_head = 4
    n_kv_head = 4
    head_dim = n_embd // n_head

    cross_attn = CrossAttention(
        config=ToyModuleConfig(
            n_embd=n_embd,
            seq_len=seq_len,  # Unused by cross attn module itself
            cross_attn=CrossAttentionConfig(
                n_head=n_head,
                cond_dim=1,  # Unused by cross-attn module itself
                n_kv_head=n_kv_head,
            ),
        ),
        layer_idx=0,
    )

    # Set c_proj weights to non-zero so we don't just have an identity map
    with torch.no_grad():
        torch.nn.init.normal_(cross_attn.c_proj.weight, std=0.1)
    cross_attn.eval()

    x_seq = torch.randn((batch_size, seq_len, n_embd))
    cond_seq = torch.randn((batch_size, seq_len, n_embd))
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim)

    with torch.no_grad():
        out_original = cross_attn(
            x_seq, cond_seq, (cos, sin), (cos, sin), kv_cache=None, current_x_pos=None
        )
    # Peturb token at the midpoint
    t = seq_len // 2
    cond_perturbed = cond_seq.clone()
    cond_perturbed[:, t, :] += 10.0

    with torch.no_grad():
        out_perturbed = cross_attn(
            x_seq,
            cond_perturbed,
            (cos, sin),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )

    assert torch.equal(out_original[:, :t, :], out_perturbed[:, :t, :]), (
        "Outputs before the perturbed position differ — causality violated"
    )
    # The perturbed position (and after) should differ
    assert not torch.equal(out_original[:, t:, :], out_perturbed[:, t:, :]), (
        "Outputs at/after perturbation are identical — model ignores the perturbation"
    )


def test_casuality_smaller_x_seq():
    """Changing a future token must not affect past outputs"""
    torch.manual_seed(4)
    batch_size = 8
    n_embd = 64
    seq_len = 32
    x_seq_len = 4
    n_head = 4
    n_kv_head = 4
    head_dim = n_embd // n_head

    cross_attn = CrossAttention(
        config=ToyModuleConfig(
            n_embd=n_embd,
            seq_len=seq_len,  # Unused by cross attn module itself
            cross_attn=CrossAttentionConfig(
                n_head=n_head,
                cond_dim=1,  # Unused by cross-attn module itself
                n_kv_head=n_kv_head,
            ),
        ),
        layer_idx=0,
    )

    # Set c_proj weights to non-zero so we don't just have an identity map
    with torch.no_grad():
        torch.nn.init.normal_(cross_attn.c_proj.weight, std=0.1)
    cross_attn.eval()

    x_seq = torch.randn((batch_size, x_seq_len, n_embd))
    cond_seq = torch.randn((batch_size, seq_len, n_embd))
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim)
    with torch.no_grad():
        out_original = cross_attn(
            x_seq,
            cond_seq,
            (cos[:, :x_seq_len], sin[:, :x_seq_len]),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )

    out_original.shape == (batch_size, x_seq_len, n_embd), "Output shape is incorrect"

    # Peturb token at a point beyond the x-seq's last pos
    t = x_seq_len
    cond_perturbed = cond_seq.clone()
    cond_perturbed[:, t, :] += 10.0

    with torch.no_grad():
        out_perturbed = cross_attn(
            x_seq,
            cond_perturbed,
            (cos[:, :x_seq_len], sin[:, :x_seq_len]),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )

    assert torch.equal(out_original, out_perturbed), (
        "Outputs before the perturbed position differ — causality violated"
    )

    # Peturb token within x-seq's length
    t = 0
    cond_perturbed = cond_seq.clone()
    cond_perturbed[:, t, :] += 10.0

    with torch.no_grad():
        out_perturbed = cross_attn(
            x_seq,
            cond_perturbed,
            (cos[:, :x_seq_len], sin[:, :x_seq_len]),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )

    assert not torch.equal(out_original[:, t:], out_perturbed[:, t:]), (
        "Outputs at/after perturbation are identical — model ignores the perturbation"
    )


def test_kv_cache_matches_forward():
    torch.manual_seed(4)
    batch_size = 8
    n_embd = 64
    seq_len = 32
    n_head = 4
    n_kv_head = 4
    head_dim = n_embd // n_head

    cross_attn = CrossAttention(
        config=ToyModuleConfig(
            n_embd=n_embd,
            seq_len=seq_len,  # Unused by cross attn module itself
            cross_attn=CrossAttentionConfig(
                n_head=n_head,
                cond_dim=1,  # Unused by cross-attn module itself
                n_kv_head=n_kv_head,
            ),
        ),
        layer_idx=0,
    )

    # Set c_proj weights to non-zero so we don't just have an identity map
    with torch.no_grad():
        torch.nn.init.normal_(cross_attn.c_proj.weight, std=0.1)
    cross_attn.eval()

    starting_x = torch.randn((batch_size, 1, n_embd))
    cond_seq = torch.randn((batch_size, seq_len, n_embd))
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim)

    v1_uncached_x = starting_x.clone()

    #############################################################################
    # Testing autoregressive generation                                         #
    #############################################################################

    #### Uncached cond_seq and x ####
    for i in range(2):
        new_x = cross_attn(
            v1_uncached_x,
            cond_seq,
            (cos[:, i : i + 1], sin[:, i : i + 1]),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )[:, -1, :].unsqueeze(1)
        v1_uncached_x = torch.cat([v1_uncached_x, new_x], dim=1)

    #### Cached cond seq and cached x ####
    cross_kv_cache = CrossAttnKVCache(
        batch_size, n_head, seq_len, head_dim, 1, device="cpu", dtype=torch.float32
    )
    v2_x = starting_x.clone()

    # First pass we build the CrossAttnKVCache
    new_x = cross_attn(
        v2_x,
        cond_seq,
        (cos[:, 0:1], sin[:, 0:1]),
        (cos, sin),
        kv_cache=cross_kv_cache,
        current_x_pos=0,
    )[:, -1, :].unsqueeze(1)
    v2_x = torch.cat([v2_x, new_x], dim=1)

    # Now that we have cache no need to pass cond_seq in or its freqs
    new_x = cross_attn(
        new_x,
        None,
        (cos[:, 1:2], sin[:, 1:2]),
        None,
        kv_cache=cross_kv_cache,
        current_x_pos=1,
    )[:, -1, :].unsqueeze(1)
    v2_x = torch.cat([v2_x, new_x], dim=1)

    # Won't necessarily be bitwise the same but very close

    (
        torch.testing.assert_close(
            v1_uncached_x,
            v2_x,
            rtol=1e-5,
            atol=1e-6,
            msg="Uncached output does not match cached output within tolerance for autoregressive generation.",
        ),
    )

    #############################################################################
    # Testing for flow-matching style sampling                                  #
    #############################################################################

    cross_kv_cache = CrossAttnKVCache(
        batch_size, n_head, seq_len, head_dim, 1, device="cpu", dtype=torch.float32
    )
    x = torch.randn((batch_size, seq_len, n_embd))
    #### No conditioning cache ####

    v1_x = x.clone()
    for i in range(2):
        v1_x = cross_attn(
            v1_x,
            cond_seq,
            (cos, sin),
            (cos, sin),
            kv_cache=None,
            current_x_pos=None,
        )
    ### Use conditioning cache ###
    v2_x = x.clone()
    # This will fill kv-cache
    v2_x = cross_attn(
        v2_x,
        cond_seq,
        (cos, sin),
        (cos, sin),
        kv_cache=cross_kv_cache,
        current_x_pos=seq_len - 1,
    )
    # Now we rely on the cache
    v2_x = cross_attn(
        v2_x,
        None,
        (cos, sin),
        None,
        kv_cache=cross_kv_cache,
        current_x_pos=seq_len - 1,
    )
    torch.testing.assert_close(
        v1_x,
        v2_x,
        rtol=1e-5,
        atol=1e-6,
        msg="Uncached output does not match cached output within tolerance for fm-style generation.",
    )


def test_c_proj_gradients():
    torch.manual_seed(4)
    batch_size = 8
    n_embd = 64
    seq_len = 32

    cfg = ToyModuleConfig(
        n_embd=n_embd,
        seq_len=seq_len,
        cross_attn=CrossAttentionConfig(n_head=4, n_kv_head=4),
    )
    model = ToyModule(cfg)
    x_seq = torch.randn((batch_size, seq_len, 1))
    cond_seq = torch.randn((batch_size, seq_len, 1))

    model.zero_grad()
    pred = model(x_seq, cond_seq)

    loss = F.mse_loss(pred, torch.ones_like(x_seq))
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss not a scalar"

    loss.backward()

    assert model.cross_attn.c_proj.weight.grad is not None
    assert torch.any(model.cross_attn.c_proj.weight.grad != 0)


@pytest.mark.slow
def test_cross_attention_learns_2z():
    """Cross-attention toy model should learn to output 2*z from conditioning."""
    torch.manual_seed(42)

    seq_len = 32
    n_embd = 64
    batch_size = 256
    n_steps = 100
    lr = 5e-3

    cfg = ToyModuleConfig(
        n_embd=n_embd,
        seq_len=seq_len,
        cross_attn=CrossAttentionConfig(n_head=4, n_kv_head=4),
    )
    model = ToyModule(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(n_steps):
        x = torch.randn(batch_size, seq_len, 1)
        z = torch.randn(batch_size, seq_len, 1)
        target = 2.0 * z

        pred = model(x, z)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()  # type: ignore
    assert final_loss <= 0.3, (
        f"Cross-attention failed to learn 2z, final loss={final_loss:.4f}"
    )
