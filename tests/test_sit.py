import torch
from models.sit import SiT, SiTConfig, InterpolantTransformer, LinearInterpolant
from models.utils.cross_attention import CrossAttention, CrossAttentionConfig


def make_tiny_config():
    return SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.1,
        in_and_out_biases=True,
    )


def test_forward_pass_shape():
    """SiT output shape should match (B, T, input_dim)."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    out = model(x, t, y)
    assert out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {out.shape}"
    )


def test_zero_init_output():
    """Final layer is zero-initialised, so SiT should output near-zero at init."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    model.eval()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        out = model(x, t, y)
    assert out.abs().max() < 1e-5, (
        f"Expected near-zero output at init, got max {out.abs().max()}"
    )


def test_gradient_flow():
    """Backprop through SiT should produce non-zero gradients.

    At init, adaLN zero-init means gates are zero, so gradients don't flow
    back through the blocks to input_projection. We check final_layer gets
    gradients at init, then verify input_projection gets gradients after one
    optimizer step (once gates are non-zero).
    """
    cfg = make_tiny_config()
    model = SiT(cfg)
    model.train()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)

    # At init: final layer should get gradients even with zero-init
    model.zero_grad()
    out = model(x, t, y)
    loss = out.sum()
    loss.backward()
    assert model.final_layer.linear.weight.grad is not None
    assert torch.any(model.final_layer.linear.weight.grad != 0), (
        "final_layer should get non-zero gradients at init"
    )

    # After one optimizer step, gates open up and input_projection should get gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.step()
    model.zero_grad()
    out = model(x, t, y)
    out.sum().backward()
    assert model.input_projection.weight.grad is not None
    assert torch.any(model.input_projection.weight.grad != 0), (
        "input_projection should get non-zero gradients after one step"
    )


def test_conditioning_dropout():
    """With force_drop_ids=1, conditioning should be the null embedding (zeros)."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    B = 3
    cond = torch.randn(B, cfg.cond_dim)

    # Force all dropped
    force_drop = torch.ones(B)
    dropped = model.cond_embedder(cond, train=False, force_drop_ids=force_drop)
    assert torch.allclose(dropped, torch.zeros_like(dropped)), (
        "Dropped embeddings should be zero"
    )

    # Force none dropped
    force_keep = torch.zeros(B)
    kept = model.cond_embedder(cond, train=False, force_drop_ids=force_keep)
    assert not torch.allclose(kept, torch.zeros_like(kept)), (
        "Kept embeddings should be non-zero"
    )


def test_flow_matching_loss_finite_and_differentiable():
    """calc_loss should return a finite scalar with gradients."""
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()
    B, T = 4, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)

    model.zero_grad()

    loss = model.compute_loss(static_input=y, x=None, targets=x_1)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"


def test_sampling_shape():
    """sampling should produce output with the same shape as input."""
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.eval()
    B, T = 2, cfg.sequence_len
    x_0 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        x_out, x_in = model.sample(x_0, y, num_steps=5, method="euler")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )

    with torch.no_grad():
        x_out, x_in = model.sample(x_0, y, num_steps=5, method="heun")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )


def test_single_batch_overfit():
    """FlowMatchingTransformer should memorise a single batch."""
    torch.manual_seed(42)

    cfg = SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
    )
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()

    assert (
        cfg.sequence_len is not None
        and cfg.data_dim is not None
        and cfg.cond_dim is not None
    )
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    I_t = model.interpolant.calc_It(x_0, x_1, t[:, None, None])
    dot_I_t = model.interpolant.calc_It_dot(x_0, x_1, t[:, None, None])

    # We don't have control of the rng when we use .compute_loss for a SiT
    # so we lift the loss function out

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(200):
        optimizer.zero_grad()
        v_t = model.v_t(I_t, t, y)
        loss = torch.nn.functional.mse_loss(v_t, dot_I_t)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 1e-3, f"Model failed to overfit single batch, loss={final_loss}"


def test_cross_attention_training():
    torch.manual_seed(42)
    seq_cond_dim = 2
    cfg = SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
        cross_attn=CrossAttentionConfig(n_head=4, cond_dim=seq_cond_dim, n_kv_head=4),
    )
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    cond_seq = torch.randn(B, cfg.sequence_len, seq_cond_dim)

    # Forward pass produces correct shape
    I_t = model.interpolant.calc_It(x_0, x_1, t[:, None, None])
    output = model.v_t(I_t, t, y, cond_seq=cond_seq)
    assert output.shape == (B, cfg.sequence_len, cfg.data_dim)

    # Loss computation works
    dot_I_t = model.interpolant.calc_It_dot(x_0, x_1, t[:, None, None])
    loss = torch.nn.functional.mse_loss(output, dot_I_t)
    assert loss.ndim == 0, "Loss not a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    # Backward pass produces gradients and some are non-zero
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"
