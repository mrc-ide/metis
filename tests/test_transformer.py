import torch
import torch.nn.functional as F

from models.transformer import Block, Transformer, TransformerConfig
from models.utils.cross_attention import CrossAttentionConfig


def test_forward_pass_shape(transformer_model, tiny_transformer_config):
    """Verifies that the model accepts input and produces output of the correct shape."""
    batch_size = 2
    seq_len = tiny_transformer_config.sequence_len

    x = torch.randn(batch_size, seq_len - 1, tiny_transformer_config.x_input_dim)
    static_input = torch.randn(batch_size, tiny_transformer_config.static_input_dim)
    y = transformer_model(x, static_input)

    assert y.shape == (batch_size, seq_len, tiny_transformer_config.x_input_dim)


def test_masked_loss(transformer_model, tiny_transformer_config):
    seq_len = tiny_transformer_config.sequence_len

    inputs = torch.concat(
        [
            torch.zeros((1, seq_len - 1, tiny_transformer_config.x_input_dim)),
            torch.ones((1, seq_len - 1, tiny_transformer_config.x_input_dim)),
        ]
    )
    static_inputs = torch.concat(
        [
            torch.zeros((1, tiny_transformer_config.static_input_dim)),
            torch.ones((1, tiny_transformer_config.static_input_dim)),
        ]
    )
    targets = torch.concat(
        [
            torch.zeros((1, seq_len, tiny_transformer_config.x_input_dim)),
            -1 * torch.ones((1, seq_len, tiny_transformer_config.x_input_dim)),
        ]
    )
    mask = torch.tensor([1, 0])
    valid_output = transformer_model(inputs[:1, ...], static_inputs[:1, ...])
    valid_loss = F.mse_loss(valid_output, targets[:1, ...], reduction="mean")
    loss = transformer_model(inputs, static_inputs, targets=targets, mask=mask)
    assert loss.ndim == 0, "Loss not a scalar"
    assert torch.isclose(loss, valid_loss), "Loss not close enough to valid loss"


def test_input_proj_gradients(transformer_model, tiny_transformer_config):
    """Check that x_input_proj and static_input_proj have gradients and that they're not all zero for random inputs"""
    batch_size = 2
    seq_len = tiny_transformer_config.sequence_len

    x = torch.randn(batch_size, seq_len - 1, tiny_transformer_config.x_input_dim)
    static_inputs = torch.randn(batch_size, tiny_transformer_config.static_input_dim)
    targets = torch.randn(batch_size, seq_len, tiny_transformer_config.x_input_dim)

    transformer_model.zero_grad()
    loss = transformer_model(x, static_inputs, targets=targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss not a scalar"

    loss.backward()

    assert transformer_model.x_input_proj.weight.grad is not None
    assert torch.any(transformer_model.x_input_proj.weight.grad != 0)
    assert transformer_model.static_input_proj.weight.grad is not None
    assert torch.any(transformer_model.static_input_proj.weight.grad != 0)


def test_causality(transformer_model, tiny_transformer_config):
    """Changing a future token must not affect past outputs (autoregressive property)."""
    torch.manual_seed(0)
    seq_len = tiny_transformer_config.sequence_len
    x = torch.randn(1, seq_len - 1, tiny_transformer_config.x_input_dim)
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)

    # Set c_proj weights to non-zero so attention actually contributes
    with torch.no_grad():
        for block in transformer_model.h:
            assert isinstance(block, Block)  # This is for the typechecker
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.1)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.1)

    transformer_model.eval()
    with torch.no_grad():
        out_original = transformer_model(x, static_input).clone()

    # Perturb token at the midpoint
    t = seq_len // 2
    x_perturbed = x.clone()
    x_perturbed[:, t, :] += 10.0

    with torch.no_grad():
        out_perturbed = transformer_model(x_perturbed, static_input).clone()

    # Positions before the perturbation must be bitwise identical
    assert torch.equal(out_original[:, :t, :], out_perturbed[:, :t, :]), (
        "Outputs before the perturbed position differ — causality violated"
    )
    # The perturbed position (and after) should differ
    assert not torch.equal(out_original[:, t:, :], out_perturbed[:, t:, :]), (
        "Outputs at/after perturbation are identical — model ignores the perturbation"
    )


def test_sliding_window_restricts_attention():
    """Sliding window attention should restrict context to the window size.

    We check that after an S layer tokens have only influenced each other within the window.

    Since c_proj is zero at init (attention contributes nothing), we manually set c_proj
    to non-zero so the attention output actually propagates.
    """
    seq_len = 8
    # Short window = seq_len//2 = 16.
    window_cfg = TransformerConfig(
        sequence_len=seq_len,
        x_input_dim=4,
        static_input_dim=2,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="S",
    )
    m = Transformer(window_cfg)
    m.init_weights()

    # Window is token pos + seq_len//2 left context

    #   0 1 2 3 4 5 6 7
    # 0 x o o o o o o o
    # 1 x x o o o o o o
    # 2 x x x o o o o o
    # 3 x x x x o o o o
    # 4 x x x x x o o o
    # 5 o x x x x x o o
    # 6 o o x x x x x o
    # 7 o o o x x x x x

    # Remove the last block so that we only have the S block
    m.h = torch.nn.ModuleList([block for _, block in zip(range(1), m.h)])

    # Set c_proj weights to non-zero so attention actually contributes
    with torch.no_grad():
        for block in m.h:
            assert isinstance(block, Block)  # This is for the typechecker
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.1)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.1)
    m.eval()

    torch.manual_seed(1)
    x = torch.randn(1, seq_len - 1, window_cfg.x_input_dim)
    static_input = torch.randn(1, window_cfg.static_input_dim)

    with torch.no_grad():
        out_original = m(x, static_input).clone()

    # Perturb x pos 0 which is overall pos 1 since we prepend the static input
    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 100.0

    with torch.no_grad():
        out_perturbed = m(x_perturbed, static_input).clone()

    # Pos 3 can attend to pos 1 so it should be affected
    within_window_pos = 3
    assert not torch.equal(
        out_original[:, within_window_pos, :], out_perturbed[:, within_window_pos, :]
    ), "Position within sliding window should be affected by perturbation"

    # Pos 6 cannot attend to pos 1 so it should not be affected
    outside_window_pos = 6
    assert torch.equal(
        out_original[:, outside_window_pos, :], out_perturbed[:, outside_window_pos, :]
    ), "Position outside sliding window should not be affected by perturbation"

    # Model output shape is correct
    assert out_original.shape == (1, seq_len, window_cfg.x_input_dim)


def test_zero_init_residual_identity(tiny_transformer_config):
    """At init, c_proj and mlp.c_proj are zeros, so blocks should be near-identity."""
    from models.transformer import norm

    m = Transformer(tiny_transformer_config)
    m.init_weights()
    m.eval()

    torch.manual_seed(2)
    x = torch.randn(
        1, tiny_transformer_config.sequence_len - 1, tiny_transformer_config.x_input_dim
    )
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)

    with torch.no_grad():
        actual = m(x, static_input)
        # Manual path: output_proj(norm(norm(input_proj(x))))
        # Because residual projections are zero, each block is identity on the residual stream.
        h = torch.cat(
            [m.static_input_proj(static_input)[:, None, :], m.x_input_proj(x)], dim=1
        )
        h = norm(h)
        # Each block: x = x + attn(norm(x)) but attn c_proj is zero => attn returns 0
        #             x = x + mlp(norm(x)) but mlp c_proj is zero => mlp returns 0
        # So after all blocks, h is unchanged.
        h = norm(h)
        expected = m.output_proj(h)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Zero-init blocks should be near-identity, max diff: {(actual - expected).abs().max()}"
    )


def test_single_batch_overfit(tiny_transformer_config):
    """A small model should memorize one batch easily."""
    torch.manual_seed(3)
    cfg = TransformerConfig(
        sequence_len=32,
        x_input_dim=4,
        static_input_dim=5,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
    )
    m = Transformer(cfg)
    m.init_weights()
    m.train()

    # Fixed batch
    x = torch.randn(4, cfg.sequence_len - 1, cfg.x_input_dim)
    static_input = torch.randn(4, cfg.static_input_dim)
    y = torch.randn(4, cfg.sequence_len, cfg.x_input_dim)

    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-3)

    for _ in range(1000):
        optimizer.zero_grad()
        loss = m(x, static_input, targets=y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 1e-3, f"Model failed to overfit single batch, loss={final_loss}"


def test_generate_matches_forward(transformer_model, tiny_transformer_config):
    """Autoregressive generation should match the last-position output of a forward pass."""
    transformer_model.eval()
    torch.manual_seed(4)

    # Given only static input
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)
    with torch.no_grad():
        full_out = transformer_model(None, static_input)  # (1, 1, input_dim)
        forward_last = full_out[:, -1:, :]  # (1, 1, input_dim)

    # Generate 1 step
    gen_steps = list(
        transformer_model.generate(static_input.clone(), None, max_output_steps=1)
    )
    assert len(gen_steps) == 1
    generated = gen_steps[0]  # (1, 1, input_dim)

    assert torch.allclose(forward_last, generated, atol=1e-5), (
        f"Generate and forward mismatch (only static given), max diff: {(forward_last - generated).abs().max()}"
    )

    # Given static input and some x
    x = torch.randn(1, 4, tiny_transformer_config.x_input_dim)
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)

    with torch.no_grad():
        full_out = transformer_model(x, static_input)  # (1, 5, input_dim)
        forward_last = full_out[:, -1:, :]  # (1, 1, input_dim)

    # Generate 1 step
    gen_steps = list(
        transformer_model.generate(static_input.clone(), x.clone(), max_output_steps=1)
    )
    assert len(gen_steps) == 1
    generated = gen_steps[0]  # (1, 1, input_dim)

    assert torch.allclose(forward_last, generated, atol=1e-5), (
        f"Generate and forward mismatch (static and x given), max diff: {(forward_last - generated).abs().max()}"
    )


def test_kv_cache_generate_matches_naive_generate(
    transformer_model, tiny_transformer_config
):
    transformer_model.eval()
    torch.manual_seed(4)

    # Given only static
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)

    naive_out = torch.cat(
        list(transformer_model.generate(static_input, None, 2, use_kv_cache=False)),
        dim=1,
    )
    cached_out = torch.cat(
        list(transformer_model.generate(static_input, None, 2, use_kv_cache=True)),
        dim=1,
    )

    assert naive_out.shape == cached_out.shape, (
        f"Shape mismatch (only static given)! Naive ({naive_out.shape}) vs KV Cache ({cached_out.shape})"
    )
    max_diff = (naive_out - cached_out).abs().max().item()
    assert max_diff < 1e-4, (
        f"Outputs diverge (only static given)! max diff = {max_diff}"
    )

    # Given static and some x
    x = torch.randn(1, 4, tiny_transformer_config.x_input_dim)
    static_input = torch.randn(1, tiny_transformer_config.static_input_dim)

    naive_out = torch.cat(
        list(transformer_model.generate(static_input, x, 2, use_kv_cache=False)), dim=1
    )
    cached_out = torch.cat(
        list(transformer_model.generate(static_input, x, 2, use_kv_cache=True)), dim=1
    )

    assert naive_out.shape == cached_out.shape, (
        f"Shape mismatch (static and some x given)! Naive ({naive_out.shape}) vs KV Cache ({cached_out.shape})"
    )
    max_diff = (naive_out - cached_out).abs().max().item()
    assert max_diff < 1e-4, (
        f"Outputs diverge (static and some x given)! max diff = {max_diff}"
    )


def test_gqa_shapes_and_output():
    """Test grouped-query attention (n_kv_head < n_head)."""
    cfg = TransformerConfig(
        sequence_len=32,
        x_input_dim=4,
        static_input_dim=3,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
    )
    m = Transformer(cfg)
    m.init_weights()

    torch.manual_seed(5)

    x = torch.randn(2, cfg.sequence_len - 1, cfg.x_input_dim)
    static_input = torch.randn(2, cfg.static_input_dim)

    # Forward pass should work
    y = m(x, static_input)
    assert y.shape == (2, cfg.sequence_len, cfg.x_input_dim)

    # Backward pass should work
    targets = torch.randn_like(y)
    loss = m(x, static_input, targets=targets)
    loss.backward()

    # Verify gradients exist
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_cross_attention_cond_dim_larger_than_input_dim():
    """Cross-attention should work when cond_dim > input_dim."""
    x_input_dim = 2
    cond_dim = 6  # 3x larger than input_dim
    cfg = TransformerConfig(
        sequence_len=32,
        x_input_dim=x_input_dim,
        static_input_dim=2,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
        cross_attn=CrossAttentionConfig(n_head=2, cond_dim=cond_dim, n_kv_head=2),
    )
    m = Transformer(cfg)
    m.init_weights()

    torch.manual_seed(7)
    x = torch.randn(2, cfg.sequence_len - 1, x_input_dim)
    static_input = torch.randn(2, cfg.static_input_dim)

    # static input is prepended to cond
    cond = torch.randn(2, cfg.sequence_len - 1, cond_dim)

    # Forward pass produces correct shape
    y = m(x, static_input, cond=cond)
    assert y.shape == (2, cfg.sequence_len, x_input_dim)

    # Loss + backward
    targets = torch.randn_like(y)
    loss = m(x, static_input, targets=targets, cond=cond)
    loss.backward()

    # All params should have gradients allocated
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
    # c_proj is zero-init so only it gets non-zero grad at init (no gradient flows back through zeros)
    for block in m.h:
        assert isinstance(block, Block)
        assert torch.any(block.cross_attn.c_proj.weight.grad != 0), (
            "Cross-attention c_proj has zero gradients"
        )


def test_cross_attention_training():
    """Cross-attention conditioning: forward shape, loss, and gradients."""
    cond_dim = 3
    cfg = TransformerConfig(
        sequence_len=32,
        x_input_dim=4,
        static_input_dim=2,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
        cross_attn=CrossAttentionConfig(n_head=2, cond_dim=cond_dim, n_kv_head=2),
    )
    m = Transformer(cfg)
    m.init_weights()

    torch.manual_seed(6)
    x = torch.randn(2, cfg.sequence_len - 1, cfg.x_input_dim)
    static_input = torch.randn(2, cfg.static_input_dim)
    # Static input prepended to cond seq
    cond = torch.randn(2, cfg.sequence_len - 1, cond_dim)

    # Forward pass produces correct shape
    y = m(x, static_input, cond=cond)
    assert y.shape == (2, cfg.sequence_len, cfg.x_input_dim)

    # Loss computation works
    targets = torch.randn_like(y)
    loss = m(x, static_input, targets=targets, cond=cond)
    assert loss.ndim == 0, "Loss not a scalar"

    # Backward pass produces gradients on cross-attention parameters
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
    # c_proj is zero-init so only it gets non-zero grad at init (no gradient flows back through zeros)
    for block in m.h:
        assert isinstance(block, Block)
        assert torch.any(block.cross_attn.c_proj.weight.grad != 0), (
            "Cross-attention c_proj has zero gradients"
        )


def test_cross_attn_kv_cache_generate_matches_naive_generate():
    cond_dim = 3
    cfg = TransformerConfig(
        sequence_len=32,
        x_input_dim=4,
        static_input_dim=2,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
        cross_attn=CrossAttentionConfig(n_head=2, cond_dim=cond_dim, n_kv_head=2),
    )
    model = Transformer(cfg)
    model.init_weights()

    # Ensure the cross-attention projection is non-zero
    for block in model.h:
        torch.nn.init.kaiming_normal(block.cross_attn.c_proj.weight)
    model.eval()

    # Given only static and cond seq
    static_input = torch.randn(1, cfg.static_input_dim)
    # -1 since we will prepend ths static input
    conditioning_sequence = torch.randn(
        1, cfg.sequence_len - 1, cfg.cross_attn.cond_dim
    )

    naive_out = torch.cat(
        list(
            model.generate(
                static_input,
                None,
                2,
                use_kv_cache=False,
                conditioning_sequence=conditioning_sequence,
            )
        ),
        dim=1,
    )
    cached_out = torch.cat(
        list(
            model.generate(
                static_input,
                None,
                2,
                use_kv_cache=True,
                conditioning_sequence=conditioning_sequence,
            )
        ),
        dim=1,
    )

    assert naive_out.shape == cached_out.shape, (
        f"Shape mismatch (only static given)! Naive ({naive_out.shape}) vs KV Cache ({cached_out.shape})"
    )
    max_diff = (naive_out - cached_out).abs().max().item()
    assert max_diff < 1e-5, (
        f"Outputs diverge (only static given)! max diff = {max_diff}"
    )

    # Given static and some x
    x = torch.randn(1, 4, cfg.x_input_dim)

    naive_out = torch.cat(
        list(
            model.generate(
                static_input,
                x,
                2,
                use_kv_cache=False,
                conditioning_sequence=conditioning_sequence,
            )
        ),
        dim=1,
    )
    cached_out = torch.cat(
        list(
            model.generate(
                static_input,
                x,
                2,
                use_kv_cache=True,
                conditioning_sequence=conditioning_sequence,
            )
        ),
        dim=1,
    )

    assert naive_out.shape == cached_out.shape, (
        f"Shape mismatch (static and some x given)! Naive ({naive_out.shape}) vs KV Cache ({cached_out.shape})"
    )
    max_diff = (naive_out - cached_out).abs().max().item()
    assert max_diff < 1e-5, (
        f"Outputs diverge (static and some x given)! max diff = {max_diff}"
    )
