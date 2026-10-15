import torch
import torch.nn.functional as F
from models.mlp import ResidualMLP, ResidualMLPConfig


def test_forward_pass_shape(mlp_model, tiny_mlp_config):
    """Verifies that the model accepts input and produces output of the correct shape."""
    batch_size = 4
    x = torch.randn(batch_size, tiny_mlp_config.input_dim)
    y = mlp_model(x)
    assert y.shape == (batch_size, tiny_mlp_config.output_dim)


def test_masked_loss(mlp_model, tiny_mlp_config):
    """Masked-out samples should not contribute to the loss."""
    inputs = torch.concat(
        [
            torch.zeros((1, tiny_mlp_config.input_dim)),
            torch.ones((1, tiny_mlp_config.input_dim)),
        ]
    )
    targets = torch.concat(
        [
            torch.zeros((1, tiny_mlp_config.output_dim)),
            -1 * torch.ones((1, tiny_mlp_config.output_dim)),
        ]
    )
    mask = torch.tensor([1, 0])
    valid_output = mlp_model(inputs[:1, ...])
    valid_loss = F.mse_loss(valid_output, targets[:1, ...], reduction="mean")
    loss = mlp_model(inputs, targets=targets, mask=mask)
    assert loss.ndim == 0, "Loss not a scalar"
    assert torch.isclose(loss, valid_loss), "Loss not close enough to valid loss"


def test_gradient_flow(mlp_model, tiny_mlp_config):
    """Check that input_layer has gradients and that they're not all zero."""
    batch_size = 4
    x = torch.randn(batch_size, tiny_mlp_config.input_dim)
    targets = torch.randn(batch_size, tiny_mlp_config.output_dim)

    mlp_model.zero_grad()
    loss = mlp_model(x, targets=targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss not a scalar"

    loss.backward()

    assert mlp_model.input_layer.weight.grad is not None
    assert torch.any(mlp_model.input_layer.weight.grad != 0)


def test_zero_init_residual_identity(tiny_mlp_config):
    """At init, each ResidualBlock's last layer is zero so blocks should be identity."""
    m = ResidualMLP(tiny_mlp_config)
    m.eval()

    torch.manual_seed(2)
    x = torch.randn(2, tiny_mlp_config.input_dim)

    with torch.no_grad():
        actual = m(x)
        # Since residual block last-layer weights are zero-initialised,
        # each block is identity on the residual stream.
        expected = m.output_layer(m.input_layer(x))

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Zero-init blocks should be near-identity, max diff: {(actual - expected).abs().max()}"
    )


def test_single_batch_overfit():
    """A small model should memorize one batch easily."""
    torch.manual_seed(3)
    cfg = ResidualMLPConfig(
        input_dim=4,
        output_dim=8,
        num_blocks=3,
        hidden_dim=32,
        num_layers_per_block=2,
    )
    m = ResidualMLP(cfg)
    m.train()

    assert cfg.input_dim is not None and cfg.output_dim is not None
    x = torch.randn(8, cfg.input_dim)
    y = torch.randn(8, cfg.output_dim)

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)

    for _ in range(100):
        optimizer.zero_grad()
        loss = m(x, targets=y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 1e-3, f"Model failed to overfit single batch, loss={final_loss}"
