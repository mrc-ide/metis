import pytest
from models.transformer import Transformer, TransformerConfig
from models.mlp import ResidualMLP, ResidualMLPConfig
from models.itarflow import ImprovedTARFlow, iTarFlowConfig


@pytest.fixture
def tiny_transformer_config():
    """Returns a tiny config for fast testing."""
    return TransformerConfig(
        sequence_len=32,
        x_input_dim=4,
        static_input_dim=4,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
    )


@pytest.fixture
def transformer_model(tiny_transformer_config):
    """Returns an initialized model (weights loaded)."""
    m = Transformer(tiny_transformer_config)
    m.init_weights()
    return m


@pytest.fixture
def tiny_mlp_config():
    return ResidualMLPConfig(
        input_dim=6,
        output_dim=20,
        num_blocks=3,
        hidden_dim=32,
        num_layers_per_block=2,
    )


@pytest.fixture
def mlp_model(tiny_mlp_config):
    return ResidualMLP(tiny_mlp_config)


@pytest.fixture
def tiny_itarflow_config():
    """Returns a tiny iTarFlow config for fast testing."""
    return iTarFlowConfig(
        num_timesteps=8,
        out_channels=2,
        channels=16,
        patch_size=1,
        num_layers=2,
        num_blocks=2,
        num_heads=2,
        static_in=3,
        num_static_tokens=4,
    )


@pytest.fixture
def itarflow_model(tiny_itarflow_config):
    """Returns an iTarFlow model (CPU, float32)."""
    return ImprovedTARFlow(tiny_itarflow_config)
