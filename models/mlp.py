import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable


@dataclass
class ResidualMLPConfig:
    _target_: str = "models.mlp.create_residual_mlp"
    input_dim: int = -1
    output_dim: int = -1
    num_blocks: int = -1
    hidden_dim: int = -1
    num_layers_per_block: int = -1


def create_residual_mlp(**kwargs):
    return ResidualMLP(ResidualMLPConfig(**kwargs))


class OneLayer(nn.Module):
    def __init__(
        self,
        config: ResidualMLPConfig,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.act = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ResidualBlock(nn.Module):
    def __init__(self, config: ResidualMLPConfig):
        super().__init__()
        assert config.num_layers_per_block >= 2, (
            "Num layers per block needs to be at least 2"
        )
        self.layers = nn.ModuleList(
            [
                OneLayer(
                    config,
                    activation_fn=F.relu
                    if i < (config.num_layers_per_block) - 1
                    else lambda x: x,  # No activation on last layer in block
                )
                for i in range(config.num_layers_per_block)
            ]
        )
        self.norm = nn.Identity()
        # Initialise the residual projection to zeros so we start with identity
        assert isinstance(self.layers[-1], OneLayer)
        torch.nn.init.zeros_(self.layers[-1].linear.weight)
        # Typical initialisation is normal with var (1/sqrt(dim))
        # here we sample from a uniform distribution with the same
        # var, but the hard bounding prevents any extreme values at init
        s = 3**0.5 * config.hidden_dim**-0.5
        for layer in self.layers[:-1]:
            assert isinstance(layer, OneLayer)
            torch.nn.init.uniform_(layer.linear.weight, -s, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        for layer in self.layers:
            y = layer(y)
        return x + y


class ResidualMLP(nn.Module):
    def __init__(self, config: ResidualMLPConfig):
        super().__init__()
        self.norm = nn.Identity()
        self.input_layer = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        torch.nn.init.zeros_(self.input_layer.bias)

        self.blocks = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.num_blocks)]
        )
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        torch.nn.init.zeros_(self.output_layer.bias)

    def get_device(self):
        return self.input_layer.weight.device

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
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, D = inputs.size()

        x = self.input_layer(inputs)
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_layer(x)

        if targets is not None:
            loss = F.mse_loss(x, targets, reduction="none")
            if mask is not None:
                loss = loss * mask[:, None]
                return loss.sum() / (
                    mask.sum() * x.size(1)
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
    ):
        assert pred_and_obs_idx is None, (
            "MLP does not support conditioning on sequences."
        )
        b = targets.shape[0]
        return self.forward(
            inputs=static_input, targets=targets.reshape(b, -1), mask=mask
        )
