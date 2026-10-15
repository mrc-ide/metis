import contextlib
from enum import Enum

import torch


class PrecisionMode(Enum):
    FP32 = "fp32"  # Strict fp32, no tensor core approximations
    TF32 = "tf32"  # TF32 tensor cores for matmul
    BF16_AMP = "bf16"  # BF16 automatic mixed precision via torch.amp.autocast


def configure_precision(mode: PrecisionMode):
    """Configure global torch precision settings.

    Must be called before model creation for full effect.

    - FP32: Highest precision, disables tf32 approximations entirely.
    - TF32: Allows tf32 tensor cores for matmul and cudnn (PyTorch default on Ampere+).
    - BF16_AMP: Same backend settings as TF32; bf16 is handled by autocast in the training loop.
    """
    if mode == PrecisionMode.FP32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == PrecisionMode.TF32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif mode == PrecisionMode.BF16_AMP:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        raise ValueError(f"Unknown precision mode: {mode}")


def training_autocast(mode: PrecisionMode, device_type: str):
    """Return an autocast context manager for the given precision mode.

    - BF16_AMP: returns torch.amp.autocast with bfloat16
    - FP32/TF32: returns a no-op context manager (computation stays in fp32)
    """
    if mode == PrecisionMode.BF16_AMP:
        return torch.amp.autocast(device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
