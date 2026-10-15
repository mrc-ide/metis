import torch
import os


def make_deterministic(seed: int):
    """Enable fully deterministic training (comparable to JAX).

    Caveats:
    - Performance hit (~2-5x slower than non-deterministic) due to SDPA math backend fallback.
    - Must be called BEFORE model creation and any CUDA operations.
    - CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA kernel launch.
    - SDPA falls back to the math backend (flash/mem-efficient are non-deterministic).
    - torch.compile may conflict with deterministic algorithms; disable it when using this.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Flash and memory-efficient SDPA backends are non-deterministic
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("Running in deterministic mode, will likely be much slower")
