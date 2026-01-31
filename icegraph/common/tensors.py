# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch import Tensor


class DualResidentTensor:
    """
    Tensor with a canonical CPU copy and a cached GPU copy.

    The CPU tensor is the source of truth.
    A GPU tensor is materialized lazily on demand for a given device.
    Mutations to the GPU tensor do not propagate back to CPU.
    """
    _cpu: Tensor
    _gpu: Tensor | None = None

    def __init__(self, tensor: Tensor):
        if tensor.device.type != "cpu":
            raise ValueError(f"Input tensor must be on CPU, got {tensor.device}.")

        # cache the cpu tensor
        self._cpu = tensor

    def on(self, device: torch.device | str) -> Tensor:
        # normalize
        device = torch.device(device)

        # return cpu tensor if device is cpu
        if device.type == "cpu":
            return self._cpu

        # move to device if no gpu tensor
        if self._gpu is None or self._gpu.device != device:
            self._gpu = self._cpu.to(device=device)

        # return gpu tensor
        return self._gpu
