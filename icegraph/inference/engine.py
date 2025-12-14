# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Union

import torch

from .models import CoreModel


class InferenceEngine:

    def __init__(self, model: CoreModel) -> None:
        self.model = model

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        """Takes in a dense array of shape [N, F] where N = # DOMs and F = # features."""
        # forward pass a copy of the tensor to avoid side effects
        return self.model.forward(inputs.clone())

    __call__ = infer
