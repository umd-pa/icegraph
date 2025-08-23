# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from icegraph.trainer.callbacks.base import Normalizer
else:
    Trainer = Normalizer = None


class CoreModel(torch.nn.Module):
    """
    Core inference-ready model that bundles:
      - The trained nn.Module
      - The normalizer used during training
      - Metadata/config needed for preprocessing and postprocessing
    """

    _T = TypeVar("_T", bound=Normalizer)

    def __init__(
        self,
        net: torch.nn.Module,
        normalizer: type[_T],
        metadata: dict,
    ):
        super().__init__()
        self.net = net
        self.normalizer = normalizer
        self.metadata = metadata

    def forward(self, batch):
        """
        Forward pass through the normalizer (if applicable) and the model.
        """
        batch = self.normalizer.dispatch(batch)
        return self.net(batch)