# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING, Union, Self
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch import Tensor

from icegraph._version import __version__
from icegraph.console import Console
from icegraph.trainer.arch import ModelFactory
from icegraph.trainer.normalizers import NormalizerFactory

if TYPE_CHECKING:
    from icegraph.trainer.normalizers import Normalizer
else:
    Trainer = Normalizer = None


class CoreModel(torch.nn.Module):
    """
    Core inference-ready model.
    """

    def __init__(self, network: torch.nn.Module, normalizer: torch.nn.Module, metadata: dict) -> None:
        super().__init__()

        self.network = network
        self.normalizer = normalizer
        self.metadata = metadata

        self._provenance_check()

    def _provenance_check(self) -> None:
        def _omit_patch_number(version: str) -> str:
            return version.rsplit(".", 1)[0]

        model_version = _omit_patch_number(self.metadata["model"]["version"])
        runtime_version = _omit_patch_number(__version__)

        if not model_version == runtime_version:
            Console.out(
                f"Version mismatch: model was created using IceGraph {model_version}, but the current "
                f"runtime is {runtime_version}. Compatibility is not guaranteed.",
                severity=2
            )

    def forward(self, data: Union[Tensor, Batch]) -> Tensor:
        if not isinstance(data, (Tensor, Batch)):
            raise TypeError("Input data must be a torch Tensor or Batch object.")

        with torch.inference_mode():
            # normalize the input data (tensor or batch object)
            normalized_data: Union[Tensor, Batch] = self.normalizer.dispatch(data)

            if isinstance(normalized_data, Batch):
                # unpack the Batch object, only pass features
                inferred = self.network(normalized_data.x, normalized_data.batch)

            elif isinstance(normalized_data, torch.Tensor):
                # pass the single feature tensor to the network
                inferred = self.network(normalized_data)

            return self.normalizer.dispatch(inferred, inverse=True, target="labels")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> Self:
        state = torch.load(str(path), **kwargs)

        # load the normalizer state
        norm_name, norm_state = state["normalizer"]

        # build the normalizer instance
        normalizer = NormalizerFactory.create(norm_name)
        normalizer.load_state_dict(norm_state)

        # load the network state
        net_name, net_state = state["network"]

        # build the network instance
        network = ModelFactory.create(net_name)
        network.load_state_dict(net_state)

        # construct the core model
        model = cls(network, normalizer, state["metadata"])

        return model