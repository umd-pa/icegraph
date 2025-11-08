# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import sys
from typing import Tuple, Optional, Union, Callable

from torch import nn
import torch_scatter
import torch
from torch_geometric.data import Batch

from .base import TaskStrategy
from .exceptions import InvalidStrategyError
from .metrics import RegressionMetrics, MulticlassMetrics


def resolve_strategy(name: str, call: bool = True) -> Union[TaskStrategy, Callable[[...], ...]]:
    __module__ = sys.modules[__name__]
    try:
        cls = getattr(__module__, name)
    except AttributeError:
        raise ValueError(f"TaskStrategy class '{name}' not found in {__module__.__name__}")
    if not callable(cls) and call:
        raise TypeError(f"{name} is not callable.")
    return cls() if call else cls


class RegressionStrategy(TaskStrategy):
    task = "regression"

    def loss_function(self) -> nn.Module:
        return nn.MSELoss(reduction=self._enforced_reduction)

    def make_metrics(self) -> RegressionMetrics:
        return RegressionMetrics()

    def adapt_targets(self, batch: Batch, out: torch.Tensor) -> torch.Tensor:
        y = batch.y

        # if y is node-level scalar labels: aggregate to graph
        if y.dim() == 1 and y.size(0) == batch.batch.size(0):
            y = torch_scatter.scatter_mean(y, batch.batch, dim=0)

        # [y] -> [y, 1]
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # cast to fp32 if required (should already be fp32)
        if y.dtype != torch.float32:
            y = y.to(torch.float32)

        return y

    def out_channels(self, trainer) -> int:
        return trainer.registry.train_dataset.num_target_labels

    def in_channels(self, trainer) -> int:
        return trainer.registry.train_dataset.num_node_features

    def filter_eval(
            self,
            out: torch.Tensor,
            target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(out) & torch.isfinite(target)
        out_f = out[finite].view(-1, 1)
        target_f = target[finite].view(-1, 1)

        return out_f, target_f, finite


class MulticlassStrategy(TaskStrategy):
    task = "multiclass"

    def loss_function(self) -> nn.Module:
        # weights moved to correct device in post_init_check()
        return nn.CrossEntropyLoss(reduction=self._enforced_reduction)

    def make_metrics(self) -> MulticlassMetrics:
        # return a metrics object each run/epoch
        return MulticlassMetrics(**self.kwargs)

    def adapt_targets(self, batch: Batch, out: torch.Tensor) -> torch.Tensor:
        y = batch.y

        # [B] -> indices
        if y.dim() == 1:
            return y.long()

        # [B, 1] -> squeeze to [B]
        if y.dim() == 2 and y.size(1) == 1:
            return y.view(-1).long()

        # one-hot [B, C], validate against logits width
        if y.dim() == 2 and y.size(1) == out.size(1):
            # treat as one-hot regardless of dtype; CE uses indices
            return y.argmax(dim=1).long()

        raise ValueError(
            f"Expected multiclass labels as indices [B]/[B,1] or one-hot [B,C] where C==logits ({out.size(1)}). "
            f"Got y.shape={tuple(y.shape)}."
        )

    def out_channels(self, trainer) -> int:
        # for classification, need to count total number opf classes in training split (will be same as val/test)
        attrs = trainer.registry.attrs
        try:
            indices = set()
            for i, data in attrs.items():
                classmap = data["map"]["classmap"]

                # if there is more than one target label for classification, raise; should be using multilabel strategy
                n_classes = len(classmap)
                if n_classes > 1:
                    raise InvalidStrategyError(
                        f"Found {n_classes} labels in classmap. Multiclass strategy not equipped to handle multilabel "
                        f"classification (n_classes > 1); use Multilabel strategy."
                    )

                mapping = next(iter(classmap.values()))
                indices.update(mapping.values())
            return len(indices)
        except KeyError:
            raise KeyError("Could not find classmap in dataset attributes (attrs[idx]['map']['classmap'])")

    def in_channels(self, trainer) -> int:
        return trainer.registry.train_dataset.num_node_features

    def filter_eval(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return out, target, None
