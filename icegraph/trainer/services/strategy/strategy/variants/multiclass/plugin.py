# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Any

from icegraph.types.data import AttributeDomain, ModelInputRole

from ...strategy import Strategy

from .config import Config

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Multiclass"]


class Multiclass(Strategy[Config]):
    name: ClassVar[str] = "multiclass"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def build(self) -> None:
        return

    def out_channels(self) -> int:
        # for classification, need to count total number of classes in training split (will be same as val/test)
        classmap: dict[str, dict[str, int]] = {}

        # iterate over shard attributes
        for attr in self._ctx.data.attrs:
            chunk = attr[AttributeDomain.LOCAL].get("classmap")
            if chunk is None:
                raise RuntimeError(
                    f"Could not find key 'classmap' in shard LOCAL attrs. Problematic shard: id={attr.shard_id}"
                )

            # build the classmap from each chunk
            for label, mapping in chunk.items():
                if label in classmap:
                    classmap[label].update(mapping)
                else:
                    classmap[label] = mapping

        return sum(len(m) for m in classmap.values())

    def in_channels(self) -> int:
        return len(self._ctx.data.columns(ModelInputRole.FEATURES))

    def adapt_targets(self, targets: Tensor) -> Tensor:
        # [B]
        if targets.dim() == 1:
            return targets.long()

        # [B, 1] -> [B]
        if targets.dim() == 2 and targets.size(1) == 1:
            return targets.view(-1).long()

        # one-hot [B, C]
        if targets.dim() == 2 and targets.size(1) == self.out_channels:
            # treat as one-hot regardless of dtype; CE uses indices
            return targets.argmax(dim=1).long()

        raise ValueError(
            f"Expected multiclass labels as indices [B]/[B, 1] or one-hot [B, C] "
            f"where C == logits ({self.out_channels}). Got shape={targets.shape}."
        )
