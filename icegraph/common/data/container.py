# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast, Callable, Mapping

from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from jaxtyping import Int, Float
import torch

from icegraph.common.data import DataRole, ColumnarRole

from ..tensors import SegmentedTensor, SegmentLayout

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GraphData", "RawGraphBatch", "GraphBatch", "ProcessedGraphBatch"]


class GraphData(Data):
    if TYPE_CHECKING:
        # features
        features:       Float[Tensor, "N F"]

        # truth
        targets:        Float[Tensor, "1 T"] | Int[Tensor, "1 T"]
        auxiliary:      Float[Tensor, "1 A"] | Int[Tensor, "1 A"]

        # weights
        simweights:     Float[Tensor, "0"] | Float[Tensor, "1"]

        # edges
        # this is just type hinting so ignore pyright complaints
        edge_index:     Int[Tensor, "2 E"]          # pyright: ignore[reportIncompatibleMethodOverride]
        edge_attr:      Float[Tensor, "E ATTR"]     # pyright: ignore[reportIncompatibleMethodOverride]

    def __cat_dim__(self, key: str, value: Tensor, *args, **kwargs) -> int:
        if key in {"features", "targets", "auxiliary", "edge_attr"}:
            return 0

        return super().__cat_dim__(key, value, *args, **kwargs)


class RawGraphBatch(Batch):
    if TYPE_CHECKING:
        # features
        features:       Float[Tensor, "M F"]  # M = sum(N_i)

        # truth
        targets:        Float[Tensor, "B T"] | Int[Tensor, "B T"]
        auxiliary:      Float[Tensor, "B A"] | Int[Tensor, "B A"]

        # weights
        simweights:     Float[Tensor, "0"] | Float[Tensor, "B"]

        # edges
        edge_index:     Int[Tensor, "2 K"]  # K = sum(E_i)
        edge_attr:      Float[Tensor, "K ATTR"]

        # batch vector
        batch:          Int[Tensor, "M"]

    def to_device(
        self,
        device: torch.device | str | int | None = None, *,
        non_blocking: bool = False,
    ) -> Self:
        # .to exists, just not stubbed
        return super().to(device=device, non_blocking=non_blocking)  # pyright: ignore[reportAttributeAccessIssue]

    if TYPE_CHECKING:
        @classmethod
        def from_data_list(
            cls,
            data_list: list[BaseData],
            follow_batch: list[str] | None = None,
            exclude_keys: list[str] | None = None,
        ) -> Self: ...


class GraphBatch(Batch):
    # features (jagged, store flattened, in the vast majority of cases each segment is unity)
    features:       SegmentedTensor  # M = sum(N_i)

    # truth (both jagged, but stored flattened)
    targets:        SegmentedTensor
    auxiliary:      SegmentedTensor

    # weights
    simweights:     Float[Tensor, "0"] | Float[Tensor, "B"]

    # edges
    edge_index:     Int[Tensor, "2 K"]  # K = sum(E_i)
    edge_attr:      Float[Tensor, "K ATTR"]

    # batch vector
    batch:          Int[Tensor, "M"]

    def to_device(
        self,
        device: torch.device | str | int,
        non_blocking: bool = False,
    ) -> Self:
        cls = type(self)

        # get each attr and apply
        kwargs: dict[str, Tensor | SegmentedTensor] = {}
        for role in DataRole.all():
            kwargs[role.value] = getattr(self, role.value).to(device=device, non_blocking=non_blocking)

        return cls(**kwargs)

    def to_dtype(self, mapping: Mapping[DataRole, torch.dtype]) -> Self:
        cls = type(self)

        # get each attr and apply
        kwargs: dict[str, Tensor | SegmentedTensor] = {
            role.value: getattr(self, role.value) for role in DataRole.all()
        }
        for role, dtype in mapping.items():
            kwargs[role.value] = kwargs[role.value].to(dtype=dtype)

        return cls(**kwargs)

    def detach(self) -> Self:
        cls = type(self)

        kwargs: dict[str, Tensor | SegmentedTensor] = {}
        for role in DataRole.all():
            kwargs[role.value] = getattr(self, role.value).detach()

        return cls(**kwargs)

    @classmethod
    def from_raw_batch(
            cls,
            batch: RawGraphBatch,
            get_layout: Callable[[ColumnarRole, torch.device], SegmentLayout],
    ) -> Self:
        kwargs: dict[str, Tensor | SegmentedTensor] = {
            role.value: getattr(batch, role.value) for role in DataRole.all()
        }

        # pack to segmented tensor for specified roles
        for role in DataRole.columnar():
            tensor = kwargs[role.value]
            assert not isinstance(tensor, SegmentedTensor)

            layout = get_layout(role, tensor.device)
            kwargs[role.value] = SegmentedTensor(tensor, layout)

        return cls(**kwargs)


class ProcessedGraphBatch(GraphBatch):
    # output (jagged, but stored flattened)
    out: SegmentedTensor

    def detach(self) -> Self:
        cls = type(self)

        kwargs: dict[str, Tensor | SegmentedTensor] = {}
        for role in DataRole.all():
            kwargs[role.value] = getattr(self, role.value).detach()

        # out is not a DataRole enum member (intentionally), so handle explicitly
        kwargs["out"] = self.out.detach()

        return cls(**kwargs)

    @classmethod
    def from_graph_batch(
            cls,
            batch: GraphBatch,
            *,
            out: SegmentedTensor
    ) -> Self:
        kwargs: dict[str, Tensor | SegmentedTensor] = {}
        for role in DataRole.all():
            kwargs[role.value] = getattr(batch, role.value).detach()

        # out is not a DataRole enum member (intentionally), so handle explicitly
        kwargs["out"] = out

        return cls(**kwargs)
