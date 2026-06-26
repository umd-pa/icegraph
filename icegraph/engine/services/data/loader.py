# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from functools import partial
from typing import Any, TYPE_CHECKING, Iterator, cast

from torch.utils.data import DataLoader
from torch_geometric.data.data import BaseData

from icegraph.common.data import RawGraphBatch, GraphData

from .dataset import GraphDataset


def graph_collate(
    data_list: list[GraphData],
    follow_batch: list[str] | None = None,
    exclude_keys: list[str] | None = None,
) -> RawGraphBatch:
    # build batch as normal
    batch = RawGraphBatch.from_data_list(
        cast("list[BaseData]", data_list), follow_batch=follow_batch, exclude_keys=exclude_keys
    )

    return batch


class GraphDataLoader(DataLoader):
    def __init__(
        self,
        dataset: GraphDataset,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["collate_fn"] = partial(
            graph_collate,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys
        )

        super().__init__(dataset, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Forwards epoch to the dataset, epoch update is visible to each worker."""
        dataset: GraphDataset = self.dataset  # type: ignore
        dataset.set_epoch(epoch)

    if TYPE_CHECKING:
        # the idiots who made pytorch dont know how to use generics
        # so now I have to do it myself
        def __iter__(self) -> Iterator[RawGraphBatch]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
