# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any, TYPE_CHECKING, Iterator

from torch.utils.data import DataLoader

from icegraph.common.data import RawGraphBatch

from .dataset import GraphDataset


def _identity_collate(batch: RawGraphBatch) -> RawGraphBatch:
    # the dataset yields fully assembled batches; module-level so spawn workers can pickle it
    return batch


class GraphDataLoader(DataLoader):
    def __init__(self, dataset: GraphDataset, **kwargs: Any) -> None:
        # batching happens in the dataset (batch_size=None disables torch's collation)
        kwargs["batch_size"] = None
        kwargs["collate_fn"] = _identity_collate

        super().__init__(dataset, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Forwards epoch to the dataset, epoch update is visible to each worker."""
        dataset: GraphDataset = self.dataset  # type: ignore
        dataset.set_epoch(epoch)

    if TYPE_CHECKING:
        # the idiots who made pytorch dont know how to use generics
        # so now I have to do it myself
        def __iter__(self) -> Iterator[RawGraphBatch]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
