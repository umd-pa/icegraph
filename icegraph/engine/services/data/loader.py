# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from functools import partial
from typing import Any, Mapping

from torch.utils.data import DataLoader

from icegraph.common.data import DataRole
from icegraph.common.data import RawGraphBatch
from icegraph.common.tensors import SegmentLayout

from .dataset import GraphDataset


def graph_collate(
    data_list: list[Any],
    follow_batch: list[str] | None = None,
    exclude_keys: list[str] | None = None,
) -> RawGraphBatch:
    # build batch as normal
    batch = RawGraphBatch.from_data_list(
        data_list, follow_batch=follow_batch, exclude_keys=exclude_keys
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
