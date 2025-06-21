# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Tuple, TYPE_CHECKING, TypeVar, Type

import numpy as np

if TYPE_CHECKING:
    from icegraph.data.base.models import IGData

__all__ = ["set_cache_inst", "cache_event_worker"]


t = TypeVar("t", bound="IGData")

# Module‐level storage for the current IGData instance
_current_cache_inst: Type[t]

def set_cache_inst(inst: Type[t]) -> None:
    """
    Register the IGData instance for use by worker processes.

    Args:
        inst (IGData): The dataset instance whose cache is being populated.
    """
    global _current_cache_inst
    _current_cache_inst = inst


def cache_event_worker(idx: int) -> int:
    """
    Worker function to build cache for a single event.

    This function is invoked in parallel by a multiprocessing Pool. It pulls
    labels and features for the given index from the registered IGData instance
    and writes them to the on-disk cache.

    Args:
        idx (int): Event index.
    """
    inst = _current_cache_inst
    if inst is None:
        raise RuntimeError("IGData instance not registered for cache_event_worker")

    row = inst.truth_df.iloc[idx]
    labels = row[inst.target_labels].to_numpy(dtype=np.float32)

    keys = {col: int(row[col]) for col in inst.event_id_columns}

    features = inst._get_features_for_event(keys)

    inst._data_cache.register(inst, idx, features, labels)

    return idx