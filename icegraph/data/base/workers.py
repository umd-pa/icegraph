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


def cache_event_worker(item: Tuple[int, str]) -> None:
    """
    Worker function to build cache for a single event.

    This function is invoked in parallel by a multiprocessing Pool. It pulls
    labels and features for the given event_id from the registered IGData instance
    and writes them to the on-disk cache.

    Args:
        item (Tuple[int, str]): A tuple of (index, event_id).
    """
    idx, event_id = item
    inst = _current_cache_inst

    # Extract labels for this event
    labels = np.array([inst.label_map[label][event_id] for label in inst.target_labels])
    # Extract features for this event
    features = inst._get_features_for_event(event_id)

    # Register into the cache
    inst._data_cache.register(inst, idx, features, labels)