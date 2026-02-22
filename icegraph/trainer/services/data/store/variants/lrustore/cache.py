# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any
from collections import OrderedDict
from typing import Iterator

from ...readers import Reader

__all__ = ["ShardLRUCache"]


class ShardLRUCache:

    def __init__(self, readers: list[Reader], capacity: int) -> None:
        """Initialize a shard LRU cache handler."""
        self._readers:      list[Reader]    = readers
        self._capacity:     int             = capacity

        # validate capacity
        if not self._capacity > 1:
            raise ValueError("LRU cache capacity must be an int greater than 1.")

        # currently awake cache
        self._awake: OrderedDict[int, None] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self._awake

    def __len__(self) -> int:
        return len(self._awake)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __getstate__(self) -> dict[str, Any]:
        self.clear()
        return self.__dict__.copy()

    def get_reader(self, key: int) -> Reader:
        """Get a reader by index, handles caching and evicting readers in the background."""
        # ensure index is valid
        assert (0 <= key < len(self._readers)), f"Invalid key '{key}' for readers of length {len(self._readers)}"

        # if reader is already awake, move to end and return reader
        if key in self._awake:
            self._awake.move_to_end(key, last=True)
            return self._readers[key]

        # if not already awake, add to awake
        self._awake[key] = None

        # ensure cache stays bounded to capacity
        if len(self._awake) > self._capacity:
            self.evict_last()

        # return reader
        return self._readers[key]

    def iter_readers(self) -> Iterator[Reader]:
        """Iterate over each reader."""
        for key in range(len(self._readers)):
            yield self.get_reader(key)

    def evict_last(self) -> None:
        """Evict the least recently used reader."""
        # do nothing if cache is empty
        if not self._awake:
            return

        # pop least recently used and sleep instance
        evicted = self._awake.popitem(last=False)[0]
        self._readers[evicted].sleep()

    def mru(self) -> Reader | None:
        """Return the most recently used reader (MRU), or None if empty."""
        key = next(reversed(self._awake), None)
        return None if key is None else self._readers[key]

    def clear(self) -> None:
        """Clear the whole cache."""
        # evict everything in cache
        while self._awake:
            self.evict_last()

