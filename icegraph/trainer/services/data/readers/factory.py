# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean
from typing import Any

# local package
from icegraph.types.factory import ModuleFactory

# local subpackage
from .reader import Reader
from .store import ShardStore
from .standard import LMDB

__all__ = ["ReaderFactory"]


class ReaderFactory(ModuleFactory[str, Reader]):
    @classmethod
    def create_store(cls, name: str, **kwargs: Any) -> ShardStore:
        spec = cls._typed_registry()[name]
        return ShardStore(spec, **kwargs)


# register each internal module
for module in [LMDB]:
    ReaderFactory.register(module)
