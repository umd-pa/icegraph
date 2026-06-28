Store
=====

The **store** presents the collection of dataset files as a single, indexable
sequence of records for the :doc:`record service <../index>`, and decides how much
data is held in memory. The store handles the dataset as a whole and the caching policy across it.

Usage
-----

Selected under ``services.record.store``.

.. code-block:: yaml

   services:
     record:
       store:
         name: lru-shard
         kwargs:
           cache_size: 32

Variants
--------

* :doc:`LRU Shard <variants/lrustore/index>`: caches a bounded number of file
  shards on a least-recently-used basis.

Registering a new store
-----------------------

A store is a subclass of ``Store`` that declares a ``name`` and ``version`` and
implements the abstract methods below. Register it with ``StoreFactory``.

``build(self) -> None``
   One-time setup.
``__getitem__(self, index) -> Record | list[Record]``
   Return the record(s) at an index or slice.
``__len__(self) -> int``
   Return the total number of records.
``attrs(self) -> Iterator[Attributes]``
   Iterate the per-file attributes across the dataset.
``close(self) -> None``
   Release any held resources.

.. code-block:: python

   from typing import Any, ClassVar
   from collections.abc import Iterator

   from icegraph.common.record import Attributes, Record
   from icegraph.engine.services.record.store import Store, StoreFactory

   from .config import MyStoreConfig

   class MyStore(Store[MyStoreConfig]):
       name: ClassVar[str] = "my-store"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyStoreConfig:
           return MyStoreConfig(**config)

       def build(self) -> None:
           ...

       def __getitem__(self, index: int | slice) -> Record | list[Record]:
           ...

       def __len__(self) -> int:
           ...

       def attrs(self) -> Iterator[Attributes]:
           ...

       def close(self) -> None:
           ...

   StoreFactory.register(MyStore)

.. toctree::
   :hidden:

   variants/lrustore/index
