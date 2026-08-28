Reader
======

The **reader** defines the on-disk format of a dataset file for the :doc:`record
service <../index>`. The reader is the only part of the system that
understands the storage format, so supporting a new format means writing a new
reader.

Usage
-----

Selected under ``services.record.reader``.

.. code-block:: yaml

   services:
     record:
       reader:
         name: lmdb
         kwargs: {}

Variants
--------

* :doc:`LMDB <variants/lmdb/index>`: reads datasets stored as LMDB databases.

Registering a new reader
------------------------

A reader is a subclass of ``Reader`` that declares a ``name``, a ``version``, and
the ``file_ext`` it handles, and implements the abstract methods below. Register it
with ``ReaderFactory``.

``build(self) -> None``
   One-time setup.
``_open(self, path) -> Handle``
   Open the file and return an opaque handle.
``_close(self, handle) -> None``
   Release the handle's resources.
``_attrs_dict(self) -> dict``
   Read the file's raw attributes.
``_get(self, indices) -> RecordBlock``
   Read the given rows (ascending) as one columnar block.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.common.record import RecordBlock
   from icegraph.engine.services.record.reader import Reader, ReaderFactory

   from .config import MyReaderConfig

   class MyReader(Reader[MyReaderConfig, MyHandle]):
       name: ClassVar[str] = "my-reader"
       version: ClassVar[int] = 1
       file_ext: ClassVar[str] = ".myext"

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyReaderConfig:
           return MyReaderConfig(**config)

       def build(self) -> None:
           ...

       def _open(self, path) -> MyHandle:
           ...

       def _close(self, handle: MyHandle) -> None:
           ...

       @cached_property
       def _attrs_dict(self) -> dict[str, Any]:
           ...

       def _get(self, indices) -> RecordBlock:
           ...

   ReaderFactory.register(MyReader)

.. toctree::
   :hidden:

   variants/lmdb/index
