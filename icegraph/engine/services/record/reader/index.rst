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
``record_count(self) -> int``
   Return the number of records in the file.
``get(self, index) -> Record``
   Return the record at the given index.
``_build_attrs(self) -> Attributes``
   Read the file's attributes.
``sleep(self) -> None``
   Release any open file handles until the reader is next used.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.common.record import Attributes, Record
   from icegraph.engine.services.record.reader import Reader, ReaderFactory

   from .config import MyReaderConfig

   class MyReader(Reader[MyReaderConfig]):
       name: ClassVar[str] = "my-reader"
       version: ClassVar[int] = 1
       file_ext: ClassVar[str] = ".myext"

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyReaderConfig:
           return MyReaderConfig(**config)

       def build(self) -> None:
           ...

       def record_count(self) -> int:
           ...

       def get(self, index: int) -> Record:
           ...

       def _build_attrs(self) -> Attributes:
           ...

       def sleep(self) -> None:
           ...

   ReaderFactory.register(MyReader)

.. toctree::
   :hidden:

   variants/lmdb/index
