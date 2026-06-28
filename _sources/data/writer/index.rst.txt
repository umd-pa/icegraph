Writer
======

The **writer** is the final stage of the :doc:`pipeline <../index>`. A pipeline
has exactly one writer.

Usage
-----

Configured under the top-level ``writer`` key.

.. code-block:: yaml

   writer:
     name: lmdb
     kwargs:
       outdir: /path/to/output

Variants
--------

* :doc:`LMDB <variants/lmdb/index>`: writes the dataset as an LMDB database.

Registering a new writer
------------------------

A writer is a subclass of ``Writer`` that declares a ``name`` and ``version`` and
implements the per-envelope write. Register it with ``WriterFactory``.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.data.writer import Writer, WriterFactory
   from icegraph.data.envelope import Envelope

   from .config import MyWriterConfig

   class MyWriter(Writer[MyWriterConfig]):
       name: ClassVar[str] = "my-writer"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyWriterConfig:
           return MyWriterConfig(**config)

       def build(self) -> None:
           ...

       def _process(self, item: Envelope) -> Envelope | None:
           ...  # persist the envelope

   WriterFactory.register(MyWriter)

.. toctree::
   :hidden:

   variants/lmdb/index
