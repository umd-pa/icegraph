Writer
======

The **writer** is the final stage of the :doc:`pipeline <../index>`. A pipeline
has exactly one writer.

Usage
-----

Configured under the top-level ``writer`` key. The output directory is handled by the
pipeline.

.. code-block:: yaml

   writer:
     name: zarr
     kwargs:
       prefix: corsika

Each :doc:`collected <../collector/index>` block is written as one shard, named as:
``{prefix}.shard.{index}{suffix}``, or ``shard.{index}{suffix}``
without a prefix, where ``index`` is zero padded to six digits (``corsika.shard.000012.zarr``).
``index`` represents the order in which the blocks were dispatched from the collector.

Shards are numbered starting from 0 in every run, so a shard left in the output directory by an
earlier run is overwritten or mixed in with this one. The pipeline warns when the
output directory is not empty. Independent runs writing to one directory must each set a
different ``prefix``.

Every writer takes:

.. list-table::
   :header-rows: 1
   :widths: 14 62 12 12

   * - Option
     - Description
     - Type
     - Default
   * - ``prefix``
     - Prefix of every shard name.
     - str
     - ``null``

Variants
--------

* :doc:`Zarr <variants/zarr/index>`: writes each shard as a Zarr group.
* :doc:`LMDB <variants/lmdb/index>`: writes each shard as an LMDB database.

Registering a new writer
------------------------

A writer is a subclass of ``Writer`` that declares a ``name`` and ``version`` and
implements the per-envelope write. Register it with ``WriterFactory``.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.data.writer import Writer, WriterFactory
   from icegraph.data.envelope import Envelope

   from .config import MyWriterConfig  # a subclass of WriterConfig

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

   variants/zarr/index
   variants/lmdb/index
