Record Decoder
==============

The **record decoder** reads columnar blocks of dataset records for the
:doc:`decode service <../index>` and decodes their columns: the node features,
the targets, the graph connectivity, and any auxiliary columns.

Usage
-----

Selected under ``services.decode.records``.

.. code-block:: yaml

   services:
     decode:
       records:
         name: standard
         kwargs: {}

Variants
--------

* :doc:`Standard <variants/standard/index>`: decodes blocks written in the
  standard columnar schema.

Registering a new record decoder
--------------------------------

A record decoder is a subclass of ``RecordDecoder`` that declares a ``name`` and
``version`` and implements the lookup of a raw column from a block by key.
Register it with ``RecordDecoderFactory``.

The ``RecordDecoder`` additionally provides optional hooks for role-specific
decoding overrides. Roles whose row count varies per record (features, edges)
return the flat values together with per-record row counts:

``_extract_features(self, block, key) -> tuple[Tensor, counts] | None``

``_extract_targets(self, block, key) -> Tensor | None``

``_extract_auxiliary(self, block, key) -> Tensor | None``

``_extract_edge_index(self, block, key) -> tuple[Tensor, counts] | None``

``_extract_edge_attr(self, block, key) -> Tensor | None``

``_extract_simweights(self, block, key) -> Tensor | None``

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.common.record import Column, RecordBlock
   from icegraph.engine.services.decode.records import RecordDecoder, RecordDecoderFactory

   from .config import MyConfig

   class MyRecordDecoder(RecordDecoder[MyConfig]):
       name: ClassVar[str] = "my-records"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyConfig:
           return MyConfig(**config)

       def build(self) -> None:
           ...

       def extract(self, block: RecordBlock, key: str) -> Column | None:
           ...

   RecordDecoderFactory.register(MyRecordDecoder)

.. toctree::
   :hidden:

   variants/standard/index
