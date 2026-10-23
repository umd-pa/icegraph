Record Decoder
==============

The **record decoder** reads columnar blocks of dataset records for the
:doc:`decode service <../index>` and decodes their columns: the node features,
the targets, the graph connectivity, any auxiliary columns, and the per-record
weights.

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
  standard columnar schema, and weights IceCube simulation.

Registering a new record decoder
--------------------------------

A record decoder is a subclass of ``RecordDecoder`` that declares a ``name`` and
``version`` and implements the lookup of a raw column from a block by key.
Register it with ``RecordDecoderFactory``.

The ``RecordDecoder`` additionally provides optional hooks for role-specific
decoding overrides. Roles whose row count varies per record (features) return the
flat values together with per-record row counts:

``_extract_features(self, block, key) -> tuple[Tensor, counts] | None``

``_extract_targets(self, block, key) -> Tensor | None``

``_extract_auxiliary(self, block, key) -> Tensor | None``

``_extract_weights(self, block, key) -> Tensor | None``

The weights hook returns *final* weights, one per record: a decoder whose data
stores the inputs to a weight calculation rather than a weight computes them here.

Decoding that needs more than the block itself is served by the attach context:
``self._ctx.attrs`` walks every shard and ``self._ctx.global_attrs`` holds the
dataset globals, while ``self._ctx.columns(key)`` and ``self._ctx.dtypes(key)``
give the names and the written dtype of each column packed into a stored key, as
the attribute decoder resolved them, so a packed column can be put back the way it
was. Attributes a decoder reads itself are validated against a schema it declares;
see ``validate_attrs``.

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