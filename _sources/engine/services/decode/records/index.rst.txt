Record Decoder
==============

The **record decoder** reads an individual dataset record for the :doc:`decode
service <../index>` and extracts its tensors: the node features, the targets, the
graph connectivity, and any auxiliary columns.

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

* :doc:`Standard <variants/standard/index>`: extracts tensors from records written
  in the standard schema.

Registering a new record decoder
--------------------------------

A record decoder is a subclass of ``RecordDecoder`` that declares a ``name`` and
``version`` and implements extraction of a tensor from a record by key. Register it
with ``RecordDecoderFactory``.

The ``RecordDecoder`` additionally provides optional hooks for label-specific
extraction overrides:

``_extract_features(self, record, key) -> Tensor | None``

``_extract_targets(self, record, key) -> Tensor | None``

``_extract_auxiliary(self, record, key) -> Tensor | None``

``_extract_edge_index(self, record, key) -> Tensor | None``

``_extract_edge_attr(self, record, key) -> Tensor | None``

``_extract_simweights(self, record, key) -> Tensor | None``

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.record import Record
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

       def extract(self, record: Record, key: str) -> Tensor | None:
           ...

   RecordDecoderFactory.register(MyRecordDecoder)

.. toctree::
   :hidden:

   variants/standard/index
