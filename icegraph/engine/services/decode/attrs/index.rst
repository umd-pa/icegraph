Attribute Decoder
=================

The **attribute decoder** reads dataset-level metadata for the :doc:`decode service
<../index>`: the names of the columns, their per-column statistics, and the set of
values observed for label columns. This lets the rest of the run
size its tensors and resolve normalization parameters without scanning the data
itself.

Usage
-----

Selected under ``services.decode.attrs``.

.. code-block:: yaml

   services:
     decode:
       attrs:
         name: standard
         kwargs: {}

Variants
--------

* :doc:`Standard <variants/standard/index>`: reads the standard attribute schema written by
  the pipeline.

Registering a new attribute decoder
-----------------------------------

An attribute decoder is a subclass of ``AttributeDecoder`` that declares a ``name``
and ``version`` and implements the abstract extraction methods below. Each receives
``attrs`` (a callable yielding the per-file attributes) and ``global_attrs`` (the
dataset-wide attributes) as keyword arguments. Register it with
``AttributeDecoderFactory``.

``build(self) -> None``
   One-time setup.
``_extract_columns(self, role, *, attrs, global_attrs) -> list[str] | None``
   The column names for a role.
``_extract_offsets(self, role, *, attrs, global_attrs) -> ArrayI | None``
   The per-column offsets for a role.
``_extract_keys(self, split, *, attrs, global_attrs) -> ArrayI | None``
   The record keys belonging to a split.
``_extract_stats(self, split, role, *, attrs, global_attrs) -> StatisticService``
   The statistics for a split and role.
``_extract_count_by_weight_group(self, *, attrs, global_attrs) -> dict[str, int]``
   The record count per weight group.

.. code-block:: python

   from typing import Any, ClassVar
   from collections.abc import Callable, Iterator

   from icegraph.statistics import StatisticService
   from icegraph.typing.common import ArrayI
   from icegraph.common.record import Attributes, GlobalAttributes
   from icegraph.engine.services.decode.attrs import AttributeDecoder, AttributeDecoderFactory

   from .config import MyConfig

   Attrs = Callable[[], Iterator[Attributes]]

   class MyAttributeDecoder(AttributeDecoder[MyConfig]):
       name: ClassVar[str] = "my-attrs"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyConfig:
           return MyConfig(**config)

       def build(self) -> None:
           ...

       def _extract_columns(self, role: str, *, attrs: Attrs, global_attrs: GlobalAttributes) -> list[str] | None:
           ...

       def _extract_offsets(self, role: str, *, attrs: Attrs, global_attrs: GlobalAttributes) -> ArrayI | None:
           ...

       def _extract_keys(self, split: int, *, attrs: Attrs, global_attrs: GlobalAttributes) -> ArrayI | None:
           ...

       def _extract_stats(self, split: int, role: str, *, attrs: Attrs, global_attrs: GlobalAttributes) -> StatisticService:
           ...

       def _extract_count_by_weight_group(self, *, attrs: Attrs, global_attrs: GlobalAttributes) -> dict[str, int]:
           ...

   AttributeDecoderFactory.register(MyAttributeDecoder)

.. toctree::
   :hidden:

   variants/standard/index
