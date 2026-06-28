Transformer
===========

The **transformer** is the :doc:`component <../index>` that applies a per-feature
value-space transform before normalization, compressing the dynamic range of
features that span many orders of magnitude (for example mapping a charge into log
space). Like the :doc:`normalizer <../normalizer/index>` it is a pure, invertible
value map: it changes values without changing tensor shape, and it normalizes
targets only when the task requests it.

Usage
-----

The transformer occupies the ``components.transformer`` slot.

.. code-block:: yaml

   components:
     transformer:
       name: standard
       kwargs:
         transforms:
           energy: { space: log, base: 10 }

How it Works
------------

The transformer assigns each feature column a transform space and applies the
corresponding mapping, vectorized across columns. Columns with no explicit configuration
are left in linear space (untouched). The transform composes with the normalizer:
the transformer reshapes the value distribution, then the normalizer scales it.

Variants
--------

* :doc:`Standard <variants/standard/index>`: per-column selection of a linear, log,
  or inverse-hyperbolic-sine space.

Registering a new transformer
-----------------------------

A transformer is a subclass of ``Transformer`` that declares a ``name`` and
``version`` and implements two methods:

``transform(self, t, /, role, *, inverse) -> Tensor``
   Map the values of ``t`` for the given column ``role``, returning a tensor of the
   **same shape**; undo the mapping when ``inverse`` is ``True``.
``_build_spec_list(self, role) -> list[TransformerSpec]``
   Describe, per column, which space applies. The base caches this and wraps
   ``transform`` with shape and finiteness checks.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.data import ColumnarRole
   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.transformer import Transformer, TransformerFactory
   from icegraph.engine.components.transformer.types import TransformerSpec

   from .config import MyTransformerConfig

   class MyTransformer(Transformer[MyTransformerConfig]):
       name: ClassVar[str] = "my-transformer"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyTransformerConfig:
           return MyTransformerConfig(**config)

       def build(self) -> None:
           ...

       def transform(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
           ...  # return the mapped values, same shape as t.data

       def _build_spec_list(self, role: ColumnarRole) -> list[TransformerSpec]:
           ...

   TransformerFactory.register(MyTransformer)

.. toctree::
   :hidden:

   variants/standard/index
