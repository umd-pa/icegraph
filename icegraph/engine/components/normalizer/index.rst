Normalizer
==========

The **normalizer** is the :doc:`component <../index>` that rescales feature and
target tensors into a numerically conditioned range before they reach the model,
keeping input columns on comparable scales so optimization is well behaved. It alters
tensor values without changing their
shape, and it is invertible, so model predictions can be mapped back into the
original units. By default it normalizes input features only. Targets are
normalized only when the :doc:`policy <../../policy/index>` requests it.

Usage
-----

The normalizer occupies the ``components.normalizer`` slot.

.. code-block:: yaml

   components:
     normalizer:
       name: zscore
       kwargs: {}

How it Works
------------

A normalizer derives its parameters from data rather than from fixed settings. It
reads training-split statistics from the dataset computed during processing.
The resolved parameters are stored as buffers
on the component, so they are written into the checkpoint and reused unchanged at
inference, guaranteeing that the same transform is applied during training and deployment.
Whether targets are normalized is resolved once and
then retained, so a model reloaded from a checkpoint behaves consistently without
its original configuration.

Subclasses
----------

.. toctree::
   :maxdepth: 2

   variants/affine/index

Variants
--------

The normalizer slot has no variants registered directly on the base;
every selectable normalizer is provided by one of the subclasses above.

Registering a new normalizer
----------------------------

A normalizer is a subclass of ``Normalizer`` that declares a ``name`` and
``version``, validates its configuration, and implements the transform. The base
class drives both the forward and inverse passes through a single required method:

``normalize(self, t, /, role, *, inverse) -> Tensor``
   Map the values of ``t`` (a segmented tensor) for the given column ``role`` and
   return a tensor of the **same shape**. When ``inverse`` is ``True``, undo the mapping.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor
   from pydantic import BaseModel

   from icegraph.common.data import ColumnarRole
   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.normalizer import Normalizer, NormalizerFactory

   class Config(BaseModel):
       ...  # declare and validate any options the normalizer accepts

   class MyNormalizer(Normalizer[Config]):
       name: ClassVar[str] = "my-normalizer"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> Config:
           return Config(**config)

       def build(self) -> None:
           ...  # one-time setup, e.g. registering buffers

       def normalize(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
           ...  # return the mapped values, with the same shape as t.data

   NormalizerFactory.register(MyNormalizer)

.. code-block:: yaml

   components:
     normalizer:
       name: my-normalizer
       kwargs: {}
