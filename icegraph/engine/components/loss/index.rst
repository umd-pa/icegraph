Loss
====

The **loss** is the :doc:`component <../index>` that defines the objective
minimized during training. It compares the model's output to the targets and
reduces the comparison to a single scalar that gradients are computed against.

Usage
-----

The loss occupies the ``components.loss`` slot.

.. code-block:: yaml

   components:
     loss:
       name: cross-entropy
       kwargs: {}

How it Works
------------

A loss receives the model output and the targets, computes a per-head loss for each
prediction head, and sums them into a single scalar. The base class enforces that
the result is scalar.

Variants
--------

* :doc:`Cross Entropy <variants/cross_entropy/index>`: classification loss from
  logits.
* :doc:`NLL <variants/nll/index>`: negative log-likelihood for classification.
* :doc:`MSE <variants/mse/index>`: mean squared error for regression.
* :doc:`L1 <variants/l1/index>`: mean absolute error for regression.

Registering a new loss
----------------------

A loss is a subclass of ``LossFunction`` that declares a ``name`` and ``version``
and implements the objective:

``loss(self, out, target) -> Tensor``
   Compare the model output to the targets and return a scalar tensor.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.loss import LossFunction, LossFactory

   from .config import MyLossConfig

   class MyLoss(LossFunction[MyLossConfig]):
       name: ClassVar[str] = "my-loss"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyLossConfig:
           return MyLossConfig(**config)

       def build(self) -> None:
           return

       def loss(self, out: SegmentedTensor, target: SegmentedTensor, /) -> Tensor:
           ...  # return a scalar tensor

   LossFactory.register(MyLoss)

.. toctree::
   :hidden:

   variants/cross_entropy/index
   variants/nll/index
   variants/mse/index
   variants/l1/index
