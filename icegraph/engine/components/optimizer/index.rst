Optimizer
=========

The **optimizer** is the :doc:`component <../index>` that updates the model's
weights from their gradients during training. It defines the optimization
algorithm and its hyperparameters.

Usage
-----

The optimizer occupies the ``components.optimizer`` slot.

.. code-block:: yaml

   components:
     optimizer:
       name: adam
       kwargs:
         lr: 0.0002
         weight_decay: 0.00005

How it Works
------------

When the run is assembled, the optimizer takes the model component's parameters and
wraps the corresponding optimization algorithm. It exposes the two operations the
training loop drives: a step that applies an update, and a gradient reset.

Variants
--------

* :doc:`Adam <variants/adam/index>`: adaptive per-parameter step sizes with
  decoupled weight decay.
* :doc:`SGD <variants/sgd/index>`: stochastic gradient descent, optionally with
  momentum and Nesterov acceleration.

Registering a new optimizer
---------------------------

An optimizer is a subclass of ``Optimizer`` that declares a ``name`` and
``version``, validates its configuration, performs one-time setup in ``build``,
binds to the model parameters on attach, and implements the two operations the
training loop calls:

``build(self) -> None``
   One-time setup, before the model parameters are available.
``step(self) -> None``
   Apply one optimization update.
``zero_grad(self, set_to_none=True) -> None``
   Clear the accumulated gradients.

The underlying algorithm is typically constructed in ``on_attach``, where the model
component (and therefore its parameters) is available.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.common.engine import ComponentKind
   from icegraph.engine.components.optimizer import Optimizer, OptimizerFactory

   from .config import MyOptimizerConfig

   class MyOptimizer(Optimizer[MyOptimizerConfig]):
       name: ClassVar[str] = "my-optimizer"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyOptimizerConfig:
           return MyOptimizerConfig(**config)

       def build(self) -> None:
           ...  # one-time setup before the model is available

       def on_attach(self) -> None:
           model = self._ctx.components.require(ComponentKind.MODEL, required_by=type(self))
           ...  # build the optimizer from model.parameters()

       def step(self) -> None:
           ...

       def zero_grad(self, set_to_none: bool = True) -> None:
           ...

   OptimizerFactory.register(MyOptimizer)

.. toctree::
   :hidden:

   variants/adam/index
   variants/sgd/index
