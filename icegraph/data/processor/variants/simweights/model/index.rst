Flux Model
==========

The **flux model** supplies the physical spectrum that the :doc:`simweights
<../index>` processor reweights simulated events onto. It is selected by name under
the processor's ``flux`` option.

Usage
-----

.. code-block:: yaml

   flux:
     name: gaisser-h4a
     kwargs: {}

Variants
--------

* :doc:`Gaisser H4a <variants/gaisser_h4a/index>`: the Gaisser H4a cosmic-ray flux
  model.
* :doc:`Power Law <variants/power_law/index>`: a configurable power-law flux.

Registering a new flux model
----------------------------

A flux model is a subclass of ``FluxModel`` that declares a ``name`` and
``version`` and implements the abstract methods below. Register it with
``FluxModelFactory``.

``build(self) -> None``
   One-time setup.
``__call__(self, *args, **kwargs) -> ArrayLike``
   Evaluate the flux; called by ``simweights`` during weight computation.

.. code-block:: python

   from typing import Any, ClassVar

   from numpy.typing import ArrayLike

   from icegraph.data.processor.variants.simweights.model import FluxModel, FluxModelFactory

   from .config import MyFluxConfig

   class MyFlux(FluxModel[MyFluxConfig]):
       name: ClassVar[str] = "my-flux"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyFluxConfig:
           return MyFluxConfig(**config)

       def build(self) -> None:
           ...

       def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> ArrayLike:
           ...

   FluxModelFactory.register(MyFlux)

.. toctree::
   :hidden:

   variants/gaisser_h4a/index
   variants/power_law/index
