Regression Metrics
==================

The **regression metrics** are the family of :doc:`metrics <../../index>` for
continuous targets. Every one of them is a mean over elementwise residuals
between the model output and the target:

.. code-block:: text

   value = resolve( mean( residual(out - target) ) )

How it works
------------

The family keeps a single accumulator per head (the summed residual and the
count of elements folded into it). Both are exactly additive, so batches fold in
by addition and shards merge the same way, and the epoch value does not depend on
how the evaluation set happened to divide.

Residuals are taken columnwise across the whole packed row and then scattered
into per-head totals, so heads of any width cost one pass rather than one pass
each.

The ``resolve`` step exists because not every metric reports the mean itself. A
transform that does not distribute over addition
cannot be folded into the accumulator without changing the answer, so it is
applied once, after the full mean is known.

Variants
--------

* :doc:`MAE <variants/mae/index>`: mean of ``|out - target|``.
* :doc:`MSE <variants/mse/index>`: mean of ``(out - target)^2``.
* :doc:`RMSE <variants/rmse/index>`: the square root of the mean of
  ``(out - target)^2``, back in target units.

All three are best at ``0``, which the family reports as the optimum.

Registering a new regression metric
-----------------------------------

A regression metric is a subclass of ``RegressionMetric`` that declares a ``name``
and ``version`` and supplies the residual it accumulates. The base implements the
whole monoid (``initial``, ``update_state``, ``combine`` and ``finalize``) plus
the ``optimum``, so a subclass implements only:

``residual(self, diff) -> Tensor``
   The elementwise error to accumulate, given ``out - target``. ``diff`` is a
   fresh tensor and may be rewritten in place.
``resolve(self, mean) -> Tensor``
   Optional. Map the per-head mean residual to the reported value. Defaults to the
   mean itself.
``repr(self) -> str``
   A short string label for the metric.

A metric that takes no options inherits the family's empty ``Config``. One that
does declares its own config model and overrides ``validate_config`` alongside its
binding of the config type.

.. code-block:: python

   # plugin.py
   from typing import ClassVar

   from torch import Tensor

   from icegraph.engine.services.metrics.metric import MetricFactory
   from icegraph.engine.services.metrics.metric.variants.regression.plugin import RegressionMetric
   from icegraph.engine.services.metrics.metric.variants.regression.config import Config

   class MaxError(RegressionMetric[Config]):
       name: ClassVar[str] = "max-error"
       version: ClassVar[int] = 1

       def repr(self) -> str:
           return "max_error"

       def residual(self, diff: Tensor) -> Tensor:
           ...  # elementwise error to accumulate

   MetricFactory.register(MaxError)

.. toctree::
   :hidden:

   variants/mae/index
   variants/mse/index
   variants/rmse/index
