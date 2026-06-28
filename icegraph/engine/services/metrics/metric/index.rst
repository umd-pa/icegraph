Metric
======

A **metric** is a single evaluation measure computed by the :doc:`metrics service
<../index>`, such as an error or an accuracy. Metrics are selected by name in the
service's ``select`` list and reported per prediction head.

Usage
-----

Each entry of ``services.metrics.select`` selects a metric.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: rmse
           kwargs: {}
         - name: top-k-acc
           kwargs: { k: 1 }

How it Works
------------

A metric accumulates state incrementally: it starts from an empty accumulator,
folds each batch into it, and resolves the accumulator to a per-head value on
demand. Accumulators also merge across processes, so a metric is exact over the
full evaluation set and correct under distributed execution.

Variants
--------

* :doc:`MAE <variants/mae/index>`: mean absolute error.
* :doc:`MSE <variants/mse/index>`: mean squared error.
* :doc:`RMSE <variants/rmse/index>`: root mean squared error.
* :doc:`Top-K Accuracy <variants/top_k_acc/index>`: fraction of samples whose true
  class is within the top ``k`` scores.

Registering a new metric
------------------------

A metric is a subclass of ``Metric`` that declares a ``name`` and ``version`` and
expresses its accumulation over a freely chosen state type ``S`` through the
abstract methods below. Register it with ``MetricFactory``.

``optimum(self) -> float``
   The best attainable value, as a property (used for reporting direction).
``repr(self) -> str``
   A short string label for the metric.
``initial(self) -> S``
   Create an empty accumulator.
``update_state(self, state, out, target) -> S``
   Fold one batch into the accumulator and return it.
``combine(self, a, b) -> S``
   Merge two accumulators (associative, with ``initial()`` as identity).
``finalize(self, state) -> Tensor``
   Resolve the accumulator to a per-head value.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.services.metrics.metric import Metric, MetricFactory

   from .config import MyMetricConfig

   class MyMetric(Metric[MyMetricConfig, MyState]):
       name: ClassVar[str] = "my-metric"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyMetricConfig:
           return MyMetricConfig(**config)

       @property
       def optimum(self) -> float:
           ...

       def repr(self) -> str:
           ...

       def initial(self) -> MyState:
           ...

       def update_state(self, state: MyState, out: SegmentedTensor, target: SegmentedTensor) -> MyState:
           ...

       def combine(self, a: MyState, b: MyState) -> MyState:
           ...

       def finalize(self, state: MyState) -> Tensor:
           ...

   MetricFactory.register(MyMetric)

.. toctree::
   :hidden:

   variants/mae/index
   variants/mse/index
   variants/rmse/index
   variants/top_k_acc/index
