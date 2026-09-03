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

A metric may resolve to more than one number per head. Most report a single
value, but per-class recall (for example) reports one per class, so a head's slot is a 1-D
tensor whose width the metric chooses and then keeps fixed for the run.

Subclasses
----------

.. toctree::
   :maxdepth: 2

   variants/regression/index
   variants/classification/index

The two families differ in what they read. A regression metric compares the model
output against a continuous target columnwise, while a classification metric reads a
head's columns as class scores and its single target column as a class index. A
run should select from the family matching its :doc:`policy
<../../../policy/index>`. Pointing a classification metric at continuous targets
raises rather than reporting a meaningless number.

Variants
--------

The metric slot has no variants registered directly on the base; every selectable
metric is provided by one of the subclasses above.

Registering a new metric
------------------------

Most new metrics belong in one of the families above, which supply the
accumulator and leave only the part that differs. A metric that fits neither is a
subclass of ``Metric`` that declares a ``name`` and ``version`` and expresses its
accumulation over a freely chosen state type ``S`` through the abstract methods
below. Register it with ``MetricFactory``.

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
``finalize(self, state) -> HeadValues``
   Resolve the accumulator to per-head values, one tuple entry per head.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.services.metrics.metric import Metric, MetricFactory, HeadValues

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

       def finalize(self, state: MyState) -> HeadValues:
           ...

   MetricFactory.register(MyMetric)
