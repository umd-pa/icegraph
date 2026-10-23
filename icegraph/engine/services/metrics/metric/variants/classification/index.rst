Classification Metrics
======================

The **classification metrics** are the family of :doc:`metrics <../../index>` for
categorical targets. They read each head's columns as scores over that head's
classes and its single target column as a class index, which is the layout the
:doc:`multiclass <../../../../../policy/variants/multiclass/index>` policy produces.

How it works
------------

Heads are ragged so the family walks them one
at a time rather than working across the packed row, pairing a head's scores with
its labels and rejecting a target block that is not one column per head.

The metrics here range from a confusion matrix to a binned score histogram,
and each declares its own state.
What they share is how the model output is read, and that splits them in two:

* **Arg-max metrics** look only at which class scored highest. They are invariant
  to whether the model emits logits, log-probabilities or probabilities, and take
  no configuration for it.
* **Distribution metrics** read the predicted probabilities themselves. These take
  a ``from_logits`` option, ``true`` by default.

Confusion-matrix metrics
------------------------

``ConfusionMetric`` pins the accumulator to one ``[K, K]`` matrix per head, indexed
``[true, predicted]``. Counting over two batches and adding is counting over their
union, so the monoid is trivial and folding across batches or merging across
processes is exact.

* :doc:`Macro-F1 <variants/macro_f1/index>`: F1 averaged with equal weight per class.
* :doc:`Macro-Recall <variants/macro_recall/index>`: recall averaged with equal
  weight per class.
* :doc:`Balanced Accuracy <variants/balanced_acc/index>`: accuracy as if every
  class were equally represented, optionally adjusted so chance scores ``0``. In
  its default form this is the same quantity as macro-recall.
* :doc:`Per-Class Recall <variants/per_class_recall/index>`: recall reported one
  value per class rather than one per head.
* :doc:`Cohen's Kappa <variants/cohen_kappa/index>`: agreement against what the
  class marginals would reach by chance.

A class the evaluation set never contains has an undefined score. These metrics
report it as ``nan`` per class and leave it out of any macro average, rather than
counting it as a zero that would drag the average down.

Other metrics
-------------

* :doc:`Top-K Accuracy <variants/top_k_acc/index>`: fraction of samples whose true
  class is within the top ``k`` scores. Arg-max style, but a rank count rather
  than a confusion matrix, and it runs across the whole packed row at once.
* :doc:`Cross-Entropy <variants/cross_entropy/index>`: mean negative
  log-probability of the true class.
* :doc:`Expected Calibration Error <variants/ece/index>`: gap between stated
  confidence and observed accuracy, over binned confidences.
* :doc:`AUPRC <variants/auprc/index>`: macro-averaged area under the
  precision-recall curve, over binned scores.

Registering a new classification metric
---------------------------------------

A metric that is a pure function of the confusion matrix subclasses
``ConfusionMetric`` and implements only ``reduce``, the function from one head's
matrix to that head's 1-D value. ``recall``, which returns per-class recall
alongside the support to mask undefined classes with, is available on the base.

.. code-block:: python

   # plugin.py
   from typing import ClassVar

   from torch import Tensor

   from icegraph.engine.services.metrics.metric import MetricFactory
   from icegraph.engine.services.metrics.metric.variants.classification.confusion import ConfusionMetric
   from icegraph.engine.services.metrics.metric.variants.classification.config import Config

   class Specificity(ConfusionMetric[Config]):
       name: ClassVar[str] = "specificity"
       version: ClassVar[int] = 1

       @property
       def optimum(self) -> float:
           return 1.0

       def repr(self) -> str:
           return "specificity"

       def reduce(self, confusion: Tensor) -> Tensor:
           ...  # 1-D value for this head, from its [K, K] matrix

   MetricFactory.register(Specificity)

Anything else subclasses ``ClassificationMetric`` directly, declaring its own
accumulator type and the full monoid over it.
The family still supplies the shared readers:

``heads(out, target)``
   Walk the heads, yielding ``(head, scores [B, K], labels [B])``, with the
   one-target-column-per-head invariant checked.
``probabilities(scores, *, from_logits)``
   Class probabilities for one head's scores.
``macro(values, mask)``
   Mean over the selected entries as a 1-element tensor, dropping the rest rather
   than counting them as zero.

A metric that takes no options inherits the family's empty ``Config``. One that
does declares its own config model and overrides ``validate_config`` alongside its
binding of the config type.

.. toctree::
   :hidden:

   variants/top_k_acc/index
   variants/macro_f1/index
   variants/macro_recall/index
   variants/balanced_acc/index
   variants/per_class_recall/index
   variants/cohen_kappa/index
   variants/cross_entropy/index
   variants/ece/index
   variants/auprc/index
