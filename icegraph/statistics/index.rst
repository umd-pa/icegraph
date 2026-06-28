Statistics
==========

The **statistics** subsystem computes the dataset-level summaries that the rest of
the framework relies on. A statistic is a streaming, per-column accumulator: it is
folded over the data, merges with other accumulators, and resolves to a value on
demand. The :doc:`stats processor <../data/processor/variants/stats/index>` uses
these to summarize a dataset during processing, and the engine's :doc:`normalizer
<../engine/components/normalizer/index>` reads the results back to build its
per-column scale and offset.

Usage
-----

A ``StatisticService`` is constructed with the set of statistics to track (by
name), folded over one or more arrays, and then queried for base or derived values.
Each column of a 2D array is accumulated independently.

.. code-block:: python

   import numpy as np

   from icegraph.statistics import StatisticService
   from icegraph.common.transforms import TransformSpace

   # track a set of statistics over the data
   stats = StatisticService(["mean", "m2", "min", "max", "finite_count"])

   # fold an array in (rows are samples, columns are tracked independently)
   stats.compute_from_array(np.asarray(data))

   # read a base statistic, optionally in a transform space or in a specific base
   mean = stats.get("mean")
   mean_log = stats.get("mean", space=TransformSpace.LOG)
   mean_log_base4 = stats.get("mean", space=TransformSpace.LOG, base=4)

   # read a derived quantity, computed from the tracked bases
   std = stats.std()
   value_range = stats.range()

Derived quantities require their underlying bases to be tracked.

Services computed separately (for example per shard or per process) combine
exactly, and serialize for storage and reload:

.. code-block:: python

   combined = StatisticService.merge([stats_a, stats_b])   # or: sum([stats_a, stats_b])

   struct = stats.to_struct()
   restored = StatisticService.from_struct(struct)

How it works
------------

A statistic service holds a bundle of statistics and accumulates them together. Each
statistic is tracked simultaneously in several transform spaces (linear, log, and
inverse-hyperbolic-sine), so downstream code can request, say, the mean in log space
without recomputation. Accumulators are mergeable, which lets statistics be computed
in parallel over shards and combined exactly, and they serialize into the dataset so
they travel with it.

Statistics fall into two groups. *Base* statistics are accumulated directly from the
data (the counts, the running mean, the Welford second moment, the minimum, and the
maximum). *Derived* quantities are computed from those bases on request; the service
exposes the standard deviation, variance, range, geometric mean, standard error,
coefficient of variation, RMS, and signal-to-noise ratio. Because some bases depend
on others when merging (the mean and second moment use the count statistics),
selecting a derived quantity generally means selecting the counts it builds on as
well.

Variants
--------

Running values:

* :doc:`mean <variants/mean>`: per-column mean.
* :doc:`m2 <variants/m2>`: Welford second moment, the basis for variance and
  standard deviation.
* :doc:`min <variants/minimum>`: per-column minimum.
* :doc:`max <variants/maximum>`: per-column maximum.

Counts:

* :doc:`total_count <variants/total_count>`: total number of values.
* :doc:`finite_count <variants/finite_count>`: number of finite values.
* :doc:`nan_count <variants/nan_count>`: number of NaN values.
* :doc:`zero_count <variants/zero_count>`: number of exact zeros.
* :doc:`positive_count <variants/positive_count>`: number of positive values.

Registering a new statistic
---------------------------

A statistic is a subclass of ``Statistic`` that declares a ``name``, the transform
``spaces`` it is tracked in, and its ``degree`` under linear rescaling, and
implements two operations: folding an array into the accumulator and merging two
accumulators. Register it with ``StatisticFactory``. Statistics take no
configuration.

.. code-block:: python

   from typing import ClassVar

   from icegraph.typing.common import ArrayF
   from icegraph.common.transforms import TransformSpace
   from icegraph.statistics.statistic import Statistic
   from icegraph.statistics.factory import StatisticFactory

   class MyStatistic(Statistic):
       name: ClassVar[str] = "my-statistic"
       spaces: ClassVar[tuple[TransformSpace, ...]] = TransformSpace.all()
       degree = 1

       def _compute(self, array: ArrayF) -> ArrayF:
           ...  # reduce one array to a per-column value

       @classmethod
       def _merge(cls, a, b, space) -> ArrayF:
           ...  # combine two accumulators in the given space

   StatisticFactory.register(MyStatistic)

.. toctree::
   :hidden:

   variants/mean
   variants/m2
   variants/minimum
   variants/maximum
   variants/total_count
   variants/finite_count
   variants/nan_count
   variants/zero_count
   variants/positive_count
