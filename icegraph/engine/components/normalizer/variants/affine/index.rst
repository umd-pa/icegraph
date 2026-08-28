Affine Normalizers
==================

The **affine normalizers** are the family of :doc:`normalizers <../../index>` that
rescale each feature column with a linear map:

.. code-block:: text

   normalized = (value - offset) * scale

How it works
------------

For each feature column, an affine normalizer computes its offset and scale from
that column's training-split statistics (mean, standard deviation, minimum, range,
and so on) evaluated in the column's transform space. The resulting per-column
values are cached as buffers, so the derivation runs once and is then carried with
the checkpoint. Scales are computed defensively, with the divisor floored away
from zero, so a constant or near-constant column cannot produce an unbounded
transform.

Variants
--------

* :doc:`ZScore <variants/zscore/index>`: offset by the mean, scale by the inverse
  standard deviation.
* :doc:`Unit Variance <variants/unit_variance/index>`: no offset, scale by the
  inverse standard deviation.
* :doc:`Mean Centering <variants/mean_centering/index>`: offset by the mean, no
  scaling.
* :doc:`MinMax <variants/minmax/index>`: offset by the minimum, scale by the
  inverse range.

Registering a new affine normalizer
-----------------------------------

An affine normalizer is a subclass of ``AffineNormalizer`` that declares a
``name`` and ``version`` and supplies the two quantities the linear map needs per
column. The base implements ``normalize`` and its inverse, resolves these
quantities from the training-split statistics, and caches them as buffers, so a
subclass implements only:

``_build_offset(self, stats, space, base, column_index) -> float``
   Return the offset subtracted from the given column.
``_build_scale(self, stats, space, base, column_index) -> float``
   Return the multiplicative scale applied to the given column.

.. code-block:: python

   # plugin.py
   from typing import ClassVar

   from icegraph.common.transforms import TransformSpace
   from icegraph.statistics import StatisticService
   from icegraph.engine.components.normalizer import NormalizerFactory
   from icegraph.engine.components.normalizer.variants.affine.plugin import AffineNormalizer

   class RobustScale(AffineNormalizer):
       name: ClassVar[str] = "robust-scale"
       version: ClassVar[int] = 1

       def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
           ...  # per-column offset from statistics

       def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
           ...  # per-column scale from statistics

   NormalizerFactory.register(RobustScale)


.. toctree::
   :hidden:

   variants/zscore/index
   variants/unit_variance/index
   variants/mean_centering/index
   variants/minmax/index
