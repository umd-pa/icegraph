Mean
====

:doc:`Statistic <../index>` accumulating the per-column mean, ignoring NaN values.
It is a base statistic: the normalizer reads it as the centering offset, and other
quantities such as the coefficient of variation build on it. Its parallel merge
relies on the count statistics.

Selected as ``mean``. Takes no configuration.
