Multiclass
==========

:doc:`Policy <../../index>` variant for multiclass classification. It treats each
prediction head as a discrete label and configures the run accordingly: the model
output is interpreted as per-class scores, the targets are integer class indices,
and targets are not normalized.

The set of classes is discovered from the dataset. The policy reads the distinct
target values recorded across the dataset's files and sizes each head's output to
cover them, so the number of output channels follows the data rather than being
configured by hand.

Configuration
-------------

Selected as ``name: multiclass``. Takes no options.

.. code-block:: yaml

   policy:
     name: multiclass
     kwargs: {}
