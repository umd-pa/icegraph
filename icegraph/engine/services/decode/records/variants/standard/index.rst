Standard
========

:doc:`Record decoder <../../index>` variant that decodes node features, targets,
graph connectivity, and auxiliary columns from blocks written by the data
pipeline in the standard columnar schema: each key is one column of the block,
ragged columns carry per-record row offsets. It is the counterpart to the
standard attribute decoder and the default choice for datasets produced by
IceGraph.

Configuration
-------------

Selected as ``name: standard``. Takes no options.

.. code-block:: yaml

   services:
     decode:
       records:
         name: standard
         kwargs: {}
