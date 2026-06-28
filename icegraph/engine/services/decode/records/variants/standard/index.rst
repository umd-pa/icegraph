Standard
========

:doc:`Record decoder <../../index>` variant that extracts node features, targets,
graph connectivity, and auxiliary columns from records written by the data
pipeline in the standard schema. It is the counterpart to the standard attribute
decoder and the default choice for datasets produced by IceGraph.

Configuration
-------------

Selected as ``name: standard``. Takes no options.

.. code-block:: yaml

   services:
     decode:
       records:
         name: standard
         kwargs: {}
