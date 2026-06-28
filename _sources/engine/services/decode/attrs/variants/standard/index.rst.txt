Standard
========

:doc:`Attribute decoder <../../index>` variant that reads the attributes
written by the data pipeline in the standard schema: column names, per-column statistics,
and the observed label values. It is the counterpart to the standard record decoder and the default
choice for datasets produced by IceGraph.

Configuration
-------------

Selected as ``name: standard``. Takes no options.

.. code-block:: yaml

   services:
     decode:
       attrs:
         name: standard
         kwargs: {}
