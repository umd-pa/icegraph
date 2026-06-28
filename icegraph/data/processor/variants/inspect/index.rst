Inspect
=======

:doc:`Processor <../../index>` that prints the contents of the active frame as it
passes through the pipeline. It performs no transformation and is intended as a
debugging aid for checking the shape and values of the data mid-pipeline.

Configuration
-------------

Selected as ``name: inspect``. Takes no options.

.. code-block:: yaml

   - name: inspect
     kwargs: {}
