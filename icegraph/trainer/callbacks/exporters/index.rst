Export
======

:doc:`Trainer callback <../index>` that periodically writes model checkpoints to a
``models`` directory under the run's output directory, so training can be resumed and
trained models can be used for :doc:`inference <../../../inference/index>`.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``save_interval``
     - Number of epochs between checkpoint writes.
     - int
     - ``10``

.. code-block:: python

   trainer.register_callback(CallbackSpec(callback=ExportCallback, kwargs={"save_interval": 10}))
