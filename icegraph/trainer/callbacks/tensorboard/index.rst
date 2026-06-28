TensorBoard
===========

:doc:`Trainer callback <../index>` that logs training, validation, and test
diagnostics to TensorBoard and serves them, so a run can be monitored in the browser
as it progresses. Logs are written under the run's output directory.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``port``
     - Port the TensorBoard server is served on.
     - int
     - required

.. code-block:: python

   trainer.register_callback(CallbackSpec(callback=TensorBoardCallback, kwargs={"port": 6006}))
