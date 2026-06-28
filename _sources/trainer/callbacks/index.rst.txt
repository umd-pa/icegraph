Trainer Callbacks
=================

**Trainer callbacks** observe and extend a training run through the trainer's
lifecycle, building on the engine :doc:`callback <../../engine/callbacks/index>`
machinery. They handle reporting and side effects (console output, checkpoint
export, TensorBoard logging, validation plots) without altering the training loop.

Registering
-----------

A callback is registered before the run by passing a ``CallbackSpec`` that names the
callback class and its keyword arguments:

.. code-block:: python

   from icegraph.trainer.callbacks import CallbackSpec, ConsoleCallback, ExportCallback

   with Trainer.from_yaml(config_path) as trainer:
       trainer.register_callback(CallbackSpec(callback=ConsoleCallback, kwargs={}))
       trainer.register_callback(CallbackSpec(callback=ExportCallback, kwargs={"save_interval": 10}))
       trainer.execute()

The trainer exposes lifecycle hooks a callback can implement, including run start and
teardown, the beginning and end of each epoch, of each batch, and of the training,
validation, and test phases.

Built-in callbacks
------------------

* :doc:`Console <console/index>`: live console UI for training progress.
* :doc:`Export <exporters/index>`: periodic model checkpointing.
* :doc:`TensorBoard <tensorboard/index>`: scalar and diagnostic logging to
  TensorBoard.
* :doc:`Plotters <plotters/index>`: validation and evaluation plots (parity, bias,
  confusion matrix, ROC, and more).

Writing a callback
------------------

A custom callback is a subclass of ``TrainerCallback`` that overrides the lifecycle
hooks it cares about. It is registered with a ``CallbackSpec`` exactly like a
built-in.

.. code-block:: python

   from icegraph.trainer.callbacks import TrainerCallback
   from icegraph.trainer.callbacks import context

   class MyCallback(TrainerCallback):
        def on_init(self, ctx: context.InitContext) -> None:
            ...

        def on_execute(self, ctx: context.ExecuteContext) -> None:
            ...

        def on_epoch_begin(self, ctx: context.EpochBeginContext) -> None:
            ...

        def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
            ...

        def on_batch_begin(self, ctx: context.BatchBeginContext) -> None:
            ...

        def on_batch_end(self, ctx: context.BatchEndContext) -> None:
            ...

        def on_train_begin(self, ctx: context.TrainBeginContext) -> None:
            ...

        def on_train_end(self, ctx: context.TrainEndContext) -> None:
            ...

        def on_validation_begin(self, ctx: context.ValidationBeginContext) -> None:
            ...

        def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
            ...

        def on_test_begin(self, ctx: context.TestBeginContext) -> None:
            ...

        def on_test_end(self, ctx: context.TestEndContext) -> None:
            ...

        def on_teardown(self, ctx: context.TeardownContext) -> None:
            ...
.. toctree::
   :hidden:

   console/index
   exporters/index
   tensorboard/index
   plotters/index
