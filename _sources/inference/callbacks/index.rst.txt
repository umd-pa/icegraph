Inference Callbacks
===================

**Inference callbacks** observe and extend an inference run, building on the engine
:doc:`callback <../../engine/callbacks/index>` machinery. They are the hook for
consuming a run's predictions: collecting outputs, writing them to disk, or producing
summary plots, without modifying the inference engine.

Registering
-----------

Callbacks are registered before the run with a ``CallbackSpec`` naming the callback
class and its keyword arguments:

.. code-block:: python

   from icegraph.inference.callbacks import CallbackSpec

   with BatchInference.from_yaml(config_path) as inference:
       inference.register_callback(CallbackSpec(callback=MyInferenceCallback, kwargs={}))
       inference.execute()

Writing a callback
------------------

A callback is a subclass of ``InferenceCallback`` that overrides the lifecycle hooks
it needs.

.. code-block:: python

   from icegraph.inference.callbacks import InferenceCallback
   from icegraph.inference.callbacks import context

   class MyInferenceCallback(InferenceCallback):
       def on_init(self, ctx: context.InitContext) -> None:
           ...

       def on_execute(self, ctx: context.ExecuteContext) -> None:
           ...

       def on_teardown(self, ctx: context.TeardownContext) -> None:
           ...
