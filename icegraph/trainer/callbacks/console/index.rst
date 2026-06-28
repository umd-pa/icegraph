Console
=======

:doc:`Trainer callback <../index>` that renders a live console interface for a
training run, showing progress through epochs and batches and the current training,
validation, and test status. On terminals that do not support the rich interface
(such as some IDE consoles) it falls back to plain printed output.

Registered with no arguments:

.. code-block:: python

   trainer.register_callback(CallbackSpec(callback=ConsoleCallback, kwargs={}))
