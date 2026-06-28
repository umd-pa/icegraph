Trainer
=======

The **trainer** is the concrete :doc:`engine <../engine/index>` that fits a model to
a dataset. It manages the training, validation, and testing lifecycle: it iterates
the training data in epochs, periodically evaluates on the validation split,
computes metrics, and drives checkpointing and reporting through callbacks.

Usage
-----

A trainer is built from a configuration and run with ``execute``; see the
:doc:`usage guide <../usage>` for a complete script.

.. code-block:: python

   from icegraph.trainer import Trainer

   with Trainer.from_yaml(config_path) as trainer:
       trainer.execute()

Configuration
-------------

The trainer configuration is an engine configuration (``services``, ``policy``, and
``components``) plus a few run-level keys:

.. list-table::
   :header-rows: 1
   :widths: 18 60 12 10

   * - Option
     - Description
     - Type
     - Default
   * - ``outdir``
     - Directory where checkpoints, logs, and plots are written.
     - path
     - required
   * - ``max_epochs``
     - Maximum number of training epochs.
     - int
     - required
   * - ``val_interval``
     - Number of epochs between validation passes.
     - int
     - required

It uses the :doc:`state <../engine/services/state/index>`, :doc:`record
<../engine/services/record/index>`, :doc:`data <../engine/services/data/index>`,
:doc:`metrics <../engine/services/metrics/index>`, and :doc:`decode
<../engine/services/decode/index>` services, a :doc:`policy <../engine/policy/index>`,
and the full set of :doc:`components <../engine/components/index>` (model,
transformer, normalizer, optimizer, loss).

How it works
------------

Each epoch runs the training split, updating the model through the optimizer against
the loss. Every ``val_interval`` epochs the trainer runs the validation split, where
it computes metrics and serves predictions to callbacks but performs no weight
updates. A final test pass evaluates the trained model. The run is reproducible
through the state service's seed, and can be scaled across ranks with the
``Distributed`` wrapper.

.. toctree::
   :maxdepth: 2

   callbacks/index
