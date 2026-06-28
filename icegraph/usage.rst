Usage
=====

This guide covers the basics of using IceGraph end to end: how a run is structured,
how configuration works, how to write the small driver scripts for each workflow,
and how to run distributed and extend the framework.

Introduction
------------

Every workflow follows the same shape. A short Python script builds one top-level
object from a YAML configuration and calls a single method to run it:

* :doc:`Processing <data/index>` builds a ``Pipeline`` and calls
  ``execute``.
* :doc:`Training <engine/index>` builds a ``Trainer`` engine and calls
  ``execute``.
* :doc:`Inference <engine/index>` builds a ``BatchInference`` engine and calls ``execute``.

The configuration does the heavy lifting: it names which plugins fill each slot and
supplies their options. The script only wires together the source data, the
configuration, and the output location. Because the objects are context managers,
run them inside a ``with`` block to guarantee resource cleanup.

Running under IceTray
---------------------

Any workflow that reads I3 files (the standard :doc:`I3 extractor
<data/extractor/variants/i3/index>`) depends on IceCube's IceTray and must be
launched through the provided shim, ``initicetray.sh``, in place of ``python3``:

.. code-block:: bash

   initicetray.sh process.py --source /path/to/i3 --config config/data-proc.yaml

Training and batch inference run on processed datasets and do not require the shim;
ordinary ``python`` from within the virtual environment is sufficient.

Configuration
-------------

Configuration is YAML, but both JSON and TOML are supported as alternatives. Wherever
the framework offers a choice, the config selects a plugin by name and passes its options
as ``kwargs``:

.. code-block:: yaml

   <slot>:
     name: <plugin-name>
     kwargs: { <option>: <value>, ... }

This pattern is described in full under :doc:`Plugin <common/plugins/index>`. Each
slot's reference page lists the plugins available for it and the options each
accepts. The three workflows use three top-level config shapes:

* **Processing** declares an ``extractor``, an ordered list of ``processors``, and a
  ``writer``.
* **Training** declares ``services``, a ``policy``, and ``components``, plus
  a few run-level keys (see :doc:`Trainer <trainer/index>`).
* **Inference** declares ``services`` and points ``model_path`` at a trained
  checkpoint; the model is restored from that checkpoint rather than configured.

Logging
-------

IceGraph provides a logging setup helper. Call it once near the start of a script
before building anything:

.. code-block:: python

   from icegraph.logging import configure_logging

   configure_logging(level="debug")          # console logging
   # configure_logging(level="info", log_file="run.log", json_logs=True)


You can also build your own logging helper should you need to.

Processing
----------

A processing script resolves the input files and configuration, then runs the
pipeline. The source may be a single file, a list of files, or a directory; the
output location is taken from the writer's configuration.

.. code-block:: python

   import argparse
   from pathlib import Path

   from icegraph.logging import configure_logging
   from icegraph.data import Pipeline


   def main() -> None:
       parser = argparse.ArgumentParser()
       parser.add_argument("-i", "--source", required=True, help="File, list, or directory")
       parser.add_argument("-c", "--config", required=True, help="Path to the pipeline config")
       args = parser.parse_args()

       configure_logging(level="info")

       with Pipeline.from_yaml(args.source, Path(args.config)) as pipeline:
           pipeline.execute()


   if __name__ == "__main__":
       main()

The matching configuration names the three stages:

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       gcd_path: /path/to/GCD.i3.zst
       include: [ features ]
       ml_suite: { ... }

   processors:
     - name: select
       kwargs: { key: features }
     - name: knn
       kwargs: { by: event_ids, col: dom_pos, out: [ edge_index, edge_attr ], k: 8 }
     - name: commit
       kwargs: { ids: event_ids, cols: [ features, edge_index, edge_attr ] }

   writer:
     name: lmdb
     kwargs:
       outdir: /path/to/output

Processing is independent per input file, so a large dataset can be processed by
launching the script as several parallel jobs.

Training
--------

A training script builds the :doc:`Trainer <trainer/index>` engine from a configuration and executes
it.

.. code-block:: python

   import argparse
   from pathlib import Path

   from icegraph.logging import configure_logging
   from icegraph.trainer import Trainer


   def main() -> None:
       parser = argparse.ArgumentParser()
       parser.add_argument("-c", "--config", required=True, help="Path to the training config")
       args = parser.parse_args()

       configure_logging(level="info")

       with Trainer.from_yaml(Path(args.config)) as trainer:
           trainer.execute()


   if __name__ == "__main__":
       main()

The configuration assembles the run from :doc:`services <engine/services/index>`, a
:doc:`policy <engine/policy/index>`, and :doc:`components <engine/components/index>`:

.. code-block:: yaml

   outdir: /path/to/output
   max_epochs: 10000
   val_interval: 5

   services:
     state:   { seed: 2747 }
     data:    { batch_size: 2048, chunk_size: 4096, buffer_size: 16384, num_workers: 8,
                prefetch_factor: 8, mp_context: fork, persistent_workers: true }
     decode:  { targets: [ bundle ], attrs: { name: standard, kwargs: {} },
                records: { name: standard, kwargs: {} } }
     record:  { source: [ /path/to/dataset ], reader: { name: lmdb, kwargs: {} },
                store: { name: lru-shard, kwargs: { cache_size: 32 } } }
     metrics: { select: [ { name: top-k-acc, kwargs: {} } ] }

   policy:
     name: multiclass
     kwargs: {}

   components:
     model:       { name: gcn, kwargs: { hidden_layers: 4, hidden_channels: 256 } }
     transformer: { name: standard, kwargs: { transforms: {} } }
     normalizer:  { name: zscore, kwargs: {} }
     optimizer:   { name: adam, kwargs: { lr: 0.0002, weight_decay: 0.00005 } }
     loss:        { name: cross-entropy, kwargs: {} }

Callbacks
~~~~~~~~~

:doc:`Callbacks <engine/callbacks/index>` observe a run without altering it
(console logging, metric plotting, checkpoint export, TensorBoard). Register them
before calling ``execute`` by passing a ``CallbackSpec`` naming the callback class
and its keyword arguments:

.. code-block:: python

   from icegraph.trainer.callbacks import CallbackSpec, ConsoleCallback, TensorBoardCallback

   with Trainer.from_yaml(config_path) as trainer:
       trainer.register_callback(CallbackSpec(callback=ConsoleCallback, kwargs={}))
       trainer.register_callback(CallbackSpec(callback=TensorBoardCallback, kwargs={}))
       trainer.execute()

Inference
---------

.. note::

   Currently only batched inference has been implemented. Realtime streaming inference is planned
   for a future update.

A batched inference script builds the ``BatchInference`` engine the same way:

.. code-block:: python

   import argparse
   from pathlib import Path

   from icegraph.logging import configure_logging
   from icegraph.inference import BatchInference


   def main() -> None:
       parser = argparse.ArgumentParser()
       parser.add_argument("-c", "--config", required=True, help="Path to the inference config")
       args = parser.parse_args()

       configure_logging(level="info")

       with BatchInference.from_yaml(Path(args.config)) as inference:
           inference.execute()


   if __name__ == "__main__":
       main()

The inference configuration restores the model from a checkpoint rather than
configuring components, so it names ``model_path`` and omits the ``components``
section:

.. code-block:: yaml

   outdir: /path/to/output
   model_path: /path/to/model.epoch_20.pt

   policy: ~

   services:
     state:  { seed: 2747 }
     data:   { batch_size: 1, chunk_size: 1, buffer_size: 1, num_workers: 6,
               prefetch_factor: 8, mp_context: fork, persistent_workers: true }
     decode: { features: [], auxiliary: [], attrs: { name: standard, kwargs: {} },
               records: { name: standard, kwargs: {} } }
     record: { source: [ /path/to/dataset ], reader: { name: lmdb, kwargs: {} },
               store: { name: lru-shard, kwargs: { cache_size: 32 } } }

.. warning::

   Running inference in distributed mode may currently drop chunks so that the
   sample count matches across ranks, which can lose data. Distributed execution is
   intended for training; run inference on a single process until this is resolved.

Distributed Execution
----------------------

Either engine can be run across multiple ranks by wrapping it in ``Distributed``,
without changing the configuration:

.. code-block:: python

   from icegraph.engine import Distributed
   from icegraph.trainer import Trainer

   with Distributed(Trainer).from_yaml(config_path) as trainer:
       trainer.execute()

The script is then launched internally as a multi-process job. The
:doc:`state service <engine/services/state/index>` reads environment
variables ("RANK", "WORLD", "LOCAL_RANK") to coordinate ranks.
Callbacks are registered the same way in distributed runs.

Extending the Framework
-----------------------

New behavior is added by writing a plugin and registering it. The general pattern is the following,
with additional detail provided on each slots reference page:

#. Subclass the slot's base class and declare a ``name`` and ``version``.
#. Implement the methods the base leaves abstract, and ``validate_config`` to parse
   the plugin's options.
#. Register the class with the slot's factory, for example
   ``NormalizerFactory.register(MyNormalizer)``.
#. Ensure the module that performs the registration is imported during setup, so the
   name is known before a configuration is loaded.

Once registered, the plugin is selectable by its ``name`` in the relevant config
slot exactly like a built-in.

.. warning::

   Registering a plugin with a name identical to one already present in the registry will overwrite it.
   This is intended behavior but may lead to unexpected results if not done with care.
