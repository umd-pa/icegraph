Inference
=========

The **inference** engine applies a trained model to new data to produce predictions.

.. note::

   Currently only batched inference has been implemented. Realtime streaming inference is planned
   for a future update.

Usage
-----

A batch inference engine is built from a configuration and run with ``execute``; see
the :doc:`usage guide <../usage>` for a complete script.

.. code-block:: python

   from icegraph.inference import BatchInference

   with BatchInference.from_yaml(config_path) as inference:
       inference.execute()

Configuration
-------------

The inference configuration restores its model from a checkpoint rather than
configuring it from scratch:

.. list-table::
   :header-rows: 1
   :widths: 18 60 12 10

   * - Option
     - Description
     - Type
     - Default
   * - ``outdir``
     - Directory where predictions and outputs are written.
     - path
     - required
   * - ``model_path``
     - Path to the trained model checkpoint to run.
     - path
     - required

It uses the :doc:`state <../engine/services/state/index>`, :doc:`record
<../engine/services/record/index>`, :doc:`decode <../engine/services/decode/index>`,
and :doc:`data <../engine/services/data/index>` services, and restores the model,
normalizer, and transformer :doc:`components <../engine/components/index>` from the
checkpoint. No :doc:`policy <../engine/policy/index>` is required.

How it Works
------------

The engine loads the checkpoint, reconstructing the model and its associated
components from the stored weights and buffers, then streams the dataset through
the model and produces the predictions.

.. warning::

   Running inference in distributed mode may currently drop chunks so that the sample
   count matches across ranks, which can lose data. Run inference on a single process
   until this is resolved.

.. toctree::
   :maxdepth: 2

   callbacks/index
