State
=====

The **state service** provides the process-level runtime context for a run: the
device that tensors and the model live on, the random seed that makes a run
reproducible, and, under distributed execution, the rank and world information that
identifies each process. Other services and components read this state rather than
querying the environment directly.

Usage
-----

Configured under ``services.state``.

.. code-block:: yaml

   services:
     state:
       seed: 2747

How it works
------------

On attach, the state service resolves the device (selecting the local accelerator
when one is available) and, when launched under a distributed environment, reads
the rank and world size so collective operations and data sharding are coordinated
across processes. Outside a distributed launch it falls back to a single-accelerator setup,
or CPU, in that order.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 60 10 10

   * - Option
     - Description
     - Type
     - Default
   * - ``seed``
     - Random seed used to initialize the run for reproducibility.
     - int
     - required
