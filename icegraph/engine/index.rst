Engine
======

The **engine** is the subsystem that executes a run. The two currently implemented concrete engines are the :doc:`trainer
<../trainer/index>`, which fits a model, and the :doc:`batch inference engine
<../inference/index>`, which applies a trained model to new data. Both share the
structure described here.

An engine is constructed from a configuration file (YAML, JSON, or TOML). From its
configuration it lazily assembles four subsystems:

* **Services** supply shared, run-scoped capabilities and data.
* **Policy** defines the task and the contracts each component must satisfy.
* **Components** are the configurable building blocks of the model.
* **Callbacks** observe and extend the run through lifecycle hooks.

A run is started by calling ``execute``. Engines can be wrapped for distributed,
multi-rank execution without changing their configuration.

.. toctree::
   :maxdepth: 2
   :caption: Engine Subsystems

   services/index
   policy/index
   components/index
   callbacks/index
