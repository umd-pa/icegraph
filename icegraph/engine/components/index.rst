Components
==========

**Components** are the configurable building blocks of the model that an engine
assembles for a run. They are selected under the ``components`` section of the
engine configuration, one per kind:

* **model**, the GNN itself,
* **transformer**, which applies per-feature value transforms such as a
  logarithmic or inverse-hyperbolic-sine mapping,
* **normalizer**, which rescales feature and target tensors into a conditioned
  range,
* **edges**, which builds graph connectivity on the accelerator from node feature
  columns,
* **optimizer**, the optimization algorithm that updates model weights, and
* **loss**, the objective minimized during training.

A component is the composition of two objects: it is a PyTorch ``Module`` and an
IceGraph :doc:`plugin <../../common/plugins/index>` at once. From the module it
inherits parameters, buffers, device placement, and checkpoint serialization;
from the plugin it inherits the identity, configuration, and lifecycle model.

A component tracks the device it currently occupies. On attach, a component may
be checked against any contract issued by the :doc:`policy <../policy/index>`.

.. toctree::
   :maxdepth: 2
   :caption: Component Categories

   model/index
   transformer/index
   normalizer/index
   edges/index
   optimizer/index
   loss/index
