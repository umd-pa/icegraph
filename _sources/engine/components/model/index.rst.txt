Model
=====

The **model** is the :doc:`component <../index>` that maps an input graph to a
prediction. It is the graph neural network at the center of a run: it consumes the
per-node feature tensor together with the graph connectivity and produces a
graph-level output.

Usage
-----

The model occupies the ``components.model`` slot.

.. code-block:: yaml

   components:
     model:
       name: gcn
       kwargs:
         hidden_layers: 4
         hidden_channels: 256

How it Works
------------

A model receives a node feature tensor, an edge index, optional edge attributes,
and a batch assignment vector, and returns one row of outputs per graph. The
number of input channels is resolved from the :doc:`decode service
<../../services/decode/index>` and the number of output channels from the task
contract issued by the :doc:`policy <../../policy/index>`, so the same architecture
adapts to different feature sets and tasks without manual wiring. The base class
validates the output width and wraps the result; a variant supplies only the
forward computation.

Variants
--------

* :doc:`GCN <variants/gcn/index>`: graph convolutional network over the supplied
  edge weights.
* :doc:`GravNet <variants/gravnet/index>`: learns a latent neighborhood and
  aggregates over it.

Registering a new model
-----------------------

A model is a subclass of ``Model`` that declares a ``name`` and ``version``
and implements the forward computation:

``forward_pass(self, t, /, edge_index, edge_attr, batch) -> Tensor``
   Compute the graph-level output from the node features ``t`` and the graph
   structure, returning a tensor of shape ``[num_graphs, out_channels]``. The base
   exposes ``self.in_channels`` and ``self.out_channels`` for sizing layers and
   validates the returned width.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.model import Model, ModelFactory

   from .config import MyModelConfig

   class MyModel(Model[MyModelConfig]):
       name: ClassVar[str] = "my-model"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyModelConfig:
           return MyModelConfig(**config)

       def on_attach(self) -> None:
           ...  # build layers using self.in_channels and self.out_channels

       def forward_pass(self, t: SegmentedTensor, /, edge_index: Tensor, edge_attr: Tensor, batch: Tensor | None) -> Tensor:
           ...  # return shape [num_graphs, out_channels]

   ModelFactory.register(MyModel)

.. toctree::
   :hidden:

   variants/gcn/index
   variants/gravnet/index
