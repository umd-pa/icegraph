Edges
=====

The **edge builder** is the :doc:`component <../index>` that constructs graph
connectivity on the accelerator from node feature columns.

Usage
-----

The edge builder occupies the ``components.edges`` slot.

.. code-block:: yaml

   components:
     edges:
       name: knn
       kwargs:
         k: 10
         neighbor_cols: [ dom_pos ]
         weights:
           - name: sum-in-quadrature
             kwargs: { cols: [ dom_pos ] }
           - name: gaussian
             kwargs: { cols: [ dom_pos ], sigma: 125.0 }

Columns are named by *logical group*, using the names the
:doc:`compress <../../../data/processor/variants/compress/index>` processor
records when it packs the feature block. A group that spans several physical
columns, such as a three-component position, is selected as a unit.

How it Works
------------

An edge builder receives the feature block as a segmented tensor together with the
batch vector, and returns an edge index of shape ``[2, E]`` and edge attributes of
shape ``[E, ATTR]``. The batch vector confines connectivity to within each graph,
so no edge ever crosses between two events in the same batch.

The edge builder itself computes only the edge index.
The edge attributes come from the :doc:`edge weights <weight/index>` listed under
``weights``. Each weight runs on the same edge index and returns its own
``[E, W]`` block, and the blocks are concatenated along dim 1 in the order they are
listed. ``ATTR`` is therefore the sum of the widths of every listed weight. The
same weight may be listed more than once. With no weights listed, the edge attributes have shape ``[E, 0]``.

The component runs **before** the transformer and the normalizer. Both rescale
feature columns independently of one another, which would distort the metric a
neighbour search operates in, and a detector whose geometry is anisotropic would
yield a materially different neighbour set once each axis had been standardized
separately. The engine calls the edge builder while the feature block still holds
raw values, and passes the result to the model alongside the transformed features.

Configuration
-------------

Every edge builder accepts this option in addition to its own.

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``weights``
     - Edge weights to run on the edge index, each given as ``name`` and
       ``kwargs``. Their outputs are concatenated in this order.
     - list[EdgeWeightSelection]
     - ``[]``

Sub-slots
---------

.. toctree::
   :maxdepth: 2

   weight/index

Subclasses
----------

.. toctree::
   :maxdepth: 2

   variants/knn/index

Variants
--------

* :doc:`KNN <variants/knn/index>`: connect every node to its ``k`` nearest
  neighbours within its own graph.

Registering a new edge builder
------------------------------

An edge builder is a subclass of ``EdgeBuilder`` that declares a ``name`` and
``version``, validates its configuration, and supplies the edge index. The base
class builds the configured weights, runs them, and validates the shapes and dtypes
of the result, so a subclass implements only:

``build_index(self, t, batch, num_graphs) -> Tensor``
   Return the edge index for the feature block ``t``, with shape ``[2, E]`` and
   dtype ``long``. Use ``batch`` to keep edges inside a single graph.

Its config must extend ``EdgeBuilderConfig``. The
weights are built before ``build()`` runs, and are available there as
``self.weights``.

.. code-block:: python

   # config.py
   from icegraph.engine.components.edges import EdgeBuilderConfig

   class MyConfig(EdgeBuilderConfig):
       ...

.. code-block:: python

   # plugin.py
   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.edges import EdgeBuilder, EdgeBuilderFactory

   from .config import MyConfig

   class MyEdgeBuilder(EdgeBuilder[MyConfig]):
       name: ClassVar[str] = "my-edge-builder"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyConfig:
           return MyConfig(**config)

       def build(self) -> None:
           ...  # one-time setup, e.g. registering buffers

       def build_index(self, t: SegmentedTensor, batch: Tensor, num_graphs: int | None) -> Tensor:
           ...  # return the edge index, shape [2, E], dtype long

   EdgeBuilderFactory.register(MyEdgeBuilder)

.. code-block:: yaml

   components:
     edges:
       name: my-edge-builder
       kwargs:
         weights:
           - name: gaussian
             kwargs: { cols: [ dom_pos ], sigma: 125.0 }
