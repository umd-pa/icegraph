KNN Edge Builders
=================

The **k-nearest-neighbour edge builders** are the family of
:doc:`edge builders <../../index>` that connect every node to the ``k`` nodes
closest to it within its own graph.

How it works
------------

The family resolves ``neighbor_cols`` against the feature block to obtain the
coordinates each node occupies, then runs a batched neighbour search that uses the
batch vector to keep every edge inside a single graph. Where a graph holds fewer
than ``k + 1`` nodes, that graph simply contributes fewer edges.

Edges follow the orientation ``torch_geometric`` uses for message passing:
``edge_index[0]`` is the neighbour a message travels from, and ``edge_index[1]``
is the node it arrives at. Under the default ``source_to_target`` flow, each node
therefore aggregates over its own ``k`` nearest neighbours.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``k``
     - Number of nearest neighbours connected per node. Must be at least 1.
     - int
     - required
   * - ``neighbor_cols``
     - Logical feature column groups spanning the space the neighbour search runs
       in. Must name at least one group.
     - list[str]
     - required

Subclasses accept these options in addition to any of their own.

Variants
--------

* :doc:`Sum in Quadrature <variants/sum_in_quadrature/index>`: weight each edge by
  the quadrature sum of its endpoint differences.

Registering a new KNN edge builder
----------------------------------

A KNN edge builder is a subclass of ``KNNEdgeBuilder`` that declares a ``name``
and ``version`` and supplies the weight for an already-computed edge index. The
base implements ``build_index``, so a subclass implements only:

``build_attr(self, t, edge_index) -> Tensor``
   Return one attribute row per edge, with shape ``[E, ATTR]``.

A subclass that needs options beyond ``k`` and ``neighbor_cols`` declares a config
that extends the family's and overrides ``validate_config``.

.. code-block:: python

   # config.py
   from pydantic import Field

   from icegraph.engine.components.edges.variants.knn.config import Config

   class MyConfig(Config):
       ...

.. code-block:: python

   # plugin.py
   from typing import Any, ClassVar

   from torch import Tensor

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.edges import EdgeBuilderFactory
   from icegraph.engine.components.edges.variants.knn.plugin import KNNEdgeBuilder

   from .config import MyConfig

   class MyKNNVariant(KNNEdgeBuilder[MyConfig]):
       name: ClassVar[str] = "gaussian"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyConfig:
           return MyConfig(**config)

       def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
           ...  # per-edge weight, with shape [E, ATTR]

   EdgeBuilderFactory.register(MyKNNVariant)


.. toctree::
   :hidden:

   variants/sum_in_quadrature/index
