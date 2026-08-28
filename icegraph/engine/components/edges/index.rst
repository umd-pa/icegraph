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
       name: sum-in-quadrature
       kwargs:
         k: 10
         neighbor_cols: [ dom_pos ]
         weight_cols: [ dom_pos ]

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

The component runs **before** the transformer and the normalizer. Both rescale
feature columns independently of one another, which would distort the metric a
neighbour search operates in, and a detector whose geometry is anisotropic would
yield a materially different neighbour set once each axis had been standardized
separately. The engine calls the edge builder while the feature block still holds
raw values, and passes the result to the model alongside the transformed features.

Subclasses
----------

.. toctree::
   :maxdepth: 2

   variants/knn/index

Variants
--------

The edges slot has no variants registered directly on the base;
every selectable edge builder is provided by one of the subclasses above.

Registering a new edge builder
------------------------------

An edge builder is a subclass of ``EdgeBuilder`` that declares a ``name`` and
``version``, validates its configuration, and supplies the connectivity in two
parts. The base class validates the shapes and dtypes of both:

``build_index(self, t, batch) -> Tensor``
   Return the edge index for the feature block ``t``, with shape ``[2, E]`` and
   dtype ``long``. Use ``batch`` to keep edges inside a single graph.
``build_attr(self, t, edge_index) -> Tensor``
   Return one attribute row per edge, with shape ``[E, ATTR]``.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor
   from pydantic import BaseModel

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

       def build_index(self, t: SegmentedTensor, batch: Tensor) -> Tensor:
           ...  # return the edge index, shape [2, E], dtype long

       def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
           ...  # return the edge attributes, shape [E, ATTR]

   EdgeBuilderFactory.register(MyEdgeBuilder)

.. code-block:: yaml

   components:
     edges:
       name: my-edge-builder
       kwargs: { }
