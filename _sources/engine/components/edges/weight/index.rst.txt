Edge Weight
===========

An **edge weight** computes edge attributes for an edge index that an
:doc:`edge builder <../index>` has already built. An edge builder can hold any
number of edge weights, selected by name in its ``weights`` list.

Usage
-----

Each entry of ``components.edges.kwargs.weights`` selects an edge weight.

.. code-block:: yaml

   components:
     edges:
       name: knn
       kwargs:
         k: 8
         neighbor_cols: [ dom_x, dom_y, dom_z ]
         weights:
           - name: gaussian
             kwargs: { cols: [ dom_x, dom_y, dom_z ], sigma: 50.0 }
           - name: gaussian
             kwargs: { cols: [ dom_x, dom_y, dom_z ], sigma: 125.0 }

How it Works
------------

Every edge weight receives the same feature block and the same edge index, and
returns a block of shape ``[E, W]``, one row per edge. The edge builder
concatenates these blocks along dim 1 in the order they are listed, so the model
receives edge attributes of shape ``[E, ATTR]``, where ``ATTR`` is the sum of
every ``W``.

A model that reads edge attributes must be configured to accept the right number of edges.
For example, the :doc:`GINE <../../model/variants/gine/index>` model's ``edge_channels`` must equal
``ATTR``.

Edge weights run before the transformer and the normalizer, so they see the raw
feature values.

Variants
--------

* :doc:`Sum in Quadrature <variants/sum_in_quadrature/index>`: weight each edge by
  the quadrature sum of its endpoint differences.
* :doc:`Gaussian <variants/gaussian/index>`: weight each edge by a gaussian of the
  distance between its endpoints.

Registering a new edge weight
-----------------------------

An edge weight is a subclass of ``EdgeWeight`` that declares a ``name`` and
``version`` and implements:

``build_attr(self, t, edge_index) -> Tensor``
   Return one attribute row per edge, with shape ``[E, W]``. The base class verifies
   this shape.

Register it with ``EdgeWeightFactory``.

.. code-block:: python

   from typing import Any, ClassVar

   from torch import Tensor
   from pydantic import BaseModel

   from icegraph.common.tensors import SegmentedTensor
   from icegraph.engine.components.edges.weight import EdgeWeight, EdgeWeightFactory

   class MyWeightConfig(BaseModel):
       cols: list[str]

   class MyWeight(EdgeWeight[MyWeightConfig]):
       name: ClassVar[str] = "my-weight"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyWeightConfig:
           return MyWeightConfig(**config)

       def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
           ...  # per-edge weight, with shape [E, W]

   EdgeWeightFactory.register(MyWeight)

.. code-block:: yaml

   components:
     edges:
       name: knn
       kwargs:
         k: 8
         neighbor_cols: [ dom_x, dom_y, dom_z ]
         weights:
           - name: my-weight
             kwargs: { cols: [ dom_x, dom_y, dom_z ] }

.. toctree::
   :hidden:

   variants/sum_in_quadrature/index
   variants/gaussian/index
