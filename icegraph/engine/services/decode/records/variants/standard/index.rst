Standard
========

:doc:`Record decoder <../../index>` variant that decodes node features, targets,
graph connectivity, and auxiliary columns from blocks written by the data
pipeline in the standard columnar schema: each key is one column of the block,
ragged columns carry per-record row offsets. It is the counterpart to the
standard attribute decoder and the default choice for datasets produced by
IceGraph.

It is also the decoder for IceCube simulation, and is the only part of the engine
that knows about ``simweights``: the weights role is decoded by building the
generation surface and weighting each record against a flux.

Weighting
---------

The :doc:`i3-simulation <../../../../../../data/processor/variants/simulation/index>`
processor does not store a weight. It stores the per-event generation quantities
as a column and the generating file's simulation metadata as a local attribute, so the
mixture is chosen when the data is read rather than when it is written.

On load this decoder builds the surface of each simulation set with simweights,
over the set's loaded files as one with ``nfiles`` counting them, sums the sets
of each type, weights the generation columns of each block against them, and
returns the final weights, one per record. Shards without a simulation are inferred to be real
data and have no weights.

The weights role reads the generation column through the decode service's
``keymap.weights``, which defaults to ``generation``.

Mixed simulation
~~~~~~~~~~~~~~~~

A source may mix simulation types, for instance CORSIKA files alongside NuGen
ones. The sets of each type are summed into their own surface,
each surface is weighted against its own flux, and each record is routed to one of
them. Configure one flux per type.

Routing is per record because a block spans shards. The ``i3-simulation`` processor
writes a sim code naming the set into the generation columns of every event and
declares, in the same shard's attributes, what type that code stands for and which
column carries it, so the decoder learns the mapping from the data rather than from
a convention shared with the writer.

.. note::

   Real data and simulation cannot be loaded together.

Configuration
-------------

Selected as ``name: standard``.

.. list-table::
   :header-rows: 1
   :widths: 15 57 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``flux``
     - Flux model each simulation type is weighted against, keyed by the simulation
       that generated it (``corsika``, ``nugen``). One entry is needed per type
       present in the data.
     - mapping
     - ``{}``

Each entry selects a model from one of the two packages that ship them:

.. list-table::
   :header-rows: 1
   :widths: 15 57 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``source``
     - Package the model comes from: ``simweights`` for the cosmic ray fluxes,
       ``nuflux`` for the atmospheric neutrino fluxes.
     - str
     - required
   * - ``name``
     - Model name as that package spells it, e.g. ``GaisserH4a`` or
       ``honda2006``.
     - str
     - required
   * - ``kwargs``
     - Constructor arguments for a ``simweights`` model, property assignments for
       a ``nuflux`` one.
     - mapping
     - ``{}``

.. code-block:: yaml

   services:
     decode:
       records:
         name: standard
         kwargs:
           flux:
             corsika:
               source: simweights
               name: GaisserH4a
               kwargs: { }

A source mixing CORSIKA and NuGen files (for example) configures both, and each is weighted
against its own model:

.. code-block:: yaml

   flux:
     corsika:
       source: simweights
       name: GaisserH4a
       kwargs: { }

     nugen:
       source: nuflux
       name: honda2006
       kwargs: { }

.. note::

   ``nuflux`` is not a pip dependency; it comes from the IceTray environment on
   CVMFS, and is imported only when a ``nuflux`` model is configured.
