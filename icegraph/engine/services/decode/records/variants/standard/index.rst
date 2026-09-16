Standard
========

:doc:`Record decoder <../../index>` variant that decodes node features, targets,
graph connectivity, and auxiliary columns from blocks written by the data
pipeline in the standard columnar schema: each key is one column of the block,
ragged columns carry per-record row offsets. It is the counterpart to the
standard attribute decoder and the default choice for datasets produced by
IceGraph.

It is also the decoder for IceCube simulation, and is the only part of the engine
that knows about ``simweights``: the weights role is decoded by rebuilding the
generation surface and weighting each record against a flux.

Weighting
---------

The :doc:`i3-simweight <../../../../../../data/processor/variants/simweight/index>`
processor does not store a weight. It stores the per-event generation quantities
as a column and the generating file's surface as a local attribute, so the
mixture is chosen when the data is read rather than when it is written.

On load this decoder sums the surfaces of every shard in the dataset,
rebuilds the weighter over the generation columns of each block,
and returns the final weights, one per record.

The weights role reads the generation column through the decode service's
``keymap.weights``, which defaults to ``generation``.

Mixed simulation
~~~~~~~~~~~~~~~~

A source may mix simulation types, for instance CORSIKA files alongside NuGen
ones. The shards of each type are summed into their own surface,
each surface is weighted against its own flux, and each record is routed to one of
them. Configure one flux per type.

Routing is per record because a block spans shards. The ``i3-simweight`` processor
writes a source code into the generation columns of every event and declares, in
the same shard's attributes, what that code stands for and which column carries it,
so the decoder learns the mapping from the data rather than from a convention
shared with the writer. A single type needs no routing and reads either way; a
mixture whose shards carry no source code is refused, and has to be reprocessed.

.. note::

   All shards are expected to carry a surface.

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
   * - ``surface_attr``
     - Local attribute key the generation surface was written under. Matches the
       ``attr`` option of the ``i3-simweight`` processor.
     - str
     - ``surface``

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

.. note::

   Rebuilding a surface reaches for one distribution the package root does not
   export, and relies on how a surface is scaled by its event count, neither of which
   is public API. The decoder therefore pins the release it was written against
   (``0.1.3``) and refuses any other. This goes away once the surface serializes
   itself upstream.
