Simweight
=========

:doc:`Processor <../../index>` that extracts the generation quantities needed to
weight simulation, using `simweights <https://github.com/icecube/simweights>`_.

It emits the *inputs* to a weight calculation rather than a weight. Baking a number
in during processing would fix the mixture at processing time. Instead this processor stores the per-event
generation quantities as columns and the file's generation surface as a local
attribute, and the surfaces are summed over the shards actually read when the
dataset is loaded.

A dataset may be loaded from files of more than one simulation type, which are
weighted against different fluxes. Since a block read on load spans files, each
event also carries a source code naming the simulation it came from, derived from
the simulation so that files processed independently agree on it.

Configuration
-------------

Selected as ``name: i3-simweight``.

.. list-table::
   :header-rows: 1
   :widths: 12 60 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``simulation``
     - Simulation type to weight: ``nugen``, ``corsika``.
     - str
     - required
   * - ``tables``
     - Extracted tables the weighter reads, by key. Also the tables considered
       when the weight columns are keyed.
     - str | int | list
     - required
   * - ``ids``
     - Id columns keying the output frame. The table they are read from is
       resolved internally.
     - str | int | list
     - required
   * - ``to``
     - Working frame the extracted columns are written to.
     - str
     - ``generation``
   * - ``attr``
     - Local attribute key the generation surface is stored under.
     - str
     - ``surface``

.. code-block:: yaml

   - name: i3-simweight
     kwargs:
       simulation: nugen
       tables: [ I3MCWeightDict ]
       ids: event_ids

The processor reads from the extracted frames directly, so it does not need an
active frame.

.. note::

   The columns this processor writes are joined onto the output on the id columns.
   An event holding features but no row in the weighting table joins to null. If
   the weighting tables can be absent for whole files, prefer letting the
   extractor skip those files.
