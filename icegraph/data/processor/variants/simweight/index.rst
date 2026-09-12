Simweight
=========

:doc:`Processor <../../index>` that extracts the generation quantities needed to
weight simulation, using `simweights <https://github.com/icecube/simweights>`_.

It emits the *inputs* to a weight calculation rather than a weight. Baking a number
in during processing would fix the mixture at processing time. Instead this processor stores the per-event
generation quantities as columns and the file's generation surface as a local
attribute, and the surfaces are summed over the shards actually read when the
dataset is loaded.

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
   * - ``weighter``
     - Simulation type to weight: ``nugen``, ``corsika``, ``genie``.
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
   * - ``nfiles``
     - Files the surface is normalized to. Leave at ``1``. Surfaces are summed on
       load. Set to null for simulation carrying S-frames, where simweights derives
       the surface from the file itself.
     - int | null
     - ``1``
   * - ``cols``
     - Weight columns to extract.
     - list[str]
     - ``[energy, cos_zen, pdgid, event_weight]``
   * - ``out``
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
       weighter: nugen
       tables: [ I3MCWeightDict ]
       ids: event_ids

The processor reads from the extracted frames directly, so it does not need an
active frame.

.. note::

   The columns this processor writes are joined onto the output on the id columns.
   An event holding features but no row in the weighting table joins to null. If
   the weighting tables can be absent for whole files, prefer letting the
   extractor skip those files.
