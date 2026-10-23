I3
==

:doc:`Extractor <../../index>` variant that reads IceCube I3 files. It runs the
IceTray ``ml_suite`` feature extraction over each file's frames, using a GCD file
to supply detector geometry, and emits the extracted per-event features downstream.

Because it depends on IceTray, when using this extractor the pipeline must be run
through the framework-provided IceTray-enabled Python shim (see `<../../../../usage>`,
'Running under IceTray').

Configuration
-------------

Selected as ``name: i3``.

.. list-table::
   :header-rows: 1
   :widths: 18 55 17 10

   * - Option
     - Description
     - Type
     - Default
   * - ``gcd_path``
     - Path to the GCD file.
     - path
     - required
   * - ``include``
     - Names of the feature groups to pass downstream.
     - list[str]
     - required
   * - ``ml_suite``
     - Options forwarded to the IceTray ``ml_suite`` extraction (validated by
       ``ml_suite`` itself).
     - mapping
     - required
   * - ``mclabeler``
     - Options forwarded to the IceTray ``MCLabeler`` (validated by ``MCLabeler``
       itself), see `MC labels`_. Only runs if set.
     - mapping
     - ``null``
   * - ``selection``
     - Drop simulated events that cannot be given a single truth, see `Selection`_.
       Only runs if set.
     - mapping
     - ``null``
   * - ``muongun``
     - Labels computed using with MuonGun, see `MuonGun labels`_. Only runs if set.
     - mapping
     - ``null``
   * - ``rebuild_missing_mctree``
     - Rebuild the propagated ``I3MCTree`` where it was discarded, see `Rebuilding the
       MC tree`_. Only runs if set.
     - mapping
     - ``null``
   * - ``skip_missing``
     - Skip a file that holds events but lacks one of the ``include`` keys, instead of
       failing. A skipped file is not counted toward its set, see `Files without events`_.
     - bool
     - ``false``
   * - ``suppress_icetray_output``
     - Suppress any output from IceTray.
     - bool
     - ``true``
   * - ``ids``
     - Columns identifying an event. They must be index columns the table writer books
       with every table. Processors key on them by default, see :doc:`Processor
       <../../../processor/index>`.
     - list[str]
     - ``[Run, Event, SubEvent, SubEventStream]``

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       gcd_path: /path/to/GCD.i3.zst
       include: [ charge, time ]
       ml_suite: {}
       skip_missing: true


Files without events
--------------------

``I3EventHeader`` is always booked, so the ids of every event written are read off it
into the envelope.

A file that holds no events, before or after the `Selection`_, is still passed downstream
with whatever tables it does hold. The
:doc:`collector <../../../collector/index>` adds it to a shard so it still counts toward its
set for file-count sensitive operations, like weighting.

``skip_missing`` only applies to files that actually hold events.


MC labels
---------

When ``mclabeler`` is set, ``sim_services.label_events.MCLabeler`` is added to the
tray. It writes ``classification``, ``coincident_muons``, ``bg_muon_mcpe`` and
``bg_muon_mcpe_charge`` (plus any ``key_postfix``) to each DAQ frame.

Exactly one of ``event_properties_name``, ``weight_dict_name`` and
``corsika_weight_map_name`` must be set for ``MCLabeler``.

Brief example on using the MCLabeler during processing:

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       # ...
       mclabeler:
         mctree_name: I3MCTree_preMuonProp
         corsika_weight_map_name: CorsikaWeightMap
       # ...
       include: [ features, classification ]  # 'classification' comes from MCLabeler

   processors:
     # ... other processing as usual ...

     # move to the table with MCLabeler 'classification' output
     - name: select
       kwargs:
         key: classification

     # rename the 'value' column to 'classification'
     - name: rename
       kwargs:
         map:
           value: classification

     # copy to the main truth table under construction
     - name: copy  # left joined on the event ids, so only events that kept features are labeled
       kwargs:
         to: my_truth_table  # whatever table truth is being constructed in
         cols: classification

     # ... further processing as usual ...

   writer:
     ...


Selection
---------

``selection`` provides basic tools to drop undesired frames from the stream.

.. list-table::
   :header-rows: 1
   :widths: 22 60 10 8

   * - Option
     - Description
     - Type
     - Default
   * - ``drop_coincident``
     - Drop Q frames holding more than one primary (coincident events).
     - bool
     - ``false``
   * - ``drop_oversplit``
     - Drop Q frames holding one primary that was split into more than one
       ``InIceSplit`` event.
     - bool
     - ``false``
   * - ``mctree``
     - The tree the primaries are counted in.
     - str
     - ``I3MCTree_preMuonProp``

At least one rule must be set.

A file whose every event is dropped is still kept, see `Files without events`_.

The number dropped by each enabled rule is recorded per file under ``attrs.LOCAL.dropped``,
as Q frames (``daq``) and the ``InIceSplit`` events split from them (``physics``). The
:doc:`collector <../../../collector/index>` keeps it under the file in ``attrs.LOCAL.sources``:

.. code-block:: yaml

   sources:
     /path/to/file.i3.zst:
       dropped:
         coincident: { daq: 3, physics: 4 }
         oversplit:  { daq: 7, physics: 15 }


Rebuilding the MC tree
----------------------

Simulation often discards the propagated ``I3MCTree`` to save space. When
``rebuild_missing_mctree`` is set, each Q frame that does not hold the propagated tree but does
hold saved state has it rebuilt with ``I3PropagatorModule``, using PROPOSAL for muons and
CMC for showers. The ``MMCTrackList`` is written along with the tree, so if the frame already has one
it will be deleted and replaced.

.. list-table::
   :header-rows: 1
   :widths: 22 60 10 8

   * - Option
     - Description
     - Type
     - Default
   * - ``mctree``
     - The propagated tree, rebuilt when the Q frame does not hold it.
     - str
     - ``I3MCTree``
   * - ``raw_mctree``
     - The un-propagated tree it is rebuilt from.
     - str
     - ``I3MCTree_preMuonProp``
   * - ``rng_state``
     - The saved state of the random number generator.
     - str
     - ``I3MCTree_preMuonProp_RNGState``
   * - ``random_service``
     - Type of the random number generator, one of ``SPRNG``, ``GSL`` or ``MT``. It must
       match the type the state was saved from.
     - str
     - ``SPRNG``

The tree is rebuilt after the `Selection`_, so dropped events are never propagated.

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       # ...
       rebuild_missing_mctree:
         raw_mctree: I3MCTree_preMuonProp


MuonGun labels
--------------

When ``muongun`` is set, the muons entering the detector are found with MuonGun's
``muons_at_surface``, and labels of the bundle they form are written to each Q frame:

* the **multiplicity**, the number of muons that enter, and
* the **prominence**, the share of the bundle's energy at the surface carried by its
  leading muon, ``max(E) / sum(E)``. It is 1 for a single muon, falls toward 0 as the energy
  is spread over more muons, and is NaN when no muon enters.

The surface is the convex hull of the strings in the GCD's geometry, padded by ``padding``
meters. A muon counts only if it reaches the surface from outside with energy left, so a
muon starting inside the detector does not count. Its energy is the energy it has as it
reaches the surface.

Each label is computed only if the frame key it is written to is set, and at least one must
be.

.. list-table::
   :header-rows: 1
   :widths: 22 60 10 8

   * - Option
     - Description
     - Type
     - Default
   * - ``padding``
     - Meters added around the detector hull.
     - float
     - ``0.0``
   * - ``multiplicity``
     - Frame key the multiplicity is written to.
     - str
     - ``null``
   * - ``prominence``
     - Frame key the prominence is written to.
     - str
     - ``null``

``muons_at_surface`` reads the propagated ``I3MCTree`` and the ``MMCTrackList``, so both
must be present in the files. The labels cover every muon in the Q frame, so ``muongun``
requires ``selection`` with both ``drop_coincident`` and ``drop_oversplit`` set to true for accurate counts.

The padding is stored under ``attrs.GLOBAL.muongun`` as this must be consistent across loaded shards.

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       # ...
       selection:
         drop_coincident: true
         drop_oversplit: true
       muongun:
         padding: 0.0
         multiplicity: bundle_multiplicity
         prominence: prominence
       # ...
       include: [ features, bundle_multiplicity, prominence ]

   processors:
     # select each table, rename 'value' to its label if desired and copy it to the truth
     # table under construction, same as the MC labels above
