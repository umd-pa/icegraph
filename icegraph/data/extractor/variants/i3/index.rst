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
   * - ``multiplicity``
     - Count the muons entering the detector, see `Bundle multiplicity`_. Only runs if
       set.
     - mapping
     - ``null``
   * - ``rebuild_missing_mctree``
     - Rebuild the propagated ``I3MCTree`` where it was discarded, see `Rebuilding the
       MC tree`_. Only runs if set.
     - mapping
     - ``null``
   * - ``skip_missing``
     - Skip files that contain no usable frames instead of failing.
     - bool
     - ``false``
   * - ``suppress_icetray_output``
     - Suppress any output from IceTray.
     - bool
     - ``true``

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       gcd_path: /path/to/GCD.i3.zst
       include: [ charge, time ]
       ml_suite: {}
       skip_missing: true


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
     - name: copy  # left joined, so only events that kept features are labeled
       kwargs:
         to: my_truth_table  # whatever table truth is being constructed in
         by: event_ids
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
     - ``I3MCTree``

At least one rule must be set.

The number dropped by each enabled rule is recorded per file under ``attrs.LOCAL.dropped``,
as Q frames (``daq``) and the ``InIceSplit`` events split from them (``physics``):

.. code-block:: yaml

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
     - ``RNGState``
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


Bundle multiplicity
-------------------

When ``multiplicity`` is set, the number of muons entering the detector is counted with
MuonGun's ``muons_at_surface`` and written to each Q frame.

The surface is the convex hull of the strings in the GCD's geometry, padded by ``padding``
meters. A muon counts only if it reaches the surface from outside with energy left, so a
muon starting inside the detector does not count.

.. list-table::
   :header-rows: 1
   :widths: 22 60 10 8

   * - Option
     - Description
     - Type
     - Default
   * - ``key``
     - Frame key the count is written to.
     - str
     - ``bundle_multiplicity``
   * - ``padding``
     - Meters added around the detector hull.
     - float
     - ``0.0``

``muons_at_surface`` reads the propagated ``I3MCTree`` and the ``MMCTrackList``, so both
must be present in the files. The count is that of every muon in the Q frame, so ``multiplicity``
requires ``selection`` with both ``drop_coincident`` and ``drop_oversplit`` set to true.

The settings are stored under ``attrs.GLOBAL.multiplicity``.

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       # ...
       selection:
         drop_coincident: true
         drop_oversplit: true
       multiplicity:
         key: bundle_multiplicity
         padding: 0.0
       # ...
       include: [ features, bundle_multiplicity ]

   processors:
     # select the table, rename 'value' to 'bundle_multiplicity' if desired and copy it to the truth
     # table under construction, same as the MC labels above
