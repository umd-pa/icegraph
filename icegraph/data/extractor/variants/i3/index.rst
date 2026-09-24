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
       itself). The labeler only runs if set, see `MC labels`_.
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
