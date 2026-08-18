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
