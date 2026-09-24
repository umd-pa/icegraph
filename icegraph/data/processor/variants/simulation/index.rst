Simulation
==========

:doc:`Processor <../../index>` that stores the simulation an IceCube file was
generated with, and extracts the per-event quantities needed to weight it, using
`simweights <https://github.com/icecube/simweights>`_.

It emits the *inputs* to a weight calculation rather than a weight. Baking a number
in during processing would fix the mixture at processing time. Instead this processor
stores the per-event generation quantities as columns and the file's generation info under
``attrs.LOCAL.simulation``:

.. code-block:: yaml

   simulation:
     sim_code: 173502938116405
     type: corsika
     column: sim_code
     corsika_info: ...

On load, the surface of each set is built by simweights over the files actually
read. No file depends on another, so any can be dropped without reprocessing the rest. A file
without the attribute is inferred to be real data and thus has no weights.

The generation info is stored like so:

- ``corsika_info``: the ``I3CorsikaInfo`` S-frames. CORSIKA files are required to carry them.
- ``weight_dict``: one ``I3MCWeightDict`` row per neutrino type.

The sim code, carried in the generation column
named by ``column``, routes each event to the correct weighter on load. It can be discovered
automatically from the generation info or specified manually:

- **CORSIKA**: one code for all files.
- **NuGen**: a hash of the generation info shared by every type a file threw, and of the
  neutrino flavor. A file is assumed to contain only one flavor and a file holding
  several is refused unless ``sim_code`` is configured manually.

Configuration
-------------

Selected as ``name: i3-simulation``.

.. list-table::
   :header-rows: 1
   :widths: 12 60 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``type``
     - Simulation type: ``nugen``, ``corsika``.
     - str
     - required
   * - ``ids``
     - ID columns keying the output frame, read from the event table.
     - str | int | list
     - required
   * - ``to``
     - Working frame the extracted columns are written to.
     - str
     - ``generation``
   * - ``sim_code``
     - Code used to route events to the correct weighter on load. Discovered from
       the generation info if not given.
     - int
     - discovered
   * - ``weight_dict``
     - Key of the NuGen weight dict.
     - str
     - ``I3MCWeightDict``
   * - ``corsika_info``
     - Key of the CORSIKA S-frames.
     - str
     - ``I3CorsikaInfo``
   * - ``primary``
     - Key of the CORSIKA primaries.
     - str
     - ``PolyplopiaPrimary``

.. code-block:: yaml

   - name: i3-simulation
     kwargs:
       type: corsika
       ids: event_ids

The processor reads from the extracted frames directly, so it does not need an
active frame.

.. note::

   The columns this processor writes are joined onto the output on the id columns.
   An event holding features but no row in the event table joins to null. If
   the tables can be absent for whole files, prefer letting the extractor skip
   those files.
