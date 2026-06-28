Simweights
==========

:doc:`Processor <../../index>` that computes per-event weights using the
``simweights`` library. The computed weights are written to a column and the
weight group is recorded in the dataset attributes.

.. note::

   Simweights are computed per-file with nfiles=1. On load, the decoder applies
   a normalization to each weight:

   weight /= # files with same ``weight_group`` designation

   This yields a result identical to running simweights with correct nfiles.
   Weights are therefore tied to which files are loaded, not to how big the batch
   was when the file was processed.

.. note::

   Simweights requires specific tables to be available in the envelope under 'data'. For example,
   weighting CORSIKA events requires access to 'CorsikaWeightMap' and 'PolyplopiaPrimary'.

Configuration
-------------

Selected as ``name: simweights``.

.. list-table::
   :header-rows: 1
   :widths: 16 58 16 10

   * - Option
     - Description
     - Type
     - Default
   * - ``flux``
     - Flux-model plugin selection (``name`` / ``kwargs``). See the
       :doc:`flux models <model/index>` below.
     - mapping
     - required
   * - ``weighter``
     - Name of the ``simweights`` weighter matching the simulation type.
     - str
     - required
   * - ``out``
     - Column the computed weights are written to.
     - str
     - required
   * - ``weight_group``
     - Label recorded in the attributes identifying this weight set.
     - str
     - required

.. code-block:: yaml

   - name: simweights
     kwargs:
       flux: { name: gaisser-h4a, kwargs: {} }
       weighter: CorsikaWeighter
       out: weight
       weight_group: corsika

Flux models
-----------

The flux model is itself a selectable plugin; the choices are documented under the
:doc:`flux model <model/index>` slot.

.. toctree::
   :hidden:

   model/index
