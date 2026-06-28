Standard
========

:doc:`Transformer <../../index>` variant that maps named feature columns into a
chosen value space. Each entry in ``transforms`` selects the space and log base for
one column by name; columns that are not listed stay in linear space and are passed
through unchanged.

Supported spaces:

* ``linear``: identity (no transform).
* ``log``: logarithm in the configured base. Inputs must be positive.
* ``asinh``: inverse hyperbolic sine, scaled by the log of the base. Defined for
  all real inputs, so it tolerates zeros and negatives where ``log`` cannot.

Configuration
-------------

Selected as ``name: standard``.

.. list-table::
   :header-rows: 1
   :widths: 20 45 25 10

   * - Option
     - Description
     - Type
     - Default
   * - ``transforms``
     - Mapping of feature-column name to a space selection.
     - dict[str, SpaceSelection]
     - ``{}``

Each ``SpaceSelection`` has:

.. list-table::
   :header-rows: 1
   :widths: 15 45 30 10

   * - Option
     - Description
     - Type
     - Default
   * - ``space``
     - The value space to map the column into.
     - ``linear`` | ``log`` | ``asinh``
     - required
   * - ``base``
     - Log base for ``log`` and ``asinh``. Must be positive and not equal to 1.
     - int
     - ``10``

.. code-block:: yaml

   components:
     transformer:
       name: standard
       kwargs:
         transforms:
           charge: { space: log, base: 10 }
           time:   { space: asinh, base: 10 }
