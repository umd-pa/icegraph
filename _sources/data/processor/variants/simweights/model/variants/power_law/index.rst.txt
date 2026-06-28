Power Law
=========

:doc:`Flux model <../../index>` implementing a configurable power-law spectrum,
:math:`\phi(E) = \phi_0 \, (E / E_0)^{-\gamma}`.

Configuration
-------------

Selected as ``name: power-law``.

.. list-table::
   :header-rows: 1
   :widths: 12 60 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``g``
     - Spectral index :math:`\gamma`.
     - number
     - required
   * - ``phi0``
     - Flux normalization :math:`\phi_0`.
     - number
     - required
   * - ``e0``
     - Reference energy :math:`E_0`.
     - number
     - required

.. code-block:: yaml

   flux:
     name: power-law
     kwargs:
       g: 2
       phi0: 1.0e-18
       e0: 100000.0
