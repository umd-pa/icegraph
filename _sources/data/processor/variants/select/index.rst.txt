Select
======

:doc:`Processor <../../index>` that chooses the active working frame. It loads the
named frame from the extracted data into the working area and marks it active, so
subsequent processors operate on it. Most processing sequences begin with a
``select``.

.. warning::

   ''select'' performs a deepcopy if the frame has not yet been loaded as active.
   Returning to a previously active frame does NOT repeat the deepcopy.

Configuration
-------------

Selected as ``name: select``.

.. list-table::
   :header-rows: 1
   :widths: 15 65 10 10

   * - Option
     - Description
     - Type
     - Default
   * - ``key``
     - Name of the extracted frame to make active.
     - str
     - required

.. code-block:: yaml

   - name: select
     kwargs: { key: pulses }
