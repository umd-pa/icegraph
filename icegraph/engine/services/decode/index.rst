Decode
======

The **decode service** turns raw dataset records into the tensors a run consumes.
It is the bridge between the record format and the model's input and output spaces.

Usage
-----

Configured under ``services.decode``. The decoders for attributes and records are
themselves selectable plugins.

.. code-block:: yaml

   services:
     decode:
       features: []
       targets:  [ bundle ]
       auxiliary: []
       keymap:
         truth: targets
         edge_attr: edge_weight
       attrs:
         name: standard
         kwargs: {}
       records:
         name: standard
         kwargs: {}

How it works
------------

Two decoders divide the work. The :doc:`attribute decoder <attrs/index>` reads
dataset-level metadata, such as column names, per-column statistics, and the set of
observed label values. The :doc:`record decoder <records/index>` reads an
individual record and extracts relevant tensors (such as ``features``, ``targets``, etc).

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 55 17 10

   * - Option
     - Description
     - Type
     - Default
   * - ``features``
     - Column names used as input features.
     - list[str]
     - ``[]``
   * - ``targets``
     - Column names used as training targets.
     - list[str]
     - ``[]``
   * - ``auxiliary``
     - Extra columns to decode but exclude from the loss (useful for plotting and
       analysis).
     - list[str]
     - ``[]``
   * - ``keymap``
     - Overrides mapping logical roles to the keys used in the stored data.
     - mapping
     - identity
   * - ``attrs``
     - Attribute-decoder plugin selection (``name`` / ``kwargs``).
     - mapping
     - required
   * - ``records``
     - Record-decoder plugin selection (``name`` / ``kwargs``).
     - mapping
     - required

The ``keymap`` accepts the keys ``truth`` (the source of both targets and
auxiliary), ``features``, ``edge_index``, ``edge_attr``, and ``simweights``; each
defaults to its own name.

Sub-slots
---------

.. toctree::
   :maxdepth: 2

   attrs/index
   records/index
