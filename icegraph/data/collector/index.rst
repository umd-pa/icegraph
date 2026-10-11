Collector
=========

The **collector** sits between the :doc:`extractor <../extractor/index>` and the
:doc:`processors <../processor/index>` of every :doc:`pipeline <../index>`. It packs the
envelopes of whole input files into *blocks*. Each block is passed to the processors
as one envelope and written as one shard, so shards can be much larger than the files
they come from.

How it Works
------------

Files are collected per group (see ``group_by``). A block closes once its extracted
tables reach ``min_bytes`` and the next file of its group holds events, which starts the
next block. A file without events never starts a block, it always joins one and is
counted in it. Once all input files have been extracted, any remaining partially filled
block is emitted.

A file is never split across blocks. When files are merged:

* their tables are stacked per key, in the order the files arrived.
* their event ids must not repeat, since the block is processed as one file. A repeated
  event raises.
* their global attributes must be identical.
* each file's local attributes are kept under ``attrs.LOCAL.sources``, keyed by the origin file path:

.. code-block:: yaml

   sources:
     /path/to/a.i3.zst: { dropped: { ... } }
     /path/to/b.i3.zst: {}  # held no events

Every shard records each input file in it, with events or without, so
file-count sensitive operations (like weighting, see :doc:`i3-simulation
<../processor/variants/simulation/index>`) count every file that was processed.
Shards are numbered in the order their blocks close, which is used to name the output shard (see
:doc:`Writer <../writer/index>`). Which files share a shard depends on the order
they finish extraction, so it can differ between runs.

The collector runs in one process, in addition to ``nproc``.

.. note::

   A group whose files hold no events at all cannot be written and raises.

Configuration
-------------

Configured under the top-level ``collector`` key. It can be left out to use the
defaults.

.. list-table::
   :header-rows: 1
   :widths: 14 60 14 12

   * - Option
     - Description
     - Type
     - Default
   * - ``min_bytes``
     - Size of extracted tables at which a block closes. ``0`` gives one file with events
       per block, along with any files without events that follow it.
     - int
     - ``0``
   * - ``group_by``
     - Which files may share a block. ``parent``: files in the same directory. ``all``:
       every file, for when every directory holds the same set.
     - str
     - ``parent``

.. code-block:: yaml

   extractor:
     ...
   collector:
     min_bytes: 100_000_000
     group_by: parent
   processors:
     ...

A block's working frames are held in memory while it is processed, and each processor
thread can work on a different block, so choose ``min_bytes`` with available memory in
mind.
