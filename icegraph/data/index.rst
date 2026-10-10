Pipeline
========

The **pipeline** is the data-processing subsystem. It converts raw source
files into an ML-ready graph dataset.

A pipeline is a concurrent, stage-based process. It wires these stages in sequence:

* an **extractor**, which reads source files and emits envelopes containing extracted data,
* the **collector**, which packs the envelopes of whole files into larger blocks, each
  becoming one shard,
* a series of **processors**, each applying one transformation to the data (for
  example building graph edges, selecting or renaming columns, or computing
  weights), and
* a **writer**, which persists the finished dataset.

Each stage runs on its own thread and communicates with its neighbors through
bounded queues, so stages execute in parallel and a slow stage exerts
backpressure on those upstream rather than exhausting memory. A pipeline may also be run as
multiple independent processes. Shards are numbered from 0 in every run, so independent runs
writing to one directory must each set a different writer ``prefix``.

A pipeline is assembled from a configuration that names one extractor, an ordered
list of processors, and one writer, and optionally configures the collector. Each stage is a :doc:`plugin
<../common/plugins/index>`.

.. code-block:: yaml

   extractor:
     name: i3
     kwargs: { ... }
   collector: { ... }
   processors:
     - name: select
       kwargs: { ... }
     - name: knn
       kwargs: { ... }
       ...
   writer:
     name: lmdb
     kwargs: { ... }

.. toctree::
   :maxdepth: 2
   :caption: Stages

   extractor/index
   collector/index
   processor/index
   writer/index
