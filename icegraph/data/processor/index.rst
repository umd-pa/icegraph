Processor
=========

A **processor** is a single transformation stage in the :doc:`pipeline <../index>`.
The pipeline runs an ordered list of processors between the extractor and the
writer, each applying one operation to the data as it streams through. Processors
are where feature engineering happens: selecting and renaming columns, building
graph edges, computing weights, assigning dataset splits, etc.

Usage
-----

The ``processors`` key holds an ordered list; each entry selects a processor by
name. Order matters, since each processor sees the result of the previous one.

.. code-block:: yaml

   processors:
     - name: select
       kwargs: { key: pulses }
     - name: knn
       kwargs: { by: [string, om], col: position, out: [edge_index, edge_attr], k: 8 }

How it Works
------------

The data flowing between stages is an *envelope* holding the raw extracted frames,
a set of working frames, and the committed output. Processors operate on the
*active* working frame: the :doc:`select <variants/select/index>` processor chooses
which frame is active, most processors transform that frame in place, and the
:doc:`commit <variants/commit/index>` processor writes finished columns into the
output that the writer persists. Columns can be addressed individually or through
named groups defined by the :doc:`alias <variants/alias/index>` processor.

Variants
--------

Staging:

* :doc:`select <variants/select/index>`: choose the active working frame.
* :doc:`commit <variants/commit/index>`: write finished columns to the output.
* :doc:`copy <variants/copy/index>`: copy columns into another frame.

Columns and values:

* :doc:`alias <variants/alias/index>`: define named groups of columns.
* :doc:`rename <variants/rename/index>`: rename columns.
* :doc:`map <variants/map/index>`: remap the values of a column.
* :doc:`fill <variants/fill/index>`: add or overwrite a column with a constant.
* :doc:`unique <variants/unique/index>`: record the distinct values of columns.
* :doc:`stats <variants/stats/index>`: compute per-column statistics.

Graph construction:

* :doc:`domproc <variants/dom/index>`: convert DOM identifiers to positions.
* :doc:`knn <variants/knn/index>`: build k-nearest-neighbor graph edges.
* :doc:`compress <variants/compress/index>`: stack rows into per-event arrays.
* :doc:`pivot <variants/pivot/index>`: reshape long-form data to wide.

Weighting and splitting:

* :doc:`simweights <variants/simweights/index>`: compute event weights.
* :doc:`splitmap <variants/splitmap/index>`: assign train/validation/test splits.

Debugging:

* :doc:`inspect <variants/inspect/index>`: print the active frame for inspection.

Registering a new processor
---------------------------

A processor is a subclass of ``Processor`` that declares a ``name`` and ``version``
and implements the transformation:

``_process(self, item) -> Envelope | None``
   Transform the envelope and return it, or return ``None`` to drop the event.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.data.processor import Processor, ProcessorFactory
   from icegraph.data.envelope import Envelope

   from .config import MyProcessorConfig

   class MyProcessor(Processor[MyProcessorConfig]):
       name: ClassVar[str] = "my-processor"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyProcessorConfig:
           return MyProcessorConfig(**config)

       def build(self) -> None:
           ...

       def _process(self, item: Envelope) -> Envelope | None:
           ...

   ProcessorFactory.register(MyProcessor)

.. toctree::
   :hidden:

   variants/select/index
   variants/commit/index
   variants/copy/index
   variants/alias/index
   variants/rename/index
   variants/map/index
   variants/fill/index
   variants/unique/index
   variants/stats/index
   variants/dom/index
   variants/knn/index
   variants/compress/index
   variants/pivot/index
   variants/simweights/index
   variants/splitmap/index
   variants/inspect/index
