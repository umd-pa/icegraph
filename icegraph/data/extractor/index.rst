Extractor
=========

The **extractor** is the first stage of the :doc:`pipeline <../index>`. It reads
source files and emits one envelope per file containing extracted data.
A pipeline has exactly one extractor, and the extractor determines which file format the
pipeline ingests.

Usage
-----

Configured under the top-level ``extractor`` key.

.. code-block:: yaml

   extractor:
     name: i3
     kwargs:
       gcd_path: /path/to/GCD.i3.zst
       include: [ ... ]
       ml_suite: { ... }

How it Works
------------

The extractor declares the file extension it handles; the pipeline resolves the
source paths to matching files and streams them through. For each file the
extractor reads the relevant frames and emits an envelope of raw data downstream.

Variants
--------

* :doc:`I3 <variants/i3/index>`: extracts features from IceCube I3 files.

Registering a new extractor
---------------------------

An extractor is a subclass of ``Extractor`` that declares a ``name``, a
``version``, and the ``file_ext`` it reads, and implements the per-file extraction.
Register it with ``ExtractorFactory``.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.data.extractor import Extractor, ExtractorFactory
   from icegraph.data.envelope import Envelope

   from .config import MyExtractorConfig

   class MyExtractor(Extractor[MyExtractorConfig]):
       name: ClassVar[str] = "my-extractor"
       version: ClassVar[int] = 1
       file_ext: ClassVar[str] = ".myext"

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyExtractorConfig:
           return MyExtractorConfig(**config)

       def build(self) -> None:
           ...

       def _process(self, item) -> Envelope | None:
           ...  # read the file and emit an envelope

   ExtractorFactory.register(MyExtractor)

.. toctree::
   :hidden:

   variants/i3/index
