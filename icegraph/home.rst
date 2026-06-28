Home
====

IceGraph is an end-to-end framework for building and deploying Graph Neural
Networks for reconstruction and classification in IceCube. It covers the full path
from raw detector files to a trained, deployable model through three workflows:

* **Processing** converts events from source files to an ML-ready graph dataset.
* **Training** fits a GNN to that dataset using the PyTorch framework.
* **Inference** applies a trained model to new data to produce predictions.

All three are driven by YAML configuration and share a common substrate, so most of
this documentation is organized around the framework's objects and what each is
responsible for.

To get started with scripting, see :doc:`usage`.

Authorship and Disclaimer
-------------------------

IceGraph is authored by me, Taylor St Jean. AI was used (and will continue to be used) as a
debugging and optimization tool during development, but the codebase, architecture, and all
implementation decisions are my own work.
