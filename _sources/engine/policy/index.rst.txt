Policy
======

The **policy** defines the task a run is solving and the obligations that task
places on each component. It fixes how model outputs are interpreted, how many
output channels the model must produce, whether targets are normalized, and the
structural checks applied to component inputs and outputs. It is the single place a
run declares "this is a classification problem" or "this is a regression problem".

Usage
-----

The policy occupies the top-level ``policy`` slot.

.. code-block:: yaml

   policy:
     name: multiclass
     kwargs: {}

How it works
------------

From the dataset, a policy builds a task specification: the output layout the model
must produce, the data type of the targets, and whether targets are normalized. It
then issues a *contract* to each :doc:`component <../components/index>` as that
component is attached. A contract carries task-derived parameters together with validators
that check a component conforms to the task. This is how one task definition keeps the model,
normalizer, transformer, optimizer, and loss mutually consistent.

Variants
--------

* :doc:`Multiclass <variants/multiclass/index>`: classification over discrete
  labels.
* :doc:`Regression <variants/regression/index>`: prediction of continuous targets.

Registering a new policy
------------------------

A policy is a subclass of ``Policy`` that declares a ``name`` and ``version`` and
builds the task specification:

``_build_task_spec(self) -> TaskSpec``
   Return the output offsets, target data type, and target-normalization flag for
   the task.

The base supplies default contracts for each component kind; override the
per-component contract methods only when a task needs stricter validation or extra
parameters.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.engine.policy import Policy, PolicyFactory, TaskSpec

   from .config import MyPolicyConfig

   class MyPolicy(Policy[MyPolicyConfig]):
       name: ClassVar[str] = "my-policy"
       version: ClassVar[int] = 1

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyPolicyConfig:
           return MyPolicyConfig(**config)

       def _build_task_spec(self) -> TaskSpec:
           ...  # return a TaskSpec for the task

   PolicyFactory.register(MyPolicy)

.. toctree::
   :hidden:

   variants/multiclass/index
   variants/regression/index
