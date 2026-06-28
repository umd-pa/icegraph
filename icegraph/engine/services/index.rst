Services
========

**Services** supply the shared, run-scoped capabilities and data that components,
the policy, and the engine itself draw on. Where a component computes a value, a
service provides a resource: the decoded records and dataset statistics, the
process state such as device, seed, and rank, the backing record store, the
batched data loaders, and the metric computations. Components request the services
they need by name.

Each service is configured under its own key in the ``services`` section. The
built-in services are independent subsystems, each documented on its own page:

.. code-block:: yaml

   services:
     state:   { ... }
     data:    { ... }
     record:  { ... }
     decode:  { ... }
     metrics: { ... }

How it Works
------------

A service is a :doc:`plugin <../../common/plugins/index>` keyed by its section
name. A service may declare dependencies on other services; the engine builds the
configured services, verifies that every declared dependency is present, and
attaches them in dependency order so that each service can rely on those it depends
upon. A dependency cycle is reported as a configuration error.

.. toctree::
   :maxdepth: 2
   :caption: Services

   state/index
   data/index
   record/index
   decode/index
   metrics/index

Registering a new service
-------------------------

The ``services`` section accepts user-defined services in addition to the built-in
set. A service is a subclass of ``Service`` that declares a ``name`` and
``version`` and is registered with ``ServiceFactory``; it may list other services
it needs in the ``deps`` class attribute so the engine attaches them first.

.. code-block:: python

   from typing import Any, ClassVar

   from icegraph.engine.services.service import Service
   from icegraph.engine.services.factory import ServiceFactory

   from .config import MyServiceConfig

   class MyService(Service[MyServiceConfig]):
       name: ClassVar[str] = "my-service"
       version: ClassVar[int] = 1
       deps = ("state",)

       @classmethod
       def validate_config(cls, config: dict[str, Any]) -> MyServiceConfig:
           return MyServiceConfig(**config)

       def build(self) -> None:
           ...

   ServiceFactory.register(MyService)

Once registered, other services or components reach it through
``services.require("my-service")``.
