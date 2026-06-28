Factory
=======

A **factory** is the registry that resolves a configured name to a concrete
:doc:`plugin <../plugins/index>` class. Each plugin family maintains its own
factory, which holds every variant registered for that family and produces the
one named in a configuration slot.

Registry model
--------------

A factory is a name-keyed map from plugin name to plugin class. Registries are
per-family and isolated: registering a normalizer affects only the normalizer
factory, so the same name can exist independently across families. Names are
stored case-insensitively.

Two forms exist:

* the base factory, which maps a name to its class and can instantiate it
  directly, and
* the **plugin factory**, which additionally runs the plugin's configuration
  validation and constructs the instance from the validated result. This is the
  form used by the configuration-driven families throughout the framework.

Because registration is local to a family, introducing a new variant is a matter
of defining the plugin and registering it with its family's factory. The call
sites that select a plugin by name require no change.
