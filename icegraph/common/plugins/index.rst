Plugin
======

A **plugin** is the unit of configurable, swappable behavior in IceGraph. Every
point at which the framework offers a choice is expressed as a plugin: the
normalizer, the loss function, the model, each service, the policy, and each
stage of the data pipeline are all plugins. They share one identity model, one
configuration contract, and one lifecycle.

Identity
--------

Every plugin declares two class-level attributes:

* ``name`` is the identifier used to select the plugin from configuration.
  Lookups are case-insensitive.
* ``version`` is an integer revision used to record which implementation produced
  a stored artifact or checkpoint.


Configuration
-------------

Plugins are selected and configured through a uniform slot in the YAML config:

.. code-block:: yaml

   <slot>:
     name: <plugin-name>
     kwargs: { <option>: <value>, ... }

``name`` chooses the implementation; ``kwargs`` carries that implementation's
options. Before construction, a plugin validates the raw ``kwargs`` mapping into a
typed configuration object, so invalid or missing options are rejected up front.
Slots that accept a list of plugins repeat the same ``name`` / ``kwargs`` pair per entry.

Lifecycle
---------

A plugin is created in two phases:

#. **Build.** The plugin is constructed from its validated configuration and runs
   a single build hook for setup that depends only on its own options. This phase
   is inexpensive and has no access to the wider run.
#. **Attach.** The plugin is later attached to a *context*, an object that grants
   access to the runtime collaborators it needs, such as other plugins, services,
   and the active device. Work that depends on the surrounding run is deferred to
   this phase.

This separation keeps plugins cheap to construct and describe while ensuring that
anything requiring the assembled run happens only once the run exists. Resolution
of a configured ``name`` to a concrete plugin class is the responsibility of a
:doc:`factory <../factory/index>`.
