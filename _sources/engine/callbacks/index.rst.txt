Callbacks
=========

**Callbacks** observe and extend a run by hooking into the engine's lifecycle.
They let logging, metric accumulation, plotting, checkpoint export, and similar
concerns participate in a run without altering the engine or the model. As the
engine reaches defined points in its lifecycle, it invokes the registered
callbacks, passing each a context describing that point.

Callbacks are registered with an engine before execution begins.

Callbacks are engine-specific. The concrete callbacks shipped
with a particular engine are documented alongside that engine.
