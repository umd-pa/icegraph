Status
======

The **status** is the central record of what a run is doing. Code anywhere in the
backend reports the work it is doing to it as named tasks with optional progress.
Every record logged under ``icegraph`` is routed
through it as well, before going on to the configured logging handlers as usual.
Callbacks and anything else holding the status can subscribe to status streams.

Usage
-----

The status occupies the top-level ``status`` slot. Every option has a default, so the
slot can be left out.

.. code-block:: yaml

   status:
     profile_flops: true

Services, components and the policy reach it through ``status`` on their context.
To report active work:

.. code-block:: python

   status = self._ctx.status

   with status.task("Indexing shards", total=len(paths)) as task:
       for path in paths:
           ...
           task.advance()

   # or, equivalently, for a plain loop
   for path in status.track(paths, "Indexing shards"):
       ...

Measuring a stream of batches, as the engines do for each split:

.. code-block:: python

   throughput = status.throughput("train")

   for batch in throughput.iterate(loader):
       with throughput.step(batch):
           ...

   throughput.stats()  # samples/s, batches/s, FLOP/s, fraction of time waiting on data

Following it from a callback:

.. code-block:: python

   from icegraph.engine.status import TaskEvent, LogEvent, EventKind

   class MyCallback(TrainerCallback):
       def on_status(self, ctx):
           event = ctx.event

           if isinstance(event, TaskEvent) and event.kind is EventKind.FINISHED:
               print(event.task.name, event.task.elapsed)

           elif isinstance(event, LogEvent):
               print(event.entry.levelname, event.entry.message)

``icegraph.ui.StatusView`` and ``icegraph.ui.LogView`` render the running tasks and the
recent logs for Rich displays.

How it works
------------

**Tasks.** A task opened while another is open on the same thread becomes its child,
so the running tasks form a tree, e.g. *Starting up* containing *Building components*
containing *Indexing shards*. Used as a context manager, a task finishes on exit or
fails if an exception escapes it, and either way its timing is logged. Every start,
update, finish and failure is sent to listeners. Listeners that raise are logged and
unsubscribed so a broken display cannot stop the work it reports on.

**Logs.** The status installs a handler on the ``icegraph`` logger, so every record any
module logs reaches it, is kept as a ``LogEntry``, and is sent to listeners as a
``LogEvent``. The status's own messages, such as each
task's completion and timing, take the same path. What reaches the status is whatever
the logging configuration lets through, so call ``configure_logging`` before building
the engine. Reconfiguring logging afterwards removes the handler.

**Throughput.** ``iterate`` reports the wait for a stream's first batch as a task, which
covers worker startup and the first buffer fill, and meters time blocked on the loader.
``step`` counts samples and batches. On the stream's first batch, it counts FLOPs with
torch's ``FlopCounterMode``, later batch FLOPs are computed by scaling this value using node count. Only
matmul-like operations are counted, not scatter aggregation. Rates are taken over each
meter's last ``window`` updates.

The status is per process. Loader workers receive a copy with no listeners, nothing
recorded and no logging handler, so work they do is not reported back to the main
process.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 60 10 10

   * - Option
     - Description
     - Type
     - Default
   * - ``log_history``
     - Log entries kept for inspection.
     - int
     - ``200``
   * - ``window``
     - Updates each meter keeps to compute its rate.
     - int
     - ``50``
   * - ``profile_flops``
     - Count FLOPs on the first batch of each throughput stream to report FLOP/s.
     - bool
     - ``true``
