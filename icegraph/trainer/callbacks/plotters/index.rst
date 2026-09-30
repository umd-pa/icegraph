Plotters
========

A group of :doc:`trainer callbacks <../index>` that accumulate model predictions
over the validation and test splits and render diagnostic plots to the run's output
directory. Each is registered with a ``CallbackSpec`` like any other callback, and
they accept optional keyword arguments (such as a map of class indices to display
names for the classification plots).

Regression plots:

* **Parity** (``ParityPlotter``): predicted value against true value, where points
  on the diagonal are perfect predictions.
* **Bias** (``BiasPlotter``): prediction residual as a function of the true value,
  revealing systematic over- or under-prediction.

Classification plots:

* **Confusion Matrix** (``CMPlotter``): counts of predicted versus true classes.
* **P(true)** (``PTruePlotter``): distribution of the probability the model assigns
  to the correct class.
* **P(positive)** (``BinaryPPositivePlotter``): distribution of the predicted
  positive-class probability for binary tasks.
* **ROC** (``ROCPlotter``): receiver operating characteristic curve.
* **Precision-Recall** (``PrecisionRecallPlotter``): precision against recall across
  thresholds.
* **P(class) vs. auxiliary** (``PPositiveAuxPlotter``): 2D heatmap of each class's
  one-vs-rest probability against an auxiliary column, one heatmap per class.

.. code-block:: python

   from icegraph.trainer.callbacks import CallbackSpec, ParityPlotter, ROCPlotter, PPositiveAuxPlotter

   trainer.register_callback(CallbackSpec(callback=ParityPlotter, kwargs={}))
   trainer.register_callback(CallbackSpec(callback=ROCPlotter, kwargs={}))
   trainer.register_callback(CallbackSpec(callback=PPositiveAuxPlotter, kwargs={"column": "coincident_muons"}))

Writing a plotter
-----------------

Plotters are reducers. On each evaluation batch, the base class calls ``project``
once per output head to map that head's batch to rows, bins the rows into
histograms, and at the end of the split calls ``emit`` with the finished histograms
for each head. A new plotter subclasses ``BHistogramReducer`` (continuous axes) or
``CHistogramReducer`` (axes that are already class indices), both from
``icegraph.trainer.callbacks.base``, and implements:

* ``project(batch, out, label)``: ``out`` is the output of head ``label`` and
  ``batch`` is the full ``GraphBatch``. The head's target is
  ``batch.targets.block([label])``, and auxiliary columns are read the same way from
  ``batch.auxiliary``. Returns rows ``[B, D]``, one column per histogram axis,
  optionally with a group key per row ``[B]``; each group becomes its own series.
* ``_build_bins()``: the bin count per axis.
* ``_build_bounds(ctx, label)``: the axis ranges in linear space
  (``BHistogramReducer`` only). ``ctx`` is a ``BoundsConstructorContext`` giving the
  training statistics for a role (``ctx.get_stats(role)``) and a column's index
  within them (``ctx.physical_index(column, role)``).
* ``emit(trainer, artifacts, label)``: render the named histograms for ``label``.

.. code-block:: python

   def project(self, batch, out, label):
       target = batch.targets.block([label])       # this head's target
       aux = batch.auxiliary.block(["my_column"])  # an auxiliary column

       # residual against the auxiliary column, rows [B, 2]
       return torch.cat((aux, out - target), dim=1)
