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

.. code-block:: python

   from icegraph.trainer.callbacks import CallbackSpec, ParityPlotter, ROCPlotter

   trainer.register_callback(CallbackSpec(callback=ParityPlotter, kwargs={}))
   trainer.register_callback(CallbackSpec(callback=ROCPlotter, kwargs={}))
