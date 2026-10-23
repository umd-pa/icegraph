Cohen's Kappa
=============

Classification :doc:`metric <../../index>` computing the per-head Cohen's kappa.
Kappa scores agreement between prediction and truth against the
agreement two independent raters with the same marginals would reach by chance,
``(p_o - p_e) / (1 - p_e)``. A model that has only learned the class prior scores
around ``0`` however high its raw accuracy, which is what makes kappa informative
on skewed evaluation sets. Perfect agreement is ``1``, systematic disagreement is
negative.

``weights`` generalizes this to ordered classes, charging a disagreement in
proportion to how far apart the two classes are. The default charges every
disagreement equally, which is plain Cohen's kappa.

Configuration
-------------

Selected as ``name: cohen-kappa``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``weights``
     - Disagreement cost between classes ``i`` and ``j``: ``none`` charges every
       disagreement equally, ``linear`` charges ``|i - j|`` and ``quadratic``
       charges ``(i - j)^2``. The ordered forms only make sense when the class
       indices are themselves ordered.
     - ``none`` | ``linear`` | ``quadratic``
     - ``none``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: cohen-kappa
           kwargs: { weights: quadratic }
