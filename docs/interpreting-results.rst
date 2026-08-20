Interpreting results
====================

The main output from ``noisefloat`` is a representative value plus a reliability
signal.  The representative value is usually the sample mean.  The reliability
signal is the spread across stochastic samples, reported as significant digits,
confidence intervals, diagnostics, or kernel reports.

Mean, samples, and spread
-------------------------

Every ``NFloat`` stores samples with a leading sample axis:

.. code-block:: python

   import numpy as np
   from noisefloat import NFloat, configure

   configure(n_samples=3, random_state=42)

   x = NFloat(np.array([1.0, 2.0]))
   print(x.samples.shape)  # (3, 2)
   print(x.mean)           # representative value
   print(x.std)            # spread across the three replicas

If the samples agree, the computation is insensitive to the configured rounding
perturbation.  If they spread out, the computation amplified rounding noise.

Significant digits
------------------

``digits`` estimates how many decimal digits are reliable.  It uses the CESTAC
formula:

.. code-block:: text

   digits = max(0, -log10(tau * std / (sqrt(n) * abs(mean))))

``tau`` is the Student-t critical value for the configured confidence level, and
``n`` is the number of samples.

Practical reading:

.. list-table:: Digit interpretation
   :header-rows: 1
   :widths: 24 76

   * - Digits
     - Interpretation
   * - near ``15``
     - Samples agree at double-precision scale, or all samples are identical.
   * - ``6`` to ``8``
     - Often fine for float32-style computations.
   * - ``3`` to ``5``
     - Usable in some contexts, but compare with a stabilized formula.
   * - ``1`` to ``2``
     - Treat as fragile.  Inspect cancellation, scaling, and branches.
   * - ``0``
     - Numerical noise, non-finite values, or a mean near zero with visible spread.

The exact cutoff is application-specific.  A PDE residual, a financial report,
and a neural-network activation do not need the same number of digits.

Confidence intervals
--------------------

``confidence_interval`` returns lower and upper bounds around the sample mean.
Use it when you want to know whether a value is distinguishable from zero under
stochastic rounding.

.. code-block:: python

   from noisefloat import NFloat, configure

   configure(n_samples=3, random_state=4, confidence=0.95)

   y = NFloat(1.0) - NFloat(1.0 + 2.0**-45)
   lo, hi = y.confidence_interval()

   print(y.mean)
   print(lo, hi)
   print(y.is_numerical_zero())

If the interval crosses zero, branch decisions based on the sign of that value
need extra care.

Warnings and diagnostics
------------------------

Warnings are immediate signals.  Diagnostics are stored events that you can
inspect later.

.. code-block:: python

   from noisefloat import NFloat, clear_diagnostics, configure, get_diagnostics

   configure(trace=True, n_samples=3, random_state=8)
   clear_diagnostics()

   value = NFloat(1.0 + 2.0**-40) - NFloat(1.0)

   for event in get_diagnostics():
       print(event.kind, event.operation, event.message)

Common responses:

``loss_of_accuracy_due_to_cancellation``
   Look for subtracting nearly equal quantities.  Try an algebraic identity,
   rescaling, or a library function such as ``log1p`` or ``expm1``.

``branching_instability``
   A comparison changed across samples.  Avoid making a hard branch on that value
   or add a tolerance that reflects the numerical uncertainty.

``mathematical_instability``
   Check for overflow, invalid operations, or non-finite values.

``intrinsic_instability``
   The operation produced too few reliable digits even if the source is not a
   simple cancellation pattern.

Kernel reports
--------------

For ``noisefloat.nn``, each report summarizes one wrapped kernel.

Important fields:

.. list-table:: Kernel report fields
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``kernel_name``
     - Name such as ``nfloat/Linear`` or a custom ``nfloat_operator`` name.
   * - ``phase``
     - Usually ``forward`` unless the wrapper records another phase.
   * - ``avg_digits``
     - Average element-wise digit count.
   * - ``min_digits``
     - Worst element in the output.
   * - ``max_digits``
     - Best element in the output.
   * - ``num_elements``
     - Number of output elements summarized.
   * - ``is_stable``
     - Boolean stability flag based on the configured threshold.

Example:

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatLinear, get_kernel_reports

   configure(backend="torch", n_samples=3, random_state=42)

   layer = NFloatLinear(4, 2)
   _ = layer(torch.randn(8, 4, dtype=torch.float64))

   report = get_kernel_reports()[-1]
   print(report.kernel_name, report.avg_digits, report.min_digits, report.is_stable)

When a model has many reports, sort by ``min_digits`` or inspect the exported CSV
from ``NFloatIterationTracker``.  Low minimum digits often reveal rare but
important unstable elements.

A diagnosis workflow
--------------------

1. Run the original calculation with ``NFloat`` or wrapped kernels.
2. Find outputs with low ``digits`` or kernel reports with low ``min_digits``.
3. Check whether the operation is a known pattern: cancellation, overflow,
   near-zero denominator, unstable branch, or ill-conditioned solve.
4. Compare against a stabilized version of the same calculation.
5. Repeat with another ``random_state`` and, if relevant, another precision
   setting.
6. Keep the stabilized algorithm when it consistently reports more reliable
   digits.

A low digit count does not always mean the final application result is wrong.  It
means the computation is sensitive to the configured rounding perturbation, so it
needs review before you rely on the printed digits.
