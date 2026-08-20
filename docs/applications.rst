Applications
============

``noisefloat`` is useful when a number is not enough and you also need to know
how much of that number survived floating-point error.  The examples in this
repository are small, but they map directly to common numerical workloads.

Scientific computing
--------------------

Scientific codes often mix closed-form formulas, reductions, matrix solves, and
iterative methods.  A stable-looking output can still carry only a few reliable
digits.  ``NFloat`` lets you run the same formula under stochastic rounding and
inspect where the sample spread grows.

Use it for:

* checking cancellation in derived formulas;
* comparing equivalent discretizations or update rules;
* finding unstable residuals in solvers;
* deciding whether an algorithm needs rescaling, pivoting, or a stable identity.

Repository examples:

* ``examples/sqrt_cancellation.py`` for algebraic reformulation;
* ``examples/hilbert_system.py`` for ill-conditioned linear systems;
* ``verifications/cadna_examples_1_7.py`` for CADNA-style solver and recurrence
  checks.

Minimal pattern:

.. code-block:: python

   from noisefloat import NFloat, configure, sqrt

   configure(exp_bits=8, sig_bits=23, n_samples=3, random_state=21)

   def direct(x):
       x = NFloat(x)
       return sqrt(x + NFloat(1.0)) - sqrt(x)

   def stable(x):
       x = NFloat(x)
       return NFloat(1.0) / (sqrt(x + NFloat(1.0)) + sqrt(x))

   for value in [1e2, 1e4, 1e6]:
       a = direct(value)
       b = stable(value)
       print(value, a.digits, b.digits)

Numerical algorithm design
--------------------------

When two algorithms produce similar means, compare their digit counts.  The
algorithm with more stable samples is usually the safer implementation.

Good candidates:

* direct formula versus transformed formula;
* expanded polynomial versus Horner or factored form;
* naive summation versus compensated or reordered summation;
* Gaussian elimination variants;
* fixed-point, Newton, and Jacobi iterations.

Repository examples:

* ``examples/quadratic_roots.py`` compares the direct and Vieta forms;
* ``examples/polynomial_evaluation.py`` compares expanded, Horner, and factored
  polynomial evaluation;
* CADNA verification cases 4, 5, and 7 cover recurrence and iterative methods.

Decision example:

.. code-block:: python

   from noisefloat import NFloat, configure

   configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=12)

   def expanded(x):
       x = NFloat(x)
       return x**4 - NFloat(4) * x**3 + NFloat(6) * x**2 - NFloat(4) * x + NFloat(1)

   def factored(x):
       x = NFloat(x)
       return (x - NFloat(1)) ** 4

   a = expanded(1.0001)
   b = factored(1.0001)

   print("expanded", a.mean, a.digits)
   print("factored", b.mean, b.digits)

Machine learning and deep learning
----------------------------------

Deep-learning models can hide numerical problems inside kernels.  A model may
still produce logits, but a softmax, normalization layer, attention block, or
loss can have poor numerical reliability for some batches.  ``noisefloat.nn``
records per-kernel digit estimates so you can inspect these weak points.

Use it for:

* detecting overflow-prone softmax/logsumexp implementations;
* checking normalization when activations have very small variance;
* finding attention heads that make near-tie decisions;
* comparing standard and arithmetic-level NFloat wrappers;
* exporting layer-level digit traces during training or evaluation.

Repository examples:

* ``examples/operator_level_cestac_kernels.py`` for a compact softmax,
  normalization, and attention benchmark;
* ``exps/*digits_th.py`` and ``exps/*digits_tf.py`` for model-level probes;
* ``exps/full_cestac_*`` for more detailed arithmetic-level analyses.

Compact operator probe:

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatTensor, get_kernel_reports, nfloat_operator

   configure(backend="torch", n_samples=3, random_state=2026)

   def shifted_softmax(samples):
       shifted = samples - samples.max(dim=-1, keepdim=True).values
       values = torch.exp(shifted)
       return values / values.sum(dim=-1, keepdim=True)

   softmax = nfloat_operator(shifted_softmax, name="shifted_softmax")
   logits = NFloatTensor(torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=torch.float64))

   probabilities = softmax(logits)
   report = get_kernel_reports()[-1]

   print(probabilities.value)
   print(report.avg_digits, report.is_stable)

Verification and regression testing
-----------------------------------

The ``verifications/`` directory is useful when you want known instability cases
instead of an ad hoc example.  These scripts recreate CADNA tutorial programs and
compare the qualitative source of accuracy loss.

Run them after changes to arithmetic, diagnostics, or backend rounding:

.. code-block:: bash

   PYTHONPATH=src:. python verifications/cadna_examples_1_7.py
   PYTHONPATH=src:. python verifications/cadna_examples_1_7_torch.py

The scripts report noisefloat-detected sources separately from the CADNA C
reference.  Small numeric differences are expected because the random streams are
not the same.  The important result is whether both systems identify the same
kind of instability for the same numerical algorithm.

Finance, risk, and audit-style calculations
-------------------------------------------

Any workflow with long sums, near-cancelling cash flows, scenario aggregation, or
ill-conditioned calibration can benefit from a reliability check.  ``NFloat`` is
not a risk model.  It is a numerical audit tool that shows whether rounding can
change the digits you report.

Use it for:

* stress-testing portfolio or exposure aggregations;
* checking formulas that subtract nearly equal present values;
* validating calibration routines before reporting results;
* adding a precision note to an audit trail.

A simple reduction probe:

.. code-block:: python

   import numpy as np
   from noisefloat import NFloat, configure, sum

   configure(n_samples=3, random_state=19)

   cashflows = NFloat(np.array([1e8, -1e8, 3.25, 2.75, -1.10]))
   total = sum(cashflows)

   print(total.mean)
   print(total.digits)

When not to use noisefloat
--------------------------

``noisefloat`` does not prove a numerical error bound, and it does not make an
unstable algorithm stable.  It also adds runtime and memory overhead because each
value carries multiple samples.  Use it as a diagnostic pass, a regression check,
or an experiment tool; keep the original optimized code path for production once
you know which formulas or kernels need attention.
