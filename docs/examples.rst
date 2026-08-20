Code examples
=============

This page groups the repository examples by the numerical question they answer.
Each script can be run from the repository root.  If the package is not installed
in editable mode, prefix commands with ``PYTHONPATH=src:.``.

Quick scalar check
------------------

.. code-block:: python

   from noisefloat import NFloat, configure, sqrt

   configure(n_samples=3, random_state=42)

   x = NFloat(2.0)
   y = sqrt(x)

   print("mean:", y.mean)
   print("std :", y.std)
   print("digits:", y.digits)

Use this pattern when checking one formula.  The mean is the reported value; the
standard deviation and digit estimate tell you whether the stochastic replicas
agree.

Catastrophic cancellation
-------------------------

Script: ``examples/sqrt_cancellation.py``

Application value:
   Tests whether an algebraic expression loses digits when two nearly equal
   quantities are subtracted.  This pattern appears in geometry, distance
   formulas, finite differences, and root computations.

What it compares:
   ``sqrt(x + 1) - sqrt(x)`` against the rationalized form
   ``1 / (sqrt(x + 1) + sqrt(x))``.

Run it:

.. code-block:: bash

   PYTHONPATH=src:. python examples/sqrt_cancellation.py

How to read it:
   The unstable form should show fewer significant digits as ``x`` grows.  The
   rationalized form keeps the samples much tighter because it avoids direct
   subtraction.

Quadratic roots
---------------

Script: ``examples/quadratic_roots.py``

Application value:
   Demonstrates why a textbook formula can be unsafe even for a simple
   polynomial.  The example is a useful template for validating closed-form
   expressions before using them in production code.

Core demonstration:

.. code-block:: python

   from noisefloat import NFloat, configure, sqrt

   configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=2)

   def bad_small_root(a, b, c):
       a, b, c = NFloat(a), NFloat(b), NFloat(c)
       disc = b * b - NFloat(4.0) * a * c
       return (-b - sqrt(disc)) / (NFloat(2.0) * a)

   def good_small_root(a, b, c):
       a, b, c = NFloat(a), NFloat(b), NFloat(c)
       disc = b * b - NFloat(4.0) * a * c
       return (NFloat(2.0) * c) / (-b + sqrt(disc))

   bad = bad_small_root(1, -1e4, 1)
   good = good_small_root(1, -1e4, 1)
   print(bad.mean, bad.digits)
   print(good.mean, good.digits)

Run it:

.. code-block:: bash

   PYTHONPATH=src:. python examples/quadratic_roots.py

Polynomial evaluation
---------------------

Scripts:
   ``examples/polynomial_evaluation.py`` and ``examples/rump83_polynomial.py``

Application value:
   Helps choose a numerically safer form of a polynomial.  This matters in
   interpolation, calibration curves, signal processing, and generated code where
   expanded polynomials are common.

What they show:
   ``polynomial_evaluation.py`` compares expanded, Horner, and factored forms of
   a polynomial near a cancellation point.  ``rump83_polynomial.py`` uses Rump's
   well-known example, where ordinary floating-point arithmetic can produce a
   misleading result.

Run them:

.. code-block:: bash

   PYTHONPATH=src:. python examples/polynomial_evaluation.py
   PYTHONPATH=src:. python examples/rump83_polynomial.py

Special functions near zero
---------------------------

Scripts:
   ``examples/expm1_instability.py``, ``examples/log1p_instability.py``, and
   ``examples/trig_cancellation.py``

Application value:
   Checks whether a formula should use a numerically stable library function or
   identity.  These cases occur in probability, optimization, statistics, and
   loss functions.

Typical lessons:

* use ``expm1(x)`` instead of ``exp(x) - 1`` for small ``x``;
* use ``log1p(x)`` instead of ``log(1 + x)`` for small ``x``;
* use ``2 * sin(x / 2) ** 2`` instead of ``1 - cos(x)`` near zero.

Run them:

.. code-block:: bash

   PYTHONPATH=src:. python examples/expm1_instability.py
   PYTHONPATH=src:. python examples/log1p_instability.py
   PYTHONPATH=src:. python examples/trig_cancellation.py

Linear algebra and reductions
-----------------------------

Scripts:
   ``examples/hilbert_system.py``, ``examples/dot_product_accuracy.py``,
   ``examples/unstable_sum.py``, ``examples/alternating_harmonic.py``, and
   ``examples/sum_cubes_precision.py``

Application value:
   Diagnoses loss of accuracy in solvers, dot products, and long reductions.
   These examples are relevant to simulation codes, statistics, optimization,
   and any pipeline that accumulates many terms.

Hilbert system example:

.. code-block:: python

   from noisefloat import NFloat, configure

   configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=6)

   a00 = NFloat(1.0)
   a01 = NFloat(1.0) / NFloat(2.0)
   a10 = NFloat(1.0) / NFloat(2.0)
   a11 = NFloat(1.0) / NFloat(3.0)

   determinant = a00 * a11 - a01 * a10
   print(determinant.mean, determinant.digits)

Run the full scripts:

.. code-block:: bash

   PYTHONPATH=src:. python examples/hilbert_system.py
   PYTHONPATH=src:. python examples/dot_product_accuracy.py
   PYTHONPATH=src:. python examples/unstable_sum.py

C++ and cross-language precision checks
---------------------------------------

Script: ``examples/cpp_sum_precision.py``

Application value:
   Gives a compact way to compare a Python/noisefloat check with a C++-style
   summation example.  Use this when porting numerical kernels or explaining why
   the operation order matters.

Run it:

.. code-block:: bash

   PYTHONPATH=src:. python examples/cpp_sum_precision.py

Deep-learning operator stability
--------------------------------

Script: ``examples/operator_level_cestac_kernels.py``

Application value:
   Shows how operator-level stochastic arithmetic can flag unstable kernels in a
   model-like workload.  The cases include shifted versus naive softmax,
   logsumexp overflow, LayerNorm with near-zero variance, and attention with
   near-tied logits.

Run it:

.. code-block:: bash

   PYTHONPATH=src:. python examples/operator_level_cestac_kernels.py --precision both

The script writes a CSV report by default:

.. code-block:: text

   examples/outputs/operator_level_cestac_kernels.csv

Minimal custom operator example:

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatTensor, get_kernel_reports, nfloat_operator

   configure(backend="torch", n_samples=3, random_state=2026)

   stable_matmul = nfloat_operator(torch.matmul, name="qk_matmul")
   q = NFloatTensor(torch.randn(2, 4, 8, dtype=torch.float64))
   k = torch.randn(2, 8, 4, dtype=torch.float64)

   scores = stable_matmul(q, k)
   report = get_kernel_reports()[-1]

   print(scores.value.shape)
   print(report.kernel_name, report.avg_digits, report.is_stable)

CADNA-style verifications
-------------------------

Folder: ``verifications/``

Application value:
   Provides checked examples based on CADNA tutorial programs.  These are useful
   when validating noisefloat behavior against established stochastic-arithmetic
   demonstrations rather than newly invented examples.

Included cases:

1. Rump polynomial.
2. Second-order equation.
3. Determinant of Hilbert's matrix.
4. J.-M. Muller recurrence.
5. Newton's method for a polynomial root.
6. Gaussian elimination with partial pivoting.
7. Jacobi iterative method.

Run all cases:

.. code-block:: bash

   PYTHONPATH=src:. python verifications/cadna_examples_1_7.py

Run a subset:

.. code-block:: bash

   PYTHONPATH=src:. python verifications/cadna_examples_1_7.py --examples 1,2,3

Run the PyTorch version:

.. code-block:: bash

   PYTHONPATH=src:. python verifications/cadna_examples_1_7_torch.py

The verification output separates two ideas: what noisefloat detected in the
Python run, and the CADNA C reference counts used for comparison.  The goal is
qualitative agreement about sources of accuracy loss, especially cancellation,
unstable divisions, unstable powers, and computed zeros.

Choosing an example for your use case
-------------------------------------

.. list-table:: Example map
   :header-rows: 1
   :widths: 24 38 38

   * - If you care about
     - Start with
     - What to look at
   * - Cancellation in formulas
     - ``sqrt_cancellation.py``, ``trig_cancellation.py``
     - Digit gap between direct and stabilized identities
   * - Closed-form root formulas
     - ``quadratic_roots.py``
     - Direct root versus Vieta root
   * - Polynomial form
     - ``polynomial_evaluation.py``, ``rump83_polynomial.py``
     - Expanded versus Horner or factored form
   * - Solvers and matrices
     - ``hilbert_system.py``, verification cases 3 and 6
     - Digit loss as dimension or elimination proceeds
   * - Long sums or dot products
     - ``dot_product_accuracy.py``, ``unstable_sum.py``
     - Low digits in the final reduction
   * - Neural-network kernels
     - ``operator_level_cestac_kernels.py``
     - Per-kernel ``avg_digits`` and ``is_stable``
