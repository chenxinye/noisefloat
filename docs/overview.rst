Overview
========

What noisefloat is for
----------------------

``noisefloat`` is a small stochastic-arithmetic toolkit for finding where a
floating-point computation starts to lose useful digits.  It wraps scalar,
array, or tensor values in several randomly rounded replicas, runs the same
operation on each replica, and reads the sample spread as a reliability signal.

The result is not a replacement for higher precision.  It is a way to answer a
more practical question while developing numerical code:

   "Can I still trust the digits produced by this algorithm, layer, or branch?"

The package follows the NFloat/CESTAC idea of discrete stochastic arithmetic,
but it is implemented in Python with software rounding.  It works with NumPy for
numerical experiments and with PyTorch, JAX, and TensorFlow for tensor pipelines.

Why this matters
----------------

Many numerical bugs do not look like exceptions.  The program finishes, but the
answer has already lost most of its significant digits.  Common cases include:

* subtracting nearly equal values, as in ``sqrt(x + 1) - sqrt(x)``;
* evaluating a polynomial in an expanded form near a multiple root;
* choosing a branch from a comparison whose result changes under rounding;
* solving an ill-conditioned linear system;
* computing softmax, logsumexp, normalization, or attention weights near an
  unstable regime;
* comparing model variants when accuracy appears unchanged but internal kernels
  have very different numerical stability.

``noisefloat`` makes these cases visible.  A stable expression keeps the samples
clustered and reports many significant digits.  An unstable expression produces
larger sample spread, low digit counts, warnings, or kernel reports that point to
the failing operation.

Application areas
-----------------

Scientific and engineering codes
   Use ``NFloat`` around a formula, solver step, recurrence, or reduction to
   check whether the implemented algorithm is stable for the chosen inputs.  The
   scripts in ``examples/`` cover cancellation, quadratic roots, Hilbert systems,
   dot products, logarithmic/exponential special functions, and polynomial
   evaluation.

Numerical algorithm comparison
   Run two mathematically equivalent forms and compare their reported digits.
   For example, ``examples/quadratic_roots.py`` compares the direct quadratic
   formula with the Vieta-stabilized form for the small root.  The mean values
   may look close, while the digit count shows which formula is safer.

Deep-learning operator analysis
   Use ``noisefloat.nn`` to wrap PyTorch or TensorFlow layers and export per-kernel
   digit reports.  This is useful when checking softmax/logsumexp overflow,
   normalization on near-constant activations, attention logits with near ties,
   recurrent layers, or transformer blocks.

Verification against known instability examples
   The ``verifications/`` directory recreates CADNA tutorial examples in Python.
   These examples are useful regression checks because they cover well-known
   sources of accuracy loss: Rump's polynomial, second-order equations, Hilbert
   determinants, Muller recurrence, Newton iteration, Gaussian elimination, and
   Jacobi iteration.

Core ideas
----------

``NFloat``
   A stochastic numeric container.  It stores ``n_samples`` rounded replicas of a
   scalar, NumPy array, or backend tensor.  Arithmetic and math functions are
   evaluated sample by sample.

``NFloatSTE``
   A differentiable variant of ``NFloat``.  It uses a straight-through estimator
   so gradients can pass through stochastic rounding in autograd frameworks.

``digits``
   The estimated reliable decimal digits.  It is computed from the sample mean,
   sample standard deviation, sample count, and configured confidence level.
   High values mean the replicas agree; low values mean the result is sensitive
   to rounding.

Diagnostics
   Optional trace events and warnings record unstable comparisons, numerical
   zero classifications, non-finite values, and low-significance operations.

Kernel reports
   ``noisefloat.nn`` records average, minimum, and maximum significant digits for
   wrapped neural-network kernels.  Reports can be printed or exported for later
   plotting.

How to read the digit count
---------------------------

The exact threshold depends on the problem, but these rules of thumb are useful:

* ``digits`` near 15: the stochastic samples agree at double-precision scale;
* 6 to 8 digits: usually acceptable for many float32-style workflows;
* 1 to 3 digits: inspect the formula, input scaling, or branch condition;
* 0 digits: the result is dominated by numerical noise, non-finite values, or a
  near-zero mean with visible sample spread.

Use the digit count comparatively.  The strongest signal is often the gap between
two versions of the same calculation: direct versus stabilized formula, naive
sum versus compensated or reordered sum, naive softmax versus shifted softmax,
or one model kernel versus another.
