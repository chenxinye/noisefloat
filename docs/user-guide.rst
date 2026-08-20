User guide
==========

Basic workflow
--------------

A typical ``noisefloat`` session is short:

1. Configure the backend, sample count, precision, and random seed.
2. Wrap inputs with ``NFloat`` or ``NFloatTensor``.
3. Run the same algorithm you would normally run.
4. Read ``mean`` for the representative value and ``digits`` for reliability.
5. Turn on diagnostics or kernel tracking when you need a record of where digit
   loss occurred.

Configuration
-------------

``configure`` changes the global policy used by newly created stochastic values.
Set it once at the start of a script, then reset it when you want a different
backend or precision.

.. code-block:: python

   from noisefloat import configure, get_config, reset_chopper_cache

   configure(
       backend="numpy",        # "numpy", "torch", "jax", or "tensorflow"
       n_samples=3,            # three replicas match common CESTAC practice
       exp_bits=8,             # float32-like exponent range
       sig_bits=23,            # float32-like significand
       random_state=42,        # deterministic stochastic rounding
       confidence=0.95,
       trace=True,             # keep diagnostics events
   )
   reset_chopper_cache()       # useful after changing rounding settings

   cfg = get_config()
   print(cfg.backend, cfg.n_samples, cfg.sig_bits)

Useful configuration choices:

``n_samples``
   Three samples are enough for lightweight CESTAC-style checks.  More samples
   make the digit estimate smoother but increase runtime and memory use.

``exp_bits`` and ``sig_bits``
   These emulate a target floating-point format.  Use ``8``/``23`` for a
   float32-like run and ``11``/``52`` for a float64-like run.  Smaller
   significands are useful when you want an instability to appear in a compact
   demonstration.

``random_state``
   Fix the seed for examples, tests, and documentation.  Change it when you want
   to confirm that a diagnosis is not tied to one stochastic stream.

Scalar and array arithmetic
---------------------------

``NFloat`` behaves like a numeric value, but it keeps all stochastic samples.
Statistics are computed along the sample axis.

.. code-block:: python

   import numpy as np
   from noisefloat import NFloat, configure, sqrt

   configure(backend="numpy", n_samples=3, random_state=7)

   x = NFloat(np.array([1.0, 4.0, 9.0]))
   y = sqrt(x)

   print(y.mean)      # representative value
   print(y.std)       # sample spread
   print(y.digits)    # element-wise reliable digits

The same pattern works for arithmetic operators:

.. code-block:: python

   a = NFloat(3.14)
   b = NFloat(1.41421)
   c = (a + b) / NFloat(2.0)

   print(float(c))
   print(c.confidence_interval())

Comparing two algorithms
------------------------

The most useful workflow is to compare a fragile formula with a stabilized one.
This example follows ``examples/quadratic_roots.py``.

.. code-block:: python

   from noisefloat import NFloat, configure, sqrt

   configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=2)

   def direct_small_root(a, b, c):
       a, b, c = NFloat(a), NFloat(b), NFloat(c)
       disc = b * b - NFloat(4.0) * a * c
       return (-b - sqrt(disc)) / (NFloat(2.0) * a)

   def vieta_small_root(a, b, c):
       a, b, c = NFloat(a), NFloat(b), NFloat(c)
       disc = b * b - NFloat(4.0) * a * c
       return (NFloat(2.0) * c) / (-b + sqrt(disc))

   bad = direct_small_root(1, -1e4, 1)
   good = vieta_small_root(1, -1e4, 1)

   print("direct:", bad.mean, bad.digits)
   print("vieta :", good.mean, good.digits)

The direct formula subtracts nearly equal numbers.  The Vieta form avoids that
subtraction.  A large difference in ``digits`` is a stronger warning than the raw
mean values alone.

Numerical zero and confidence intervals
---------------------------------------

Some unstable expressions produce values close to zero.  ``is_numerical_zero``
checks whether the confidence interval contains zero or whether the digit count
falls below the configured threshold.

.. code-block:: python

   from noisefloat import NFloat, configure

   configure(n_samples=3, random_state=11)
   residual = NFloat(1.0) - NFloat(1.0 + 2.0**-52)

   print(residual.mean)
   print(residual.confidence_interval())
   print(residual.is_numerical_zero())

Diagnostics
-----------

Enable ``trace=True`` when you want a record of unstable operations.  This is
useful for debugging a longer algorithm or for comparing against verification
outputs.

.. code-block:: python

   from noisefloat import (
       NFloat,
       clear_diagnostics,
       configure,
       get_diagnostics,
       print_diagnostics_summary,
   )

   configure(trace=True, n_samples=3, random_state=5)
   clear_diagnostics()

   x = NFloat(1.0)
   y = NFloat(1.0 + 2.0**-40)
   _ = y - x

   for event in get_diagnostics():
       print(event.kind, event.operation, event.digits)

   print_diagnostics_summary()

Backends
--------

``noisefloat`` can infer the backend from the input value.  You can also set the
backend explicitly for plain Python values.

.. code-block:: python

   import torch
   from noisefloat import NFloat, configure

   configure(backend="torch", n_samples=3, random_state=42)

   t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
   x = NFloat(t)

   print(x.backend_name)
   print(x.samples.shape)
   print(x.digits)

Use NumPy for small numerical experiments.  Use PyTorch, JAX, or TensorFlow when
you want tensors to stay in the framework used by the rest of the pipeline.

Automatic differentiation
-------------------------

Use ``NFloatSTE`` when stochastic rounding sits inside a differentiable path.
The forward pass is rounded; the backward pass uses a straight-through estimator.

.. code-block:: python

   import torch
   from noisefloat import NFloatSTE, configure

   configure(backend="torch", n_samples=3, random_state=42)

   x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
   y = NFloatSTE(x) * 2.0
   loss = y.samples.sum()
   loss.backward()

   print(x.grad)

PyTorch kernel reports
----------------------

``noisefloat.nn`` wraps common layers and records CESTAC statistics for each
kernel.  A plain tensor passed inside an analysis block can be converted to an
``NFloatTensor`` automatically.

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import (
       NFloatIterationTracker,
       NFloatLinear,
       NFloatReLU,
       get_kernel_reports,
   )

   configure(backend="torch", n_samples=3, random_state=42)

   model = torch.nn.Sequential(
       NFloatLinear(10, 16),
       NFloatReLU(),
       NFloatLinear(16, 2),
   )
   tracker = NFloatIterationTracker("demo")

   for step in range(3):
       batch = torch.randn(8, 10, dtype=torch.float64)
       with tracker.iteration(epoch=0, iteration=step, split="train"):
           logits = model(batch)

   print(logits.value.shape)
   print(get_kernel_reports()[-1].avg_digits)
   tracker.export("examples/outputs/demo")

The export contains ``kernel_digits.csv``, ``kernel_digits.jsonl``, and
``summary.json``.  These files are suitable for plotting digit trajectories over
training or evaluation steps.

Running repository examples
---------------------------

The examples are regular Python scripts.  Run them from the repository root with
``PYTHONPATH`` pointing to ``src`` when the package is not installed.

.. code-block:: bash

   PYTHONPATH=src:. python examples/sqrt_cancellation.py
   PYTHONPATH=src:. python examples/quadratic_roots.py
   PYTHONPATH=src:. python examples/hilbert_system.py
   PYTHONPATH=src:. python examples/operator_level_cestac_kernels.py --precision both

For CADNA-style verification cases:

.. code-block:: bash

   PYTHONPATH=src:. python verifications/cadna_examples_1_7.py
   PYTHONPATH=src:. python verifications/cadna_examples_1_7.py --examples 1,2,3

Use the verification scripts when you want known numerical pathologies rather
than a hand-written one-off example.
