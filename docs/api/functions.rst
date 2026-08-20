Function API
============

Math functions
--------------

.. automodule:: noisefloat.functions
   :members:
   :show-inheritance:

The functions in :mod:`noisefloat.functions` are thin sample-wise wrappers around
backend operations.  If the input is ``NFloat``, each stochastic sample is passed
through the function and then rounded.  If the input is a plain array or tensor,
the backend function is called directly.

Unary functions
~~~~~~~~~~~~~~~

.. list-table:: Unary functions
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Typical use
   * - ``sqrt``
     - Square roots and cancellation checks such as rationalized differences.
   * - ``exp`` / ``expm1``
     - Exponential formulas; prefer ``expm1`` near zero.
   * - ``log`` / ``log1p``
     - Logarithmic formulas; prefer ``log1p`` near zero.
   * - ``sin`` / ``cos`` / ``tan``
     - Trigonometric formulas and identities.
   * - ``asin`` / ``acos`` / ``atan``
     - Inverse trigonometric functions.
   * - ``sinh`` / ``cosh`` / ``tanh``
     - Hyperbolic functions.
   * - ``floor`` / ``ceil``
     - Discrete rounding checks.

Binary, selection, and reduction functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Array-style functions
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Typical use
   * - ``maximum`` / ``minimum``
     - Element-wise selection under stochastic perturbation.
   * - ``where``
     - Branch-sensitive array expressions.
   * - ``dot`` / ``matmul``
     - Dot products, matrix products, and linear algebra kernels.
   * - ``sum`` / ``mean``
     - Reductions and accumulation stability.
   * - ``norm``
     - Vector and matrix norm checks.

Scalar example
~~~~~~~~~~~~~~

.. code-block:: python

   from noisefloat import NFloat, configure, expm1, exp

   configure(n_samples=3, random_state=3)

   x = NFloat(1.0e-8)
   direct = exp(x) - NFloat(1.0)
   stable = expm1(x)

   print(direct.mean, direct.digits)
   print(stable.mean, stable.digits)

Array reduction example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from noisefloat import NFloat, configure, dot, sum

   configure(n_samples=3, random_state=9)

   x = NFloat(np.array([1.0, 1.0, 1.0, 1.0]))
   y = NFloat(np.array([1.0, -1.0, 1.0, -1.0 + 2.0**-40]))

   inner = dot(x, y)
   total = sum(x * y)

   print(inner.mean, inner.digits)
   print(total.mean, total.digits)

Backend selection helpers
-------------------------

.. automodule:: noisefloat.backends
   :members:
   :show-inheritance:

Backend behavior
~~~~~~~~~~~~~~~~

The active backend is chosen from the input whenever possible.  Plain Python
scalars and lists use ``configure(backend=...)``.  Mixed-backend operations are
converted through NumPy when a common backend cannot be preserved.
