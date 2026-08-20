Core API
========

Top-level package
-----------------

Most users import from ``noisefloat`` directly:

.. code-block:: python

   from noisefloat import NFloat, NFloatSTE, configure, sqrt

The top-level namespace includes:

* ``NFloat`` and ``NFloatSTE``;
* configuration helpers such as ``configure`` and ``get_config``;
* diagnostics helpers and warning classes;
* math functions from :mod:`noisefloat.functions`.

Configuration API
-----------------

.. automodule:: noisefloat.config
   :members:
   :show-inheritance:

Configuration example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from noisefloat import configure, get_config

   configure(
       backend="numpy",
       n_samples=3,
       exp_bits=8,
       sig_bits=23,
       random_state=42,
       confidence=0.95,
       trace=False,
   )

   cfg = get_config()
   print(cfg.backend, cfg.n_samples)

Common fields
~~~~~~~~~~~~~

.. list-table:: Configuration fields
   :header-rows: 1
   :widths: 22 28 50

   * - Field
     - Typical value
     - Meaning
   * - ``backend``
     - ``"numpy"`` / ``"torch"``
     - Backend used for plain Python inputs and software rounding.
   * - ``n_samples``
     - ``3``
     - Number of stochastic replicas per value.
   * - ``exp_bits``
     - ``8`` or ``11``
     - Exponent bits in the emulated format.
   * - ``sig_bits``
     - ``23`` or ``52``
     - Significand bits in the emulated format.
   * - ``random_state``
     - integer or ``None``
     - Seed for reproducible stochastic rounding.
   * - ``confidence``
     - ``0.95``
     - Confidence level used by digit and interval estimates.
   * - ``trace``
     - ``True`` or ``False``
     - Whether diagnostics events are recorded.

Core stochastic type
--------------------

.. automodule:: noisefloat.core
   :members:
   :show-inheritance:
   :no-index:

``NFloat`` usage notes
~~~~~~~~~~~~~~~~~~~~~~

``NFloat(value, n_samples=None)`` creates a stochastic value.  The raw samples
are stored with a leading sample axis.  For a scalar the shape is
``(n_samples,)``; for an array of shape ``(m, n)`` the sample shape is
``(n_samples, m, n)``.

.. code-block:: python

   import numpy as np
   from noisefloat import NFloat, configure

   configure(n_samples=3, random_state=42)

   x = NFloat(np.array([1.0, 2.0, 3.0]))
   print(x.samples.shape)
   print(x.mean)
   print(x.digits)

Key properties and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: ``NFloat`` quick reference
   :header-rows: 1
   :widths: 26 24 50

   * - Member
     - Return type
     - Use
   * - ``samples``
     - backend array/tensor
     - Raw stochastic replicas.
   * - ``n_samples``
     - ``int``
     - Number of replicas.
   * - ``shape``
     - ``tuple``
     - Value shape, excluding the sample axis.
   * - ``mean`` / ``value``
     - NumPy array or scalar
     - Representative value.
   * - ``std`` / ``var``
     - NumPy array or scalar
     - Sample spread with ``ddof=1``.
   * - ``digits``
     - NumPy array or scalar
     - Estimated reliable decimal digits.
   * - ``significant_digits()``
     - NumPy array or scalar
     - Same estimate as ``digits``.
   * - ``confidence_interval()``
     - pair of arrays/scalars
     - Lower and upper confidence bounds.
   * - ``is_numerical_zero()``
     - boolean mask
     - Whether the result should be treated as numerical zero.
   * - ``rel_error_estimate``
     - NumPy array or scalar
     - ``std / abs(mean)`` with a small epsilon.
   * - ``backend_name``
     - ``str``
     - Backend selected for this object.

Supported operators
~~~~~~~~~~~~~~~~~~~

``NFloat`` supports the arithmetic operators ``+``, ``-``, ``*``, ``/``, ``**``,
unary ``-``, and ``abs``.  Comparisons ``<``, ``<=``, ``>``, ``>=``, ``==``, and
``!=`` are also available.  Comparisons can emit ``UnstableComparisonWarning``
when different stochastic replicas disagree about the result.

.. code-block:: python

   import warnings
   from noisefloat import NFloat, UnstableComparisonWarning

   a = NFloat(1.0)
   b = NFloat(1.0 + 2.0**-40)

   with warnings.catch_warnings(record=True) as captured:
       warnings.simplefilter("always", UnstableComparisonWarning)
       decision = a < b

   print(decision)
   print([str(item.message) for item in captured])

``NFloatSTE`` usage notes
~~~~~~~~~~~~~~~~~~~~~~~~~

``NFloatSTE`` has the same public interface as ``NFloat`` but uses a
straight-through estimator during rounding.  Use it when the stochastic value
participates in a differentiable computation.

.. code-block:: python

   import torch
   from noisefloat import NFloatSTE, configure

   configure(backend="torch", n_samples=3, random_state=42)

   x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
   y = NFloatSTE(x) * 3.0
   y.samples.sum().backward()

   print(x.grad)

Exceptions and warnings
-----------------------

.. automodule:: noisefloat.exceptions
   :members:
   :show-inheritance:
