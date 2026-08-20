Deep learning API
=================

Main entrypoint
---------------

.. automodule:: noisefloat.nn
   :members:
   :show-inheritance:

``noisefloat.nn`` provides stochastic-precision analysis for neural-network
operators.  The PyTorch API is the most complete; TensorFlow/Keras wrappers are
also available for common layers.

PyTorch tensor type
-------------------

``NFloatTensor`` stores a leading sample axis and exposes tensor-oriented helpers.

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatTensor

   configure(backend="torch", n_samples=3, random_state=42)

   x = NFloatTensor(torch.randn(4, 10, dtype=torch.float64))
   print(x.samples.shape)   # (3, 4, 10)
   print(x.value.shape)     # (4, 10)
   print(x.digits.shape)    # (4, 10)

Layer wrappers
--------------

PyTorch wrappers use the ``NFloat`` prefix and can be assembled like ordinary
``torch.nn`` modules.

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatLinear, NFloatReLU, get_kernel_reports

   configure(backend="torch", n_samples=3, random_state=42)

   model = torch.nn.Sequential(
       NFloatLinear(10, 32),
       NFloatReLU(),
       NFloatLinear(32, 2),
   )

   batch = torch.randn(8, 10, dtype=torch.float64)
   output = model(batch)

   print(output.value)
   print(get_kernel_reports()[-1].avg_digits)

Common wrapper groups
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: PyTorch wrapper groups
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Examples
   * - Linear and convolution
     - ``NFloatLinear``, ``NFloatConv1d``, ``NFloatConv2d``, ``NFloatConv3d``
   * - Normalization
     - ``NFloatBatchNorm1d``, ``NFloatLayerNorm``, ``NFloatGroupNorm``
   * - Pooling
     - ``NFloatMaxPool2d``, ``NFloatAvgPool2d``, adaptive pooling wrappers
   * - Activations
     - ``NFloatReLU``, ``NFloatGELU``, ``NFloatSoftmax``, ``NFloatTanh``
   * - Recurrent and attention
     - ``NFloatRNN``, ``NFloatGRU``, ``NFloatLSTM``, ``NFloatMultiheadAttention``
   * - Transformer and losses
     - Transformer encoder/decoder layer wrappers and common loss wrappers

Analysis and reporting
----------------------

.. automodule:: noisefloat.nn.context
   :members:
   :show-inheritance:

.. automodule:: noisefloat.nn.report
   :members:
   :show-inheritance:

.. automodule:: noisefloat.nn.tracker
   :members:
   :show-inheritance:

Iteration tracking example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``NFloatIterationTracker`` when you want a CSV/JSONL record across training
or evaluation steps.

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatIterationTracker, NFloatLinear

   configure(backend="torch", n_samples=3, random_state=42)

   tracker = NFloatIterationTracker("classifier_probe")
   model = NFloatLinear(10, 2)

   for step in range(5):
       batch = torch.randn(8, 10, dtype=torch.float64)
       with tracker.iteration(
           epoch=0,
           iteration=step,
           split="train",
           global_iteration=step,
           metadata={"batch_size": 8},
       ):
           _ = model(batch)

   tracker.export("examples/outputs/classifier_probe")

The exported rows include the kernel name, phase, average/min/max digits,
element count, stability flag, split, epoch, iteration, and metadata.  This makes
it possible to plot which layers lose digits as training or inference proceeds.

Custom operator granularity
---------------------------

Use ``nfloat_operator`` when a predefined wrapper is too coarse or too fine.  The
operator receives sample tensors, runs the backend function sample by sample, and
records one kernel report.

.. code-block:: python

   import torch
   from noisefloat import configure
   from noisefloat.nn import NFloatTensor, get_kernel_reports, nfloat_operator

   configure(backend="torch", n_samples=3, random_state=2026)

   nfloat_matmul = nfloat_operator(torch.matmul, name="attention_scores")
   q = NFloatTensor(torch.randn(2, 4, 16, dtype=torch.float64))
   k = torch.randn(2, 16, 4, dtype=torch.float64)

   scores = nfloat_matmul(q, k)
   report = get_kernel_reports()[-1]

   print(scores.value.shape)
   print(report.kernel_name, report.min_digits, report.is_stable)

Arithmetic wrappers
-------------------

The ``ArithmeticNFloat*`` wrappers apply stochastic rounding at Python-visible
primitive operations inside supported modules.  Use them when you need a more
fine-grained view than one report per layer.  They are more intrusive and may be
slower than the standard ``NFloat*`` wrappers.

.. code-block:: python

   import torch
   from noisefloat.nn import ArithmeticNFloatLinear

   layer = ArithmeticNFloatLinear(16, 4)
   y = layer(torch.randn(2, 16, dtype=torch.float64))
   print(y.value.shape)

TensorFlow wrappers
-------------------

.. automodule:: noisefloat.nn.tensorflow
   :members:
   :show-inheritance:

TensorFlow/Keras wrappers use the ``TensorFlowNFloat`` prefix.

.. code-block:: python

   from noisefloat import configure
   from noisefloat.nn import TensorFlowNFloatDense, TensorFlowNFloatReLU

   configure(backend="tensorflow", n_samples=3, random_state=42)

   dense = TensorFlowNFloatDense(32)
   relu = TensorFlowNFloatReLU()

Deep-learning example scripts
-----------------------------

The repository contains larger experiment scripts under ``exps/`` and a compact
operator benchmark in ``examples/operator_level_cestac_kernels.py``.  Start with
the compact benchmark when learning the API; use the experiment scripts when you
need end-to-end model probes for image, language, translation, or segmentation
workloads.
