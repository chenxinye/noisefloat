# noisefloat

[![PyPI Downloads](https://img.shields.io/pypi/dw/noisefloat?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/noisefloat/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/noisefloat?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/noisefloat/)

**Stochastic arithmetic for Python** with native NumPy, PyTorch, JAX, and TensorFlow backends.

---

## Overview

`noisefloat` provides a `NFloat` type that wraps floating-point values with
*n_samples* stochastic copies, using backend-native random rounding
to provide CESTAC-style discrete stochastic arithmetic in Python.

Each arithmetic operation is performed independently on every sample.
The spread of samples estimates **significant digit loss** due to rounding
errors, cancellation, or ill-conditioned computations.

### Design

`noisefloat` is an independent Python implementation of stochastic
floating-point perturbation.  It uses configurable backend-native software
quantisation rather than hardware rounding-mode switches:

| Feature | noisefloat |
|---|---|
| Language | Python |
| Rounding | pychop-style software random rounding (`rmode=6`) |
| Precision | configurable `exp_bits` / `sig_bits` |
| Integration | run-time |
| Backends | numpy / torch / jax / tensorflow |

---

## Installation

```bash
pip install numpy                     # required
pip install noisefloat                # from PyPI

# or editable install from source (the repo is named "noisyfloat"):
git clone https://github.com/chenxinye/noisyfloat.git
cd noisyfloat
pip install -e .
```

### Optional dependencies

```bash
pip install scipy      # better Student-t confidence values
pip install torch      # PyTorch backend
pip install jax jaxlib # JAX backend
pip install tensorflow # TensorFlow backend
```

---

## Quick-start

```python
import noisefloat as nf
from noisefloat import NFloat, NFloatSTE, configure, sqrt

# 1. Configure (once, globally)
configure(n_samples=3, random_state=42)

# 2. Wrap values in NFloat
a = NFloat(3.14)
print(a)           # NFloat(3.14 ± 1.38e-07, digits~7.0)
print(a.digits)    # ~7  (float32 precision)

# 3. Arithmetic – all ops produce new NFloat objects
b = NFloat(1.41421)
c = a + b
print(c.digits)

# 4. Math functions
r = sqrt(NFloat(2.0))
print(r.mean, r.std)

# 5. Arrays work too
import numpy as np
v = NFloat(np.array([1.0, 4.0, 9.0]))
print(sqrt(v).mean)   # [1. 2. 3.]

# 6. PyTorch tensors work natively
import torch
configure(backend="torch", random_state=42)
t = NFloat(torch.tensor(3.14))
print(t.mean, t.digits)

# 7. Differentiable rounding with STE (PyTorch)
x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
c = NFloatSTE(x)
loss = c.samples.sum()
loss.backward()
print(x.grad)   # gradients flow through stochastic rounding via STE
```

---

## Configuration

```python
from noisefloat import configure, get_config

configure(
    backend     = "numpy",   # "numpy" | "torch" | "jax" | "tensorflow"
    exp_bits    = 8,         # exponent bits (default: float32)
    sig_bits    = 23,        # significand bits (default: float32)
    n_samples   = 3,         # stochastic replicas per value
    random_state= 42,        # seed (None = non-reproducible)
    confidence  = 0.95,      # for significant-digit CI
    trace       = False,     # record diagnostics
)

cfg = get_config()
print(cfg.n_samples)
```

---

## Reproducibility

Pass the same `random_state` seed to get identical samples:

```python
configure(n_samples=3, random_state=42)
a1 = NFloat(3.14).samples.copy()

configure(n_samples=3, random_state=42)  # same seed
from noisefloat.core import _chopper; _chopper._config_hash = ()
a2 = NFloat(3.14).samples.copy()

assert (a1 == a2).all()   # reproducible
```

---

## Significant Digits (CESTAC formula)

```
digits = max(0, -log10(τ · std / (√n · |mean|)))
```

where τ is the Student-t critical value at `confidence` level.

- **std = 0** (all samples identical) → digits = 15 (fully stable)
- **mean = 0, std > 0** → digits = 0 (numerical noise)
- **NaN / Inf** → digits = 0

---

## Unstable Comparison Detection

```python
from noisefloat import NFloat, is_unstable_comparison
import numpy as np
import warnings

# Manually construct samples that cross (for illustration)
a = NFloat.__new__(NFloat)
a._samples = np.array([1.0, 2.0, 1.5])
b = NFloat.__new__(NFloat)
b._samples = np.array([1.7, 1.7, 1.7])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = a < b    # emits UnstableComparisonWarning
    print(result, [str(x.message) for x in w])

print(is_unstable_comparison(a, b, "lt"))  # True
```

---

## Diagnostics

```python
from noisefloat import configure, get_diagnostics, clear_diagnostics, print_diagnostics

configure(trace=True)

from noisefloat import NFloat
a = NFloat(2.0) + NFloat(3.0)

events = get_diagnostics()
print(len(events))   # 1

print_diagnostics()  # human-readable summary

clear_diagnostics()
```

---

## Instability Demos

### Catastrophic cancellation

```bash
python examples/sqrt_cancellation.py
```

```
x = 1e+04
  unstable: mean=5.002e-03  std=4.41e-06  digits~2.7
  stable:   mean=4.999e-03  std=0.00e+00  digits~15.0
  *** Cancellation detected! (12.3 digits lost) ***
```

### Quadratic roots

```bash
python examples/quadratic_roots.py
```

### Ill-conditioned Hilbert system

```bash
python examples/hilbert_system.py
```

### exp(x) − 1 instability

Compares unstable `exp(x) - 1`, a Taylor series, and the stable `expm1` for
small x where the subtraction of 1 causes cancellation.

```bash
python examples/expm1_instability.py
```

### Polynomial evaluation: naive vs Horner vs factored

Evaluates p(x) = (x − 1)⁴ expanded as a sum of monomials (naive), via
Horner's method, and in factored form.  Near x = 1 the naive form suffers
severe cancellation.

```bash
python examples/polynomial_evaluation.py
```

### Trigonometric cancellation: 1 − cos(x)

For small x, `1 - cos(x)` subtracts two nearly equal values.  The stable
identity `2·sin²(x/2)` avoids this cancellation.

```bash
python examples/trig_cancellation.py
```

### Dot product of nearly-orthogonal vectors

When two vectors are nearly orthogonal, their inner product is close to zero
but each element-wise product can be large — leading to massive cancellation
in the summation.

```bash
python examples/dot_product_accuracy.py
```

---

## Automatic Differentiation (STE)

`NFloatSTE` is a drop-in subclass of `NFloat` that uses a Straight-Through
Estimator so that gradients flow through stochastic rounding.  This enables
`loss.backward()` in PyTorch training loops.

```python
import torch
from noisefloat import NFloatSTE, configure

configure(backend="torch", n_samples=3, random_state=42)

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
c = NFloatSTE(x)

# Arithmetic works identically to NFloat
d = c * 2.0
loss = d.samples.sum()
loss.backward()

print(x.grad)  # tensor([6., 6., 6.], dtype=torch.float64)
```

> **Note**: gradient flow through `NFloatSTE` is supported for PyTorch, JAX,
> and TensorFlow. NumPy uses regular stochastic rounding because it has no
> native autograd system.

---

## Deep Learning Operator Analysis

`noisefloat.nn` provides NFloat-prefixed wrappers for common deep learning
operators.  A wrapped kernel receives a stochastic tensor with three samples
by default and evaluates each sample independently:

```text
output.x = function(input.x)
output.y = function(input.y)
output.z = function(input.z)
```

Each output sample is then stochastically rounded.  The representative value
is the sample mean (`NFloatTensor.value` / `NFloatTensor.to_tensor()`), and the
sample spread is used to report significant digits.

### Layer wrappers

PyTorch wrappers use the `NFloat*` prefix:

```python
import torch
from noisefloat import configure
from noisefloat.nn import NFloatLinear, NFloatReLU, NFloatTensor, get_kernel_reports

configure(backend="torch", n_samples=3, random_state=42)

model = torch.nn.Sequential(
    NFloatLinear(10, 32),
    NFloatReLU(),
    NFloatLinear(32, 2),
)

x = NFloatTensor(torch.randn(8, 10, dtype=torch.float64))
out = model(x)

print(out.value)       # representative mean tensor
print(out.digits)      # element-wise significant digits
print(get_kernel_reports()[-1])
```

Kernel reports are recorded with NFloat-prefixed names such as
`nfloat/Linear`, `nfloat/ReLU`, and `nfloat/LSTM`.

TensorFlow/Keras wrappers use the `TensorFlowNFloat*` prefix:

```python
from noisefloat import configure
from noisefloat.nn import TensorFlowNFloatDense, TensorFlowNFloatReLU

configure(backend="tensorflow", n_samples=3, random_state=42)

layer = TensorFlowNFloatDense(16)
activation = TensorFlowNFloatReLU()
```

### Automatic train/test iteration capture

Use `NFloatIterationTracker` to collect per-iteration kernel reports during
training or evaluation.  The tracker clears the previous reports at the start
of the iteration and captures every NFloat kernel report when the block exits.

```python
import torch
from noisefloat.nn import NFloatIterationTracker, NFloatLinear

tracker = NFloatIterationTracker("demo_train")
model = NFloatLinear(10, 2)

for step in range(10):
    batch = torch.randn(8, 10, dtype=torch.float64)
    with tracker.iteration(
        epoch=0,
        iteration=step,
        split="train",      # user-defined mode: "train", "test", "val", ...
        global_iteration=step,
        use_ste=False,      # set True to use straight-through rounding
        metadata={"lr": 1e-3},
    ):
        logits = model(batch)  # plain tensors are auto-converted in the block

tracker.export("./outputs/demo")
```

The export writes:

- `kernel_digits.jsonl`
- `kernel_digits.csv`
- `summary.json`

Each row includes `epoch`, `iteration`, `global_iteration`, `split`,
`kernel_name`, `phase`, `avg_digits`, `min_digits`, `max_digits`,
`num_elements`, `is_stable`, and JSON metadata including `mode` and `use_ste`.

### Custom operator granularity

Use `nfloat_operator` when you want to choose a finer or coarser analysis
granularity than a predefined layer.  For example, you can analyze a full LSTM
layer as one kernel, or a single matrix multiply inside attention:

```python
import torch
from noisefloat.nn import NFloatTensor, nfloat_operator, get_kernel_reports

nfloat_matmul = nfloat_operator(torch.matmul, name="qk_matmul")

q = NFloatTensor(torch.randn(4, 8, 16, dtype=torch.float64))
k = torch.randn(4, 16, 8, dtype=torch.float64)

scores = nfloat_matmul(q, k)

print(scores.value)
print(get_kernel_reports()[-1].kernel_name)  # nfloat/qk_matmul
```

The same samplewise rule is used for custom operators:

```text
scores.x = matmul(q.x, k.x)
scores.y = matmul(q.y, k.y)
scores.z = matmul(q.z, k.z)
```

Floating tensor outputs are wrapped as `NFloatTensor`; tuple/list/dict outputs
are supported recursively.

---

## API Reference

### `NFloat`

| Attribute / method | Description |
|---|---|
| `.samples` | Raw stochastic samples array (n_samples, ...) |
| `.mean` | Sample mean |
| `.value` | Representative value, equal to the sample mean |
| `.std` | Sample standard deviation (ddof=1) |
| `.var` | Sample variance (ddof=1) |
| `.digits` | CESTAC significant digit estimate |
| `.significant_digits()` | Same as `.digits` |
| `.confidence_interval()` | (lower, upper) at `config.confidence` |
| `.is_numerical_zero()` | True if result is likely a numerical zero |
| `.rel_error_estimate` | std / abs(mean) |
| `.backend_name` | Active backend: `"numpy"`, `"torch"`, `"jax"`, or `"tensorflow"` |

Arithmetic: `+`, `-`, `*`, `/`, `**`, `neg`, `abs` (and `r*` variants).

Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=` — emit
`UnstableComparisonWarning` when samples disagree.

### `NFloatSTE`

Subclass of `NFloat` that uses a Straight-Through Estimator for differentiable
stochastic rounding.  All `NFloat` attributes and methods are inherited.
Gradient flow through rounding is supported for PyTorch, JAX, and TensorFlow.

### Math functions

`sqrt, exp, log, log1p, expm1, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, floor, ceil, maximum, minimum, where, dot, matmul, sum, mean, norm`

All accept `NFloat` or plain numpy arrays.

### Deep learning helpers

| Helper | Description |
|---|---|
| `NFloatTensor` | PyTorch stochastic tensor with `.samples`, `.digits`, `.value`, `.to_tensor()` |
| `NFloatLinear`, `NFloatConv*`, `NFloatLSTM`, ... | NFloat-prefixed PyTorch layer wrappers |
| `TensorFlowNFloatDense`, `TensorFlowNFloatConv*`, ... | NFloat-prefixed TensorFlow/Keras wrappers |
| `nfloat_analysis(...)` | Context manager for mode, prefix, auto-conversion, metadata, and STE |
| `NFloatIterationTracker` | Per-iteration capture/export of kernel significant-digit reports |
| `nfloat_operator(fn, name=...)` | Wrap a user-defined callable as a NFloat kernel |

---

## References
```bibtex
@misc{chen2026autos,
      title={Automated Numerical Stability Analysis of Deep Learning Operators}, 
      author={Xinye Chen},
      year={2026},
      eprint={2607.25494},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2607.25494}, 
}
```
---


## License

MIT – see [LICENSE](LICENSE).
