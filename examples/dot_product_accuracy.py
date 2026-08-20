"""
dot_product_accuracy.py – digit loss in nearly-orthogonal dot products.

When two vectors are nearly orthogonal their dot product is close to
zero, but each individual product a_i * b_i may be large.  The
summation therefore involves massive cancellation.

This example builds two vectors whose true dot product is a known
small value and compares:

- Naive: sum of elementwise products via NFloat arithmetic
- Built-in: noisefloat.dot (delegates to the backend's optimised routine)

noisefloat reveals how many significant digits survive in each case.
"""
import warnings
import numpy as np
import noisefloat as nf
from noisefloat import NFloat, configure, dot, sum as nf_sum

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=11)


def dot_naive(a, b):
    """Naive element-wise multiply and sum."""
    ca = NFloat(np.array(a))
    cb = NFloat(np.array(b))
    return nf_sum(ca * cb)


def dot_builtin(a, b):
    """Built-in noisefloat.dot."""
    ca = NFloat(np.array(a))
    cb = NFloat(np.array(b))
    return dot(ca, cb)


print("=" * 60)
print("Dot product accuracy demo")
print("Nearly-orthogonal vectors → cancellation in inner product")
print("=" * 60)

rng = np.random.default_rng(42)

for n in [10, 50, 100, 500]:
    # Construct a with known entries, then set b so that true dot = epsilon
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    # Make them nearly orthogonal by subtracting the projection
    proj = np.dot(a, b) / np.dot(a, a)
    b_orth = b - proj * a
    # Add a tiny known residual
    epsilon = 1e-6
    b_final = b_orth + epsilon * a / np.dot(a, a)
    true_dot = np.dot(a, b_final)

    naive   = dot_naive(a.tolist(), b_final.tolist())
    builtin = dot_builtin(a.tolist(), b_final.tolist())

    print(f"\nn = {n}  (true dot ≈ {true_dot:.6e})")
    print(f"  naive sum:   mean={naive.mean:.6e}  std={naive.std:.2e}  "
          f"digits~{naive.digits:.1f}")
    print(f"  built-in:    mean={builtin.mean:.6e}  std={builtin.std:.2e}  "
          f"digits~{builtin.digits:.1f}")
    if naive.digits < 2:
        print(f"  *** Severe digit loss in naive dot product! ***")
