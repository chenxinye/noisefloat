"""
quadratic_roots.py – demonstrates instability in the quadratic formula.

For ``ax^2 + bx + c = 0`` with large negative *b*, the small root computed
via the standard (minus) branch suffers catastrophic cancellation because
``-b`` and ``sqrt(disc)`` are nearly equal.

Unstable formula (small root, large |b|)::

    x_small = (-b - sqrt(b^2 - 4ac)) / (2a)

Stable alternative – Vieta's identity recovers the small root without
subtracting two nearly-equal quantities::

    x_small = (2c) / (-b + sqrt(b^2 - 4ac))

noisefloat detects the instability through sample divergence and digit loss.

Note: exp_bits=8 is used here so that b^2 (= 1e8) stays within range.
"""
import warnings
import numpy as np
import noisefloat as nf
from noisefloat import NFloat, configure, sqrt

warnings.filterwarnings("ignore")

# Use exp_bits=8 (float32 exponent range) so b^2 = 1e8 does not overflow.
# sig_bits=10 gives ~3 decimal digits – enough to expose cancellation.
configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=2)


def quadratic_bad(a, b, c):
    """Small root via direct subtraction – catastrophic cancellation."""
    a = NFloat(a)
    b = NFloat(b)
    c = NFloat(c)
    disc = b * b - NFloat(4.0) * a * c
    return (-b - sqrt(disc)) / (NFloat(2.0) * a)


def quadratic_good(a, b, c):
    """Small root via Vieta's identity – no cancellation."""
    a = NFloat(a)
    b = NFloat(b)
    c = NFloat(c)
    disc = b * b - NFloat(4.0) * a * c
    return (NFloat(2.0) * c) / (-b + sqrt(disc))


print("=" * 60)
print("Quadratic roots demo: x^2 - 1e4*x + 1 = 0")
print("Small root ≈ 1e-4")
print("=" * 60)

# High-precision reference
b_val = -1e4
disc_ref = b_val ** 2 - 4.0
x_small_ref = (-b_val - np.sqrt(disc_ref)) / 2.0   # ≈ 1e-4

bad  = quadratic_bad(1, -1e4, 1)
good = quadratic_good(1, -1e4, 1)

print(f"\nTrue small root: {x_small_ref:.6e}")
print(f"bad  root: mean={bad.mean:.6e}  std={bad.std:.2e}  digits~{bad.digits:.1f}")
print(f"good root: mean={good.mean:.6e}  std={good.std:.2e}  digits~{good.digits:.1f}")
if bad.digits < good.digits - 1:
    print(f"*** Instability detected! ({good.digits - bad.digits:.1f} digits lost "
          f"in bad formula) ***")

print("\n--- Additional b values ---")
for b_val in [-1e2, -1e6]:
    bad2  = quadratic_bad(1, b_val, 1)
    good2 = quadratic_good(1, b_val, 1)
    disc_r = b_val ** 2 - 4.0
    ref_small = (-b_val - np.sqrt(disc_r)) / 2.0
    print(f"\nb = {b_val:.0e}  (true small root: {ref_small:.6e})")
    print(f"  bad  root: mean={bad2.mean:.6e}  digits~{bad2.digits:.1f}")
    print(f"  good root: mean={good2.mean:.6e}  digits~{good2.digits:.1f}")
