"""
expm1_instability.py – demonstrates instability of exp(x) - 1 for small x.

When x is very small, exp(x) ≈ 1 + x, so computing exp(x) - 1 directly
causes catastrophic cancellation: the subtraction of 1 discards the
significant bits of x.  Two alternatives are compared:

- Unstable: exp(x) - 1  via NFloat exp/subtract
- Stable series: x + x^2/2 + x^3/6  (three-term Taylor approximation)
- Stable built-in: noisefloat.expm1(x)  (delegates to np.expm1 which uses
  a compensated algorithm internally)

noisefloat reports higher digit loss for the unstable form at low precision.
"""
import warnings
import math
import noisefloat as nf
from noisefloat import NFloat, configure, exp, expm1

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=5)


def expm1_bad(x):
    """Unstable: compute exp(x) first, then subtract 1."""
    x = NFloat(x)
    return exp(x) - NFloat(1.0)


def expm1_series(x):
    """Three-term Taylor series: x + x^2/2 + x^3/6."""
    x = NFloat(x)
    return x + x * x / NFloat(2.0) + x * x * x / NFloat(6.0)


def expm1_stable(x):
    """Stable built-in expm1 (delegates to np.expm1)."""
    x = NFloat(x)
    return expm1(x)


print("=" * 60)
print("exp(x) - 1 instability demo")
print("=" * 60)

for x_val in [1e-1, 1e-3, 1e-5, 1e-7]:
    bad    = expm1_bad(x_val)
    series = expm1_series(x_val)
    stable = expm1_stable(x_val)
    ref    = math.expm1(x_val)

    print(f"\nx = {x_val:.0e}  (true expm1 = {ref:.8e})")
    print(f"  exp(x)-1 bad:    mean={bad.mean:.8e}  std={bad.std:.2e}  "
          f"digits~{bad.digits:.1f}")
    print(f"  series approx:   mean={series.mean:.8e}  std={series.std:.2e}  "
          f"digits~{series.digits:.1f}")
    print(f"  expm1 stable:    mean={stable.mean:.8e}  std={stable.std:.2e}  "
          f"digits~{stable.digits:.1f}")
    if bad.digits < stable.digits - 1:
        print(f"  *** Instability detected! ({stable.digits - bad.digits:.1f} digits "
              f"lost in bad form) ***")
