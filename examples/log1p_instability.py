"""
log1p_instability.py – demonstrates instability of log(1 + x) for small x.

When x is very small the intermediate result 1 + x loses the low-order bits
of x before the logarithm is applied.  Two alternatives are compared:

- Unstable: log(1 + x)  via NFloat(1) + x, then log(...)
- Stable series: x - x^2/2 + x^3/3  (three-term Taylor approximation)
- Stable built-in: noisefloat.log1p(x)  (delegates to np.log1p which uses a
  compensated algorithm internally)

noisefloat reports higher digit loss for the unstable form at low precision.
"""
import warnings
import noisefloat as nf
from noisefloat import NFloat, configure, log, log1p

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=3)


def log1p_bad(x):
    """Unstable: compute 1 + x first, losing low-order bits of x."""
    x = NFloat(x)
    return log(NFloat(1.0) + x)


def log1p_series(x):
    """Three-term Taylor series: x - x^2/2 + x^3/3."""
    x = NFloat(x)
    return x - x * x / NFloat(2.0) + x * x * x / NFloat(3.0)


def log1p_stable(x):
    """Stable built-in log1p (delegates to np.log1p)."""
    x = NFloat(x)
    return log1p(x)


print("=" * 60)
print("log(1 + x) instability demo")
print("=" * 60)

import math
for x_val in [1e-1, 1e-3, 1e-5, 1e-7]:
    bad    = log1p_bad(x_val)
    series = log1p_series(x_val)
    stable = log1p_stable(x_val)
    ref    = math.log1p(x_val)

    print(f"\nx = {x_val:.0e}  (true log1p = {ref:.8f})")
    print(f"  log(1+x) bad:    mean={bad.mean:.8f}  std={bad.std:.2e}  "
          f"digits~{bad.digits:.1f}")
    print(f"  series approx:   mean={series.mean:.8f}  std={series.std:.2e}  "
          f"digits~{series.digits:.1f}")
    print(f"  log1p stable:    mean={stable.mean:.8f}  std={stable.std:.2e}  "
          f"digits~{stable.digits:.1f}")
    if bad.digits < stable.digits - 1:
        print(f"  *** Instability detected! ({stable.digits - bad.digits:.1f} digits "
              f"lost in bad form) ***")
