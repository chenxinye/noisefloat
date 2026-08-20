"""
trig_cancellation.py – demonstrates cancellation in 1 - cos(x) for small x.

For small x, cos(x) ≈ 1 - x^2/2, so computing 1 - cos(x) directly
subtracts two nearly equal values, losing significant digits.

Two stable alternatives are compared:

- Unstable: 1 - cos(x)
- Trig identity: 2 * sin(x/2)^2  (avoids the near-1 subtraction)
- Taylor series: x^2/2 - x^4/24  (two-term approximation)

noisefloat reports higher digit loss for the unstable form.
"""
import warnings
import math
import noisefloat as nf
from noisefloat import NFloat, configure, cos, sin

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=9)


def one_minus_cos_bad(x):
    """Unstable: 1 - cos(x), catastrophic cancellation for small x."""
    x = NFloat(x)
    return NFloat(1.0) - cos(x)


def one_minus_cos_trig(x):
    """Stable identity: 2 * sin(x/2)^2."""
    x = NFloat(x)
    s = sin(x / NFloat(2.0))
    return NFloat(2.0) * s * s


def one_minus_cos_taylor(x):
    """Two-term Taylor: x^2/2 - x^4/24."""
    x = NFloat(x)
    x2 = x * x
    return x2 / NFloat(2.0) - x2 * x2 / NFloat(24.0)


print("=" * 60)
print("1 - cos(x) cancellation demo")
print("=" * 60)

for x_val in [1e-1, 1e-2, 1e-3, 1e-4]:
    bad    = one_minus_cos_bad(x_val)
    trig   = one_minus_cos_trig(x_val)
    taylor = one_minus_cos_taylor(x_val)
    ref    = 1.0 - math.cos(x_val)

    print(f"\nx = {x_val:.0e}  (true value = {ref:.8e})")
    print(f"  1-cos(x) bad:    mean={bad.mean:.8e}  std={bad.std:.2e}  "
          f"digits~{bad.digits:.1f}")
    print(f"  2sin^2(x/2):     mean={trig.mean:.8e}  std={trig.std:.2e}  "
          f"digits~{trig.digits:.1f}")
    print(f"  Taylor approx:   mean={taylor.mean:.8e}  std={taylor.std:.2e}  "
          f"digits~{taylor.digits:.1f}")
    if bad.digits < trig.digits - 1:
        print(f"  *** Cancellation detected! ({trig.digits - bad.digits:.1f} digits "
              f"lost in bad form) ***")
