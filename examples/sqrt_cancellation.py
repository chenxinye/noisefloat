"""
sqrt_cancellation.py – demonstrates how stochastic arithmetic detects
catastrophic cancellation in sqrt-based computations.

Classic example::

    f(x) = sqrt(x + 1) - sqrt(x)

For large x this suffers from severe cancellation.  The mathematically
stable form is::

    f_stable(x) = 1 / (sqrt(x + 1) + sqrt(x))

noisefloat reports lower significant digits for the unstable formula.
"""
import warnings
import noisefloat as nf
from noisefloat import NFloat, configure, sqrt

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=1)


def unstable_sqrt_diff(x):
    x = NFloat(x)
    return sqrt(x + NFloat(1.0)) - sqrt(x)


def stable_sqrt_diff(x):
    x = NFloat(x)
    return NFloat(1.0) / (sqrt(x + NFloat(1.0)) + sqrt(x))


print("=" * 60)
print("sqrt cancellation demo")
print("f(x) = sqrt(x+1) - sqrt(x)")
print("=" * 60)

x_values = [1e2, 1e4, 1e6, 1e8]

for x_val in x_values:
    a = unstable_sqrt_diff(x_val)
    b = stable_sqrt_diff(x_val)

    print(f"\nx = {x_val:.0e}")
    print(f"  unstable: mean={a.mean:.6e}  std={a.std:.2e}  digits~{a.digits:.1f}")
    print(f"  stable:   mean={b.mean:.6e}  std={b.std:.2e}  digits~{b.digits:.1f}")
    if a.digits < b.digits - 1:
        print(f"  *** Cancellation detected! "
              f"({b.digits - a.digits:.1f} digits lost) ***")
