"""
alternating_harmonic.py – demonstrates error accumulation in the alternating
harmonic series.

The series::

    S_n = 1 - 1/2 + 1/3 - 1/4 + ... + (-1)^(k+1) / k

converges to log(2) ≈ 0.693147...

At low precision, rounding errors accumulate with each term.  As n grows,
noisefloat's samples diverge and the significant-digit count drops, showing
how stochastic rounding makes error propagation visible.
"""
import math
import warnings
import noisefloat as nf
from noisefloat import NFloat, configure

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=5)

TRUE_VALUE = math.log(2)


def alternating_harmonic(n):
    s = NFloat(0.0)
    for k in range(1, n + 1):
        term = NFloat(1.0) / NFloat(float(k))
        if k % 2 == 0:
            s = s - term
        else:
            s = s + term
    return s


print("=" * 60)
print("Alternating harmonic series: 1 - 1/2 + 1/3 - ...")
print(f"True limit: log(2) = {TRUE_VALUE:.8f}")
print("=" * 60)

for n in [10, 100, 1000, 5000]:
    s = alternating_harmonic(n)
    err = abs(s.mean - TRUE_VALUE)
    print(f"\nn={n:>5}: mean={s.mean:.8f}  std={s.std:.2e}  "
          f"digits~{s.digits:.1f}  err={err:.2e}")
