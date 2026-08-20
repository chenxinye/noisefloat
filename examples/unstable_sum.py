"""
unstable_sum.py – demonstrates floating-point instability in summation order.

Classic example::

    1e8 + 1 - 1e8

The mathematical result is 1, but adding the large constant first then
subtracting it can cause the small term to vanish.

We compare:
- bad_sum:  1e8 + 1.0 - 1e8  (small term added between two large terms)
- good_sum: 1e8 - 1e8 + 1.0  (large terms cancel first, then add small term)

Note: if all stochastic samples deterministically lose the small term,
NFloat/CESTAC-style tools may not detect the error (no sample disagreement).
This example illustrates both the instability and that limitation.

A second demo shows Kahan compensated summation versus naive summation for a
longer sequence with many small terms.
"""
import warnings
import numpy as np
import noisefloat as nf
from noisefloat import NFloat, configure

warnings.filterwarnings("ignore")

configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=4)


def bad_sum():
    return NFloat(1e8) + NFloat(1.0) - NFloat(1e8)


def good_sum():
    return NFloat(1e8) - NFloat(1e8) + NFloat(1.0)


print("=" * 60)
print("Unstable sum demo")
print("1e8 + 1.0 - 1e8  (true result = 1.0)")
print("=" * 60)

bad  = bad_sum()
good = good_sum()

print(f"\nbad sum:  mean={bad.mean:.6f}  std={bad.std:.2e}  digits~{bad.digits:.1f}")
print(f"good sum: mean={good.mean:.6f}  std={good.std:.2e}  digits~{good.digits:.1f}")
print()
print("Note: if all samples agree on the wrong answer (0.0), digits may still")
print("appear high because stochastic rounding tracks uncertainty, not bias.")

# ---- Kahan vs naive for a longer sequence --------------------------------- #
print("\n" + "=" * 60)
print("Kahan compensated sum vs naive sum")
print("=" * 60)

configure(exp_bits=8, sig_bits=10, n_samples=3, random_state=4)

N = 100
big = 1e8
rng = np.random.default_rng(0)
small_vals = rng.uniform(0.1, 1.0, N)
sequence = [big] + list(small_vals) + [-big]
true_sum = float(np.sum(small_vals))


def naive_seq_sum(values):
    total = NFloat(0.0)
    for v in values:
        total = total + NFloat(v)
    return total


def kahan_seq_sum(values):
    total = NFloat(0.0)
    comp  = NFloat(0.0)
    for v in values:
        y     = NFloat(v) - comp
        t     = total + y
        comp  = (t - total) - y
        total = t
    return total


naive = naive_seq_sum(sequence)
kahan = kahan_seq_sum(sequence)

print(f"\nSequence: {big:.0e}, [100 values in (0.1, 1)], -{big:.0e}")
print(f"True sum of small values: {true_sum:.6f}")
print(f"\nNaive sum:  {naive.mean:.6f}  (err={abs(naive.mean - true_sum):.2e})  "
      f"digits~{naive.digits:.1f}")
print(f"Kahan sum:  {kahan.mean:.6f}  (err={abs(kahan.mean - true_sum):.2e})  "
      f"digits~{kahan.digits:.1f}")
