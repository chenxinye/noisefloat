"""
polynomial_evaluation.py – compares naive vs Horner's method for polynomials.

Evaluating a polynomial by computing each power separately and summing
can accumulate significant rounding error, especially when terms nearly
cancel.  Horner's method restructures the computation to minimise the
number of multiplications and additions, reducing error propagation.

Example polynomial::

    p(x) = x^4 - 4x^3 + 6x^2 - 4x + 1  =  (x - 1)^4

Evaluated near x = 1 the naive form suffers severe cancellation because
each monomial is ≈ 1 while the true answer is ≈ 0.

noisefloat reveals the digit loss in the naive evaluation.
"""
import warnings
import noisefloat as nf
from noisefloat import NFloat, configure

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=7)


def poly_naive(x):
    """Naive monomial evaluation: c0 + c1*x + c2*x^2 + ... """
    x = NFloat(x)
    return (x ** NFloat(4.0)
            - NFloat(4.0) * x ** NFloat(3.0)
            + NFloat(6.0) * x ** NFloat(2.0)
            - NFloat(4.0) * x
            + NFloat(1.0))


def poly_horner(x):
    """Horner's method: ((((x - 4)*x + 6)*x - 4)*x + 1)."""
    x = NFloat(x)
    r = x - NFloat(4.0)
    r = r * x + NFloat(6.0)
    r = r * x - NFloat(4.0)
    r = r * x + NFloat(1.0)
    return r


def poly_factored(x):
    """Factored form: (x - 1)^4 — exact grouping, minimal cancellation."""
    x = NFloat(x)
    d = x - NFloat(1.0)
    return d * d * d * d


print("=" * 60)
print("Polynomial evaluation: naive vs Horner vs factored")
print("p(x) = x^4 - 4x^3 + 6x^2 - 4x + 1  =  (x - 1)^4")
print("=" * 60)

x_values = [1.0001, 1.001, 1.01, 1.1, 2.0]

for x_val in x_values:
    ref = (x_val - 1.0) ** 4

    naive    = poly_naive(x_val)
    horner   = poly_horner(x_val)
    factored = poly_factored(x_val)

    print(f"\nx = {x_val}  (true value = {ref:.8e})")
    print(f"  naive:    mean={naive.mean:.8e}  std={naive.std:.2e}  "
          f"digits~{naive.digits:.1f}")
    print(f"  horner:   mean={horner.mean:.8e}  std={horner.std:.2e}  "
          f"digits~{horner.digits:.1f}")
    print(f"  factored: mean={factored.mean:.8e}  std={factored.std:.2e}  "
          f"digits~{factored.digits:.1f}")
    if naive.digits < factored.digits - 1:
        print(f"  *** Naive evaluation lost {factored.digits - naive.digits:.1f} "
              f"digits vs factored form ***")
