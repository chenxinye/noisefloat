"""
hilbert_system.py – solving a Hilbert matrix system with noisefloat.

The Hilbert matrix H_n with H[i,j] = 1/(i+j+1) is famously ill-conditioned.
Solving H_n @ x = b via Gaussian elimination amplifies floating-point errors
rapidly as n grows.

noisefloat tracks significant-digit degradation across the solve using
element-wise NFloat arithmetic.
"""
import warnings
import noisefloat as nf
from noisefloat import NFloat, configure

warnings.filterwarnings("ignore")

configure(exp_bits=5, sig_bits=10, n_samples=3, random_state=6)


def make_hilbert_system(n):
    """Build the n×n Hilbert system using element-wise NFloat arithmetic.

    Returns (A, b) where A is a list-of-lists of NFloat scalars, b is a list
    of NFloat scalars, and the true solution is x = [1, 1, ..., 1].
    """
    A = [[NFloat(1.0) / NFloat(float(i + j + 1)) for j in range(n)]
         for i in range(n)]
    x_true = [NFloat(1.0) for _ in range(n)]

    b = []
    for i in range(n):
        s = NFloat(0.0)
        for j in range(n):
            s = s + A[i][j] * x_true[j]
        b.append(s)

    return A, b


def gaussian_elimination(A, b):
    """Naive Gaussian elimination without pivoting (element-wise NFloat)."""
    n = len(b)

    # Forward elimination
    for k in range(n):
        pivot = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            for j in range(k, n):
                A[i][j] = A[i][j] - factor * A[k][j]
            b[i] = b[i] - factor * b[k]

    # Back substitution
    x = [NFloat(0.0) for _ in range(n)]
    for i in reversed(range(n)):
        s = NFloat(0.0)
        for j in range(i + 1, n):
            s = s + A[i][j] * x[j]
        x[i] = (b[i] - s) / A[i][i]

    return x


print("=" * 60)
print("Hilbert system demo: H_n @ x = b, true x = ones")
print("=" * 60)

for n in [3, 4, 5]:
    A, b = make_hilbert_system(n)
    x = gaussian_elimination(A, b)

    print(f"\nn={n}:")
    for i, xi in enumerate(x):
        print(f"  x[{i}]: mean={xi.mean:.6f}  std={xi.std:.2e}  digits~{xi.digits:.1f}")
