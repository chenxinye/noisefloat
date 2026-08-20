# CADNA Numerical Example Verifications

This directory contains Python/noisefloat recreations of the CADNA C/C++
tutorial examples used for source-of-accuracy-loss checks.

Each verification keeps the same numerical algorithm, precision regime, and
input data as the corresponding CADNA C/C++ program, then runs it with
`NFloat` so the resulting significant digits and noisefloat-detected
instability sources can be compared with the attached CADNA C output.  The
"Noisefloat detected source(s)" block is produced by the package; the
"CADNA C reference" block is the external reference used only for comparison.

Run all verifications with:

```bash
PYTHONPATH=src:. python verifications/cadna_examples_1_7.py
```

Run a subset with:

```bash
PYTHONPATH=src:. python verifications/cadna_examples_1_7.py --examples 1,2,3
```

The script uses a deterministic Python stochastic-rounding seed for
reproducibility.  CADNA C uses its own internal random stream, so the
verification prints both the noisefloat-detected source counts and the CADNA C
reference counts from the checked CADNA C output instead of copying the
reference into the detected result.
Use `--random-state` to repeat the same algorithms with a different Python
rounding stream.

For the Rump example, the verification counts only the two printed result
expressions.  The attached C file contains an unused intermediate assignment;
that statement may contribute to CADNA's process-wide counter, but it is not
part of the displayed result comparison.

For the Jacobi example, the default verification cap follows the CADNA C
tutorial program and allows up to 1000 iterations.  To run only this example:

```bash
PYTHONPATH=src:. python verifications/cadna_examples_1_7.py \
  --examples 7 \
  --jacobi-max-iterations 1000
```

The diagnostics are expected to have the same qualitative behavior as the CADNA
examples, especially for cancellation detection.  CADNA detects cancellation in
additions/subtractions when the operands carry at least four more reliable
digits than the result.  noisefloat mirrors this criterion with
`cancellation_digits_loss=4.0` and also reports CADNA-style unstable division,
multiplication, and power-function sources when the corresponding stochastic
operand has no reliable significant digit.

The examples are:

1. Rump polynomial.
2. Second-order equation.
3. Determinant of Hilbert's matrix.
4. J.-M. Muller recurrence.
5. Newton's method for a polynomial root.
6. Gaussian elimination with partial pivoting.
7. Jacobi iterative method.
