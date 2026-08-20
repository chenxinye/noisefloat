Diagnostics API
===============

.. automodule:: noisefloat.diagnostics
   :members:
   :show-inheritance:

When to enable diagnostics
--------------------------

Use diagnostics when a single digit count is not enough and you need to know
where an unstable result was produced.  Diagnostics are especially helpful in
long algorithms, loops, verification scripts, and tests.

.. code-block:: python

   from noisefloat import (
       NFloat,
       clear_diagnostics,
       configure,
       get_diagnostics,
       get_diagnostics_summary,
       print_diagnostics,
   )

   configure(trace=True, n_samples=3, random_state=42)
   clear_diagnostics()

   a = NFloat(1.0)
   b = NFloat(1.0 + 2.0**-40)
   _ = b - a

   print_diagnostics()
   print(get_diagnostics_summary())

   for event in get_diagnostics():
       print(event.kind, event.operation, event.digits, event.location)

Main helpers
------------

.. list-table:: Diagnostic helpers
   :header-rows: 1
   :widths: 30 70

   * - Helper
     - Use
   * - ``get_diagnostics()``
     - Return recorded ``DiagnosticsEvent`` objects.
   * - ``get_diagnostics_summary()``
     - Return aggregate counts by diagnostic kind.
   * - ``clear_diagnostics()``
     - Remove previously recorded events.
   * - ``print_diagnostics()``
     - Print a human-readable event list.
   * - ``print_diagnostics_summary()``
     - Print aggregate counts.

Event fields
------------

``DiagnosticsEvent`` records the diagnostic kind, operation name, message,
optional digit estimate, and source location.  Extra details may include input
and output summaries.  Treat the event list as a debugging aid rather than as a
stable file format; for deep-learning kernel exports, use
``NFloatIterationTracker`` instead.

Typical diagnostic kinds
------------------------

.. list-table:: Common diagnostics
   :header-rows: 1
   :widths: 34 66

   * - Kind
     - Meaning
   * - ``loss_of_accuracy_due_to_cancellation``
     - Addition or subtraction lost a large number of digits.
   * - ``branching_instability``
     - Stochastic samples disagree about a comparison or branch.
   * - ``mathematical_instability``
     - Non-finite values or unstable intrinsic functions were detected.
   * - ``intrinsic_instability``
     - The operation result has too few reliable digits.

The CADNA verification scripts in ``verifications/`` use these diagnostics to
compare noisefloat-detected sources of accuracy loss with CADNA C reference
outputs.
