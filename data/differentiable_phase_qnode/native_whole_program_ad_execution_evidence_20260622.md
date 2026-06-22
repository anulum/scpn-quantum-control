# Native LLVM/JIT whole-program AD execution evidence

- artifact_id: `native-whole-program-ad-execution-20260622`
- beyond_scalar_executed: True
- executed_operation_families: ['scalar', 'determinant', 'inverse', 'solve', 'trace']
- max_value_error: 9.095e-13
- max_gradient_error: 3.411e-13 (tolerance 1.000e-06)
- fail_closed_boundaries: {'determinant': 20, 'inverse': 7, 'solve': 7}

| case | family | dim | status | value_error | gradient_error |
|------|--------|-----|--------|-------------|----------------|
| scalar_poly_3 | scalar | 3 | executed | 0.00e+00 | 0.00e+00 |
| determinant_2x2 | determinant | 2 | executed | 3.55e-15 | 8.88e-16 |
| determinant_3x3 | determinant | 3 | executed | 8.53e-14 | 1.07e-14 |
| determinant_4x4 | determinant | 4 | executed | 9.09e-13 | 3.41e-13 |
| inverse_2x2 | inverse | 2 | executed | 0.00e+00 | 6.94e-18 |
| inverse_3x3 | inverse | 3 | executed | 0.00e+00 | 6.94e-18 |
| solve_2x2 | solve | 2 | executed | 0.00e+00 | 1.39e-17 |
| solve_3x3 | solve | 3 | executed | 0.00e+00 | 6.94e-18 |
| trace_2x2 | trace | 2 | executed | 0.00e+00 | 0.00e+00 |
| trace_3x3 | trace | 3 | executed | 0.00e+00 | 0.00e+00 |
| determinant_20x20_fail_closed | determinant | 20 | fail_closed | — | — |
| inverse_7x7_fail_closed | inverse | 7 | fail_closed | — | — |

Claim boundary: Bounded SCPN-native llvmlite whole-program AD execution: static dense scalar and linear-algebra (determinant, inverse, linear solve, trace) value and gradient checked against the interpreted Program AD reference within float64 tolerance; no Enzyme, provider, hardware, performance, or beyond-declared-size promotion claim.
