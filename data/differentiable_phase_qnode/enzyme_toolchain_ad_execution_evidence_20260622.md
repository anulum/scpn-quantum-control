# Real Enzyme/LLVM reverse-mode AD execution evidence

- artifact_id: `enzyme-toolchain-ad-execution-20260622`
- toolchain_available: True
- toolchain: {'clang': 'Ubuntu clang version 18.1.3 (1ubuntu1)', 'opt': 'Ubuntu LLVM version 18.1.3', 'enzyme_plugin': 'LLVMEnzyme-18.so'}
- beyond_scalar_executed: True
- executed_operation_families: ['scalar', 'vector', 'matrix']
- max_gradient_error: 0.000e+00 (tolerance 1.000e-09)

| case | family | dim | status | gradient_error |
|------|--------|-----|--------|----------------|
| scalar_square | scalar | 1 | executed | 0.00e+00 |
| vector_sum_squares_4 | vector | 4 | executed | 0.00e+00 |
| vector_weighted_sum_4 | vector | 4 | executed | 0.00e+00 |
| matrix_trace_2x2 | matrix | 4 | executed | 0.00e+00 |
| matrix_frobenius_3x3 | matrix | 9 | executed | 0.00e+00 |

Claim boundary: Bounded real Enzyme/LLVM reverse-mode AD execution: scalar, vector and 2x2/3x3 matrix C kernels differentiated by the installed Enzyme pass and run natively, with the toolchain gradient checked against the analytic reference within float64 tolerance; no Enzyme-JAX, arbitrary-program, provider, hardware or performance claim.
