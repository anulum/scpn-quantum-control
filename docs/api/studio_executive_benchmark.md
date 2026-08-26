# Studio executive benchmark API

Module: `scpn_quantum_control.studio.executive_benchmark`

The module implements the Studio `benchmark` action for bounded dense
XY-Hamiltonian construction. Importing it performs no timing loop, filesystem
write, provider call, or hardware action.

## Construction functions

`reference_dense_xy_hamiltonian(coupling, frequencies)` builds the real dense
operator through NumPy. `coupling` must be an `n × n` matrix matching the
frequency count. The function uses little-endian Qiskit qubit ordering and
returns a `float64` matrix with shape `(2**n, 2**n)`.

`native_dense_xy_hamiltonian(coupling, frequencies)` calls the optional Rust
PyO3 kernel with the same ordering and output shape. It raises `RuntimeError`
when the extension or dense-construction symbol is unavailable.

`measure_p50_us(fn, *, warmup, repeats)` discards `warmup` calls, times exactly
`repeats` calls with `perf_counter_ns`, and returns their median in
microseconds. It does not isolate CPU affinity, host load, cache state, or
clock noise.

## Action handler

`BenchmarkActionHandler.plan()` accepts only `K_nm`, `omega`, `repeats`, and
`warmup`. The coupling matrix must be symmetric, square, zero-diagonal, and
contain between two and ten nodes. Frequencies must match the node count.
Repeats are bounded from one through 32; warm-up calls from zero through eight.
The default backend is `rust`.

`execute()` always measures the NumPy reference. On the Rust backend it also
checks native/reference parity with absolute tolerance `1e-9` and reports the
observed median ratio when the native median is positive. It appends the
validated committed benchmark-databank summary. Every result sets
`production_claim_allowed` to false and includes the shared-host timing caveat.

`generate_script()` embeds only deterministic sealed verdicts: operator shape,
native parity when applicable, and committed databank row count. The script
prints fresh and sealed reference timing values but never asserts their
equality.

## Evidence boundary

A passing action proves bounded operator construction, native/reference parity,
and committed-databank integrity for that execution. It is not an isolated
benchmark, stable speedup, production-performance claim, physical-coupling
claim, provider run, or QPU execution.

## Full autodoc

::: scpn_quantum_control.studio.executive_benchmark
    options:
      show_root_heading: false
      show_source: false
      members_order: source
