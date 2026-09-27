# scpn-quantum-engine

Rust acceleration engine for
[scpn-quantum-control](https://github.com/anulum/scpn-quantum-control), built
with [PyO3](https://pyo3.rs/) and [rayon](https://github.com/rayon-rs/rayon).

The engine is an **optional** native extension. Every routine it exports has a
pure-Python (NumPy/Qiskit) fallback in `scpn-quantum-control`, so the library
runs correctly whether or not this extension is installed. When the extension is
present, configured Python dispatchers can select an admitted native path.
Required-native requests reject missing exports; optional fallback follows the
individual consumer's policy.

## Installation

The wheel declares `scpn-quantum-control>=1.1.0` as a runtime dependency.
Hamiltonian calls require a control build containing the shared execution-memory
policy. For source development, install the control and engine builds from the
same checkout; a dependency version alone does not establish kernel parity.

The wheel is built from this directory with
[maturin](https://www.maturin.rs/):

```bash
# Build a release wheel
maturin build --release --out dist

# Or build and install into the active environment in one step
maturin develop --release
```

The native module imports as `scpn_quantum_engine`:

```python
import scpn_quantum_engine as engine

engine.order_parameter(...)          # Kuramoto order parameter
engine.feedback_policy_batch(...)    # real-time feedback policy
engine.all_xy_expectations(...)      # transverse Pauli expectations
```

## Accelerated routines

The extension exports native kernels grouped by subsystem; each mirrors a
Python reference and is checked for parity in the
`scpn-quantum-control` test suite (`tests/test_rust_ffi_validation.py`,
`tests/test_rust_new_functions.py`):

- **Kuramoto dynamics** — Euler/trajectory integration, order parameter.
- **Coupling graphs** — `K_nm` construction, analog and hybrid coupling terms.
- **Spectral and sector analysis** — Koopman generator, magnetisation sectors,
  correlation matrices, OTOC, Lanczos coefficients, symmetry-decay fits.
- **Pauli and Lindblad operators** — fast expectation values, sparse order
  parameter, jump-operator assembly.
- **Real-time feedback** — batched feedback policy, sub-microsecond jitter
  tracking, closed-loop replay.
- **Error mitigation and QEC** — PEC coefficients and sampling, DLA protected
  subspace metrics.
- **Cryptography and entropy** — ML-DSA NTT/INTT, NIST SP 800-22 statistics
  (monobit, runs, longest-run, Berlekamp–Massey).
- **Compiler autodiff primitives** — value/JVP/VJP kernels for the linear
  algebra used by the differentiable compiler.

The canonical, generated catalogue is published at
<https://anulum.github.io/scpn-quantum-control/rust_engine/>.

## Parity and fallback contract

The native kernels are bit-for-bit or tolerance-matched against their Python
references. Missing optional exports may select the configured Python path.
An explicit Rust request or failure of an admitted native dense-Hamiltonian
dispatch propagates; it does not silently recompute with another backend.

Direct dense and sparse XY Hamiltonian calls check native dimension/byte
arithmetic, then use the installed control package's shared memory reservation.
The policy package must be importable. Output vectors use fallible allocation,
inherit active owner deadlines/cancellation and recheck lifecycle during basis
iteration and before return. Native output charges add to enclosing declarations;
inputs must remain unchanged during construction. Nonfinite constructed output
refuses. See [resource admission](../docs/stable_core_product.md) and the required
installed-engine cases in `tests/test_hamiltonian_native.py`. Build, numerical
parity and performance qualification apply to the exact source being distributed.

## Licence

AGPL-3.0-or-later; a commercial licence is available. See the repository root
for licence terms and contact details.

The browser replay kernel applies a 64 MiB product ceiling to cumulative declared
storage for each call. Owned input/output copies, parser metadata, replay tables
and numerical requests share that charge; an explicit Rust caller may tighten
it with `replay_value_and_gradient_with_memory_budget`. This is a policy ceiling,
not observed browser capacity or a memory reservation. Parent admission remains
binding, allocation failures refuse, and a refused FFI call leaves output bytes
unchanged. Independent subsequent calls start with a fresh charge.
