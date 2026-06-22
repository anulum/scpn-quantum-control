# scpn-quantum-engine

Rust acceleration engine for
[scpn-quantum-control](https://github.com/anulum/scpn-quantum-control), built
with [PyO3](https://pyo3.rs/) and [rayon](https://github.com/rayon-rs/rayon).

The engine is an **optional** native extension. Every routine it exports has a
pure-Python (NumPy/Qiskit) fallback in `scpn-quantum-control`, so the library
runs correctly whether or not this extension is installed. When the extension is
present, the Python dispatchers prefer the native path and fall back silently if
a given export is unavailable.

## Installation

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
references. The Python side never requires the extension: a missing build, a
missing export, or a native error all resolve to the NumPy/Qiskit fallback.

## Licence

AGPL-3.0-or-later; a commercial licence is available. See the repository root
for licence terms and contact details.
