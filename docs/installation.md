# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Installation

# Installation

## From PyPI

```bash
pip install scpn-quantum-control
```

## From source (development)

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
```

This installs pytest, ruff, and pytest-cov for development.

## Optional dependencies

```bash
# Visualization (matplotlib)
pip install -e ".[viz]"

# IBM Quantum hardware execution
pip install -e ".[ibm]"

# Rust acceleration (158-5401x faster Hamiltonian construction)
pip install scpn-quantum-engine

# Or build from source (requires Rust toolchain):
cd scpn_quantum_engine && maturin develop --release && cd ..

# Everything
pip install -e ".[dev,viz,ibm,rust]"
```

## Rust Acceleration

The optional `scpn-quantum-engine` package provides 15 Rust-accelerated functions
via PyO3. When installed, all analysis modules transparently use the Rust fast
paths. When not installed, everything works via pure Python/NumPy.

Pre-built wheels are available for Linux (x86_64, aarch64), macOS (x86_64, ARM),
and Windows (x64). See [Rust Engine docs](rust_engine.md) for the full API and
benchmark results.

## Requirements

- Python 3.10+
- Qiskit 1.0+
- qiskit-aer 0.14+
- NumPy 1.24+
- SciPy 1.10+
- NetworkX 3.0+

## Verify installation

```bash
python -c "import scpn_quantum_control; print('OK')"
pytest tests/ -x -q  # 2,715 tests should pass (13 skipped)
```

## IBM Quantum setup (optional)

Only needed for real hardware execution. See [Hardware Guide](hardware_guide.md).

```python
from scpn_quantum_control.hardware import HardwareRunner

# One-time: save your API token
HardwareRunner.save_token("your-ibm-quantum-token")

# Connect to hardware
runner = HardwareRunner()
runner.connect()
print(f"Backend: {runner.backend_name}")
```

Free tier: 10 minutes QPU time per month on ibm_fez (Heron r2, 156 qubits).
