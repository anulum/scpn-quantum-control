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

# Everything
pip install -e ".[dev,viz,ibm]"
```

## Requirements

- Python 3.9+
- Qiskit 1.0+
- qiskit-aer 0.14+
- NumPy 1.24+
- SciPy 1.10+
- NetworkX 3.0+

## Verify installation

```bash
python -c "import scpn_quantum_control; print('OK')"
pytest tests/ -x -q  # 208 tests should pass
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
