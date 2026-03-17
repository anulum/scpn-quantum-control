# Notebooks

Interactive Jupyter notebooks covering the core workflows. Run locally
(`pip install -e ".[dev]"`) or on Google Colab.

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| [01_kuramoto_xy_dynamics](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/01_kuramoto_xy_dynamics.ipynb) | XY Hamiltonian from 4-oscillator K_nm, Trotter evolution, quantum-vs-classical comparison | R(t) trajectory, quantum-classical overlay, 8-osc scaling |
| [02_vqe_ground_state](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/02_vqe_ground_state.ipynb) | VQE ground state with three ansatz variants (K_nm-informed, TwoLocal, EfficientSU2) | Energy convergence, ansatz comparison table |
| [03_error_mitigation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/03_error_mitigation.ipynb) | ZNE on noisy Heron r2 simulator: unitary folding at scales 1/3/5, Richardson extrapolation | Extrapolation plot, mitigated vs raw error |
| [04_upde_16_layer](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/04_upde_16_layer.ipynb) | Full 16-layer SCPN UPDE as 16-qubit spin chain, per-layer coherence analysis | Per-layer R bar chart, time evolution heatmap |
| [05_crypto_and_entanglement](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/05_crypto_and_entanglement.ipynb) | CHSH Bell test, correlator matrix, QKD QBER on K_nm topology | S-parameter measurement, key rate analysis |
| [06_pec_error_cancellation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/06_pec_error_cancellation.ipynb) | PEC quasi-probability decomposition, Monte Carlo sampling, overhead analysis | PEC vs raw vs exact comparison, overhead scaling plot |
| [07_quantum_advantage_scaling](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/07_quantum_advantage_scaling.ipynb) | Classical vs quantum timing at N=4,6,8,10,12, exponential fit crossover | Scaling plot, crossover prediction |

All notebooks run on the local AerSimulator. No IBM credentials needed.

## Running locally

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```
