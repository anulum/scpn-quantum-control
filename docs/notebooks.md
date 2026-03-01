# Notebooks

Interactive Jupyter notebooks covering the core workflows. Run locally
(`pip install -e ".[dev]"`) or on Google Colab.

| Notebook | Description | Key Outputs | Prerequisites |
|----------|-------------|-------------|---------------|
| [01_kuramoto_xy_dynamics](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/01_kuramoto_xy_dynamics.ipynb) | Builds the XY Hamiltonian from a 4-oscillator Knm matrix, runs Trotter evolution, and compares quantum R(t) against the classical ODE. Extends to 8 oscillators. | R(t) trajectory plot, quantum-vs-classical overlay, 8-osc scaling | None |
| [02_vqe_ground_state](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/02_vqe_ground_state.ipynb) | Solves the Kuramoto-XY ground state via VQE with three ansatz variants (Knm-informed, TwoLocal, EfficientSU2). Compares 4-qubit and 8-qubit scaling. | Energy convergence curves, ansatz comparison table, optimal parameters | None |
| [03_error_mitigation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/03_error_mitigation.ipynb) | Demonstrates ZNE on a noisy Heron r2 simulator: unitary folding at scales 1/3/5, Richardson extrapolation, and higher-order polynomial fit. | Extrapolation plot, ZNEResult dataclass, mitigated vs raw error | None |
| [04_upde_16_layer](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/04_upde_16_layer.ipynb) | Simulates the full 16-layer SCPN UPDE as a 16-qubit spin chain. Per-layer coherence analysis and time evolution over multiple Trotter steps. | Per-layer R bar chart, time evolution heatmap, global R trajectory | None |

All notebooks run on the local AerSimulator. No IBM credentials needed.

## Running locally

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```
