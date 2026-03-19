# Notebooks

Interactive Jupyter notebooks covering all modules. Run locally
(`pip install -e ".[dev]"`) or on Google Colab.

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| [01_kuramoto_xy_dynamics](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/01_kuramoto_xy_dynamics.ipynb) | XY Hamiltonian from 4-oscillator K_nm, Trotter evolution | R(t) trajectory, quantum-classical overlay |
| [02_vqe_ground_state](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/02_vqe_ground_state.ipynb) | VQE with three ansatz variants | Energy convergence, ansatz comparison |
| [03_error_mitigation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/03_error_mitigation.ipynb) | ZNE on noisy Heron r2 simulator | Extrapolation plot, mitigated vs raw |
| [04_upde_16_layer](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/04_upde_16_layer.ipynb) | Full 16-layer SCPN UPDE as 16-qubit spin chain | Per-layer R bar chart, time evolution |
| [05_crypto_and_entanglement](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/05_crypto_and_entanglement.ipynb) | CHSH Bell test, correlator matrix, QKD QBER | S-parameter, key rate analysis |
| [06_pec_error_cancellation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/06_pec_error_cancellation.ipynb) | PEC quasi-probability decomposition, Monte Carlo | PEC vs ZNE, overhead scaling |
| [07_quantum_advantage_scaling](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/07_quantum_advantage_scaling.ipynb) | Classical vs quantum timing | Scaling plot, crossover prediction |
| [08_identity_continuity](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/08_identity_continuity.ipynb) | VQE attractor, coherence budget, entanglement witness, fingerprint, orchestrator mapping | Fidelity curves, S-parameters, phase roundtrip |
| [09_iter_disruption](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/09_iter_disruption.ipynb) | ITER 11-feature classifier, synthetic data | Feature distributions, classifier accuracy |
| [10_qsnn_training](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/10_qsnn_training.ipynb) | Parameter-shift gradient training | Loss curve, weight evolution |
| [11_surface_code_budget](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/11_surface_code_budget.ipynb) | Surface-code qubit budget, feasibility analysis | Rep vs surface code comparison, hardware feasibility table |
| [12_trapped_ion_comparison](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/12_trapped_ion_comparison.ipynb) | Superconducting vs trapped-ion noise models | Transpilation comparison, noisy Z-expectations |
| [13_cross_repo_bridges](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/13_cross_repo_bridges.ipynb) | SNN adapter, SSGF roundtrip, orchestrator mapping, fusion-core adapter | Phase roundtrip plot, warning report |

All notebooks run on the local AerSimulator. No IBM credentials needed.

## Running locally

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```
