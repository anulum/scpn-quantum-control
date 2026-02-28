# Notebooks

Interactive Jupyter notebooks covering the core workflows. Run locally
(`pip install -e ".[dev]"`) or on Google Colab.

| Notebook | What it covers |
|----------|---------------|
| [01_kuramoto_xy_dynamics](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/01_kuramoto_xy_dynamics.ipynb) | Kuramoto → XY mapping, Trotter evolution, 4-osc and 8-osc R(t), classical comparison |
| [02_vqe_ground_state](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/02_vqe_ground_state.ipynb) | VQE with Knm-informed ansatz, ansatz comparison, 4q and 8q scaling |
| [03_error_mitigation](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/03_error_mitigation.ipynb) | ZNE (unitary folding + Richardson), higher-order extrapolation, noisy simulator |
| [04_upde_16_layer](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/04_upde_16_layer.ipynb) | Full 16-layer SCPN on simulator, per-layer coherence analysis, time evolution |

## Running locally

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```

All notebooks run on the local AerSimulator. No IBM credentials needed.
