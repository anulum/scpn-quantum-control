# Roadmap

## v0.2.0 (March 2026)

- ZNE error mitigation on Kuramoto circuits (target: halve depth-85 error)
- VQE 8-qubit ground state on hardware (56 CZ gates, within coherence)
- Dynamical decoupling for UPDE-16 idle qubits
- Repeat noise baseline to track ibm_fez calibration drift

## v0.3.0 (Q2 2026)

- PEC (probabilistic error cancellation) integration
- QSNN training loop on hardware (parameter-shift STDP)
- Quantum disruption classifier trained on ITER disruption database
- Qiskit Primitives v2 migration (EstimatorV2 / SamplerV2)

## v0.4.0 (Q3 2026)

- Trapped-ion backend support (IonQ / Quantinuum)
- Quantum advantage benchmark: quantum Kuramoto vs classical ODE at N=20+
- Integration with SCPN-Fusion-Core transport solver
- Arxiv preprint: "Quantum simulation of Kuramoto phase dynamics on NISQ hardware"

## Future

- Fault-tolerant UPDE simulation (surface code logical qubits)
- Quantum RL agent for real-time plasma control
- Integration with SCPN SSGF geometry engine
