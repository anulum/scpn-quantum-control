# QNN, QGNN, and QSNN convergence examples (BL-42)

`scpn_quantum_control.ml_examples` provides one deterministic, simulator-only
training example for each of the repository's QNN, QGNN, and QSNN model
families. The suite composes the existing trainers; it does not introduce a
second optimisation engine or a new numerical hot path.

## Frozen tasks and acceptance gates

Each task fixes its data, seed, step budget, target loss, minimum loss drop,
and optional task metric before execution. A certificate passes only when the
target, loss-drop, deterministic-replay, and metric gates all pass.

| Family | Existing training route | Frozen task | Steps | Initial loss | Best loss | Gate |
|---|---|---|---:|---:|---:|---|
| QNN | Multi-frequency parameter-shift phase-QNN classifier | Separate phase features `0` and `pi` into labels `0` and `1` | 80 | 0.0229967050 | 0.0000710821 | loss <= `1e-4`, drop >= `0.02`, accuracy = `1.0` |
| QGNN | Exact message passing followed by a Phase-QNode readout gradient | Fit four seeded three-node `K_nm` graphs to synthetic Kuramoto targets | 60 | 0.508950006 | 0.003414575 | loss <= `0.005`, drop >= `0.45` |
| QSNN | Statevector `QuantumDenseLayer` with parameter-shift descent | Silence the firing probability of one quantum synapse for a unit input | 16 | 0.772880023 | 0.000000562 | loss <= `1e-5`, drop >= `0.7`, final spike = `0` |

These are small, synthetic convergence witnesses. They do not establish
generalisation, architecture-independent trainability, state-of-the-art
accuracy, production convergence, or quantum advantage.

## Python API

```python
from scpn_quantum_control.ml_examples import run_ml_convergence_suite

evidence = run_ml_convergence_suite()
assert evidence.passed

for certificate in evidence.certificates:
    print(
        certificate.spec.family.value,
        certificate.best_loss,
        certificate.passed,
    )
```

Use `required_qnn_frameworks=("jax", "pytorch")` when an environment must
execute specific QNN framework adapters. An unknown framework is rejected. A
missing required dependency or a failing installed adapter makes the suite
fail closed; evidence files are not written.

## Framework matrix

The committed 2026-07-28 local evidence records every matrix cell explicitly.
`not_applicable` means the bounded model family has no registered native
adapter; `unsupported` means the route lies outside this suite.

| Family | SCPN native route | JAX | PyTorch | TensorFlow | Hardware |
|---|---|---|---|---|---|
| QNN | ran, required | ran, agreement passed | ran, agreement passed | unavailable in the evidence environment | provider gradient unsupported |
| QGNN | ran, required | not applicable | not applicable | not applicable | outside the suite |
| QSNN | ran, required | not applicable | not applicable | not applicable | neuromorphic hardware unsupported |

The QNN JAX and PyTorch rows execute the same bounded classifier loss and agree
with its parameter-shift reference. The evidence records maximum absolute
gradient errors of about `1.11e-9` and `1.39e-17`, respectively. It does not
claim arbitrary framework parity.

## Evidence CLI

```bash
PYTHONPATH=src:oscillatools/src python scripts/run_ml_convergence_examples.py \
  --json-output data/ml_convergence_examples/bl42_convergence_evidence.json \
  --markdown-output data/ml_convergence_examples/bl42_convergence_evidence.md
```

The JSON payload uses schema `ml_convergence_examples.v1` and binds all task
specifications, loss histories, certificates, framework rows, notebook
pointers, and claim boundary with a canonical SHA-256 content digest. The
[human-readable evidence](https://github.com/anulum/scpn-quantum-control/blob/main/data/ml_convergence_examples/bl42_convergence_evidence.md)
and [machine-readable evidence](https://github.com/anulum/scpn-quantum-control/blob/main/data/ml_convergence_examples/bl42_convergence_evidence.json)
are committed together. The CLI performs no provider, QPU, or neuromorphic
hardware execution.

## Learning pointers

| Family | Next source |
|---|---|
| QNN | `scripts/run_ml_convergence_examples.py` and the public API above |
| QGNN | [Quantum Graph Neural Network](quantum_graph_neural_network.md) |
| QSNN | `notebooks/10_qsnn_training.ipynb` |

The QSNN example is a probability/synapse-angle convergence witness for the
existing dense quantum layer. It does not model temporal spike coding, LIF
membrane dynamics, STDP, event-driven execution, or neuromorphic hardware.

## Claim boundary

> deterministic synthetic local QNN/QGNN/QSNN training evidence on frozen
> small tasks; no arbitrary-architecture, generalisation, SOTA, provider, QPU,
> neuromorphic-hardware, or production convergence claim

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
