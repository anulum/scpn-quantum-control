<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — ML convergence protocols and evidence -->
# QNN, QGNN, and QSNN convergence examples

`scpn_quantum_control.ml_examples` provides one deterministic, simulator-only
training example for each of the repository's QNN, QGNN, and QSNN model
families. The suite composes the existing trainers; it does not introduce a
second optimisation engine or a new numerical hot path.

## Frozen tasks and acceptance gates

Each task fixes its data, seed, step budget, target loss, minimum loss drop,
and optional task metric before execution. A certificate passes only when the
target, loss-drop, deterministic-replay, and metric gates all pass.

| Family | Existing training route | Frozen task | Step budget | Gates |
|---|---|---|---:|---|
| QNN | Multi-frequency parameter-shift phase-QNN classifier | Separate phase features `0` and `pi` into labels `0` and `1` | 80 | loss <= `1e-4`, drop >= `0.02`, accuracy = `1.0` |
| QGNN | Exact message passing followed by a Phase-QNode readout gradient | Fit four seeded three-node `K_nm` graphs to synthetic Kuramoto targets | 60 | loss <= `0.005`, drop >= `0.45` |
| QSNN, `exact_mse_v2` | Statevector `QuantumDenseLayer` with exact MSE parameter-shift descent | Silence one quantum synapse for a unit input | 1024 | loss <= `1e-5`, drop >= `0.7`, final spike = `0` |
| QSNN, `legacy_16_step` | The same current trainer, retained predecessor protocol | The original unit-input silencing task | 16 | The same original loss, drop and spike gates |

The default QSNN example has the explicit versioned identifier
`qsnn_single_synapse_silencing_v2`. Its budget was selected after a recorded
budget study and qualified as a new protocol. Correct MSE derivatives do not
reach the original loss target in sixteen steps. The suite retains that failed
predecessor certificate under `details.predecessor`, with its original
`qsnn_single_synapse_silencing` identifier, budget and acceptance gates.
A passing current protocol does not turn the predecessor into a passing run.
Measured losses, curves and evaluation costs live in the generated evidence;
the table above specifies the protocols rather than duplicating results.

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

For the retained QSNN diagnostic, call
`run_qsnn_convergence_example(protocol="legacy_16_step")`. Its failed target
remains visible in the returned certificate and the native framework row.
Acceptance and execution flags must be actual Boolean values; strings such as
`"false"` and integer substitutes are rejected by the evidence contracts.

Use `required_qnn_frameworks=("jax", "pytorch")` when an environment must
execute specific QNN framework adapters. An unknown framework is rejected. A
missing required dependency or a failing installed adapter makes the suite
fail closed; evidence files are not written.

## Framework matrix

Every framework row records an explicit outcome. `not_applicable` means the
bounded model family has no registered native adapter; `unsupported` means the
route lies outside this suite. Required unavailable adapters fail the gate.

| Family | SCPN native route | JAX | PyTorch | TensorFlow | Hardware |
|---|---|---|---|---|---|
| QNN | Actual convergence certificate, required | Execute when installed | Execute when installed | Execute when installed | Provider gradient unsupported |
| QGNN | Actual convergence certificate, required | Not applicable | Not applicable | Not applicable | Outside the suite |
| QSNN | Actual convergence certificate, required | Not applicable | Not applicable | Not applicable | Neuromorphic hardware unsupported |

Native rows report `ran` only when their actual training certificate passes;
a failed run produces `failed`. Standalone row factories execute the canonical
training task when no certificate is supplied. The unified suite passes its
already executed certificates to those factories. A native convergence row
has no invented gradient-error measurement. QNN framework rows separately
execute the bounded classifier gradient and record their measured agreement
with its parameter-shift reference. Installed adapter failures remain visible;
no arbitrary framework parity is implied.

## Evidence CLI

```bash
PYTHONPATH=src:oscillatools/src python scripts/run_ml_convergence_examples.py \
  --json-output data/ml_convergence_examples/convergence_evidence.json \
  --markdown-output data/ml_convergence_examples/convergence_evidence.md
```

The JSON payload uses schema `ml_convergence_examples.v1` and binds all task
specifications, loss histories, certificates, framework rows, notebook
pointers, and claim boundary with a canonical SHA-256 content digest. The
[human-readable evidence](https://github.com/anulum/scpn-quantum-control/blob/main/data/ml_convergence_examples/convergence_evidence.md)
and [machine-readable evidence](https://github.com/anulum/scpn-quantum-control/blob/main/data/ml_convergence_examples/convergence_evidence.json)
are committed together. The CLI performs no provider, QPU, or neuromorphic
hardware execution.

The evidence reader requires each task specification to match its registered
protocol, including the retained QSNN predecessor. Changing a seed, step budget,
target, minimum loss drop, task identifier or backend is rejected even if the
digest is recomputed. Contract checks and the digest establish consistency;
the producer run and its source and runtime provenance establish how the
evidence was obtained.
The reader also requires the complete registered framework matrix without
duplicate rows. Native outcomes must match their training certificates;
unregistered adapters and hardware routes cannot claim execution, and gradient
errors belong only to executed framework comparisons.

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
