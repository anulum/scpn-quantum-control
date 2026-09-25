<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Metriq SCPN benchmark schema proposal -->

# Metriq SCPN Benchmark Schema Proposal

Date: 2026-05-07

## Status

Prepared for future upstream discussion. Not submitted upstream and not
accepted by Metriq-Gym.

This document defines the benchmark shape that would make an
`scpn-quantum-control` Metriq result scientifically valid. It is not a
Metriq result, not a score, and not a benchmark acceptance notice.

## Proposed Benchmark Name

```text
Kuramoto--XY parity leakage
```

## Scientific Purpose

Measure how accurately a backend preserves a simple, symmetry-defined
observable of a heterogeneous Kuramoto--XY circuit family. The ideal
nearest-neighbour XY circuit conserves excitation-number parity, so its
parity-survival reference is exactly `1` at every depth. The proposed
measurement is a symmetry-leakage diagnostic, not a test of full-state
fidelity or a nontrivial ideal parity-decay curve.

The benchmark targets a concrete NISQ question:

```text
Given the same Kuramoto--XY Hamiltonian, initial state, Trotter step, and
measurement basis, how closely does the observed parity-survival curve
match the exact small-system reference curve?
```

This makes the benchmark a hardware-fidelity and reproducibility test,
not a quantum-advantage claim.

## Required Input Fields

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `benchmark_name` | string | yes | Must be `Kuramoto-XY parity leakage`. |
| `n_qubits` | integer | yes | Number of oscillators/qubits. The locally exercised proposal is `4`; `6` and `8` require separate resource and simulator checks before inclusion. |
| `depths` | list[int] | yes | Trotter depths to evaluate. |
| `shots` | integer | yes | Shots per circuit. |
| `initial_states` | list[string] | yes | Computational-basis states in qubit-0-first convention. |
| `coupling_profile` | string | yes | Initial schema: `exp_decay_0p45_0p3`. |
| `t_step` | float | yes | Trotter step; default `0.3`. |
| `reference_mode` | string | yes | `exact_statevector` for `n <= 8`. |
| `transpilation_policy` | string | yes | Backend/default transpilation policy identifier. |

## Circuit Definition

For `n` qubits and Trotter depth `d`, prepare each listed initial state
and apply:

```text
H_XY = sum_i omega_i Z_i + sum_(i,i+1) K_(i,i+1) (X_i X_(i+1) + Y_i Y_(i+1))
```

with:

```text
omega_i = linspace(0.8, 1.2, n)
K_(i,i+1) = 0.45 * exp(-0.3)
t_step = 0.3
```

The initial schema should use nearest-neighbour Trotter layers first,
because that keeps the benchmark practical on sparse gate hardware and
matches the current SCPN hardware artefact family.

## Primary Observable

Parity survival:

```text
P_same_parity =
    sum_{bitstrings with popcount parity equal to initial popcount parity}
        observed_probability(bitstring)
```

## Primary Score

The proposed score is one minus the clipped mean absolute parity error:

```text
score = 1 - clip(mean_depth_state(|P_observed - P_exact|) / tolerance, 0, 1)
```

Provisional score tolerance, subject to maintainer review:

```text
tolerance = 0.05
```

Interpretation:

- `1.0` means the measured parity-survival curve matches the exact
  reference within negligible error;
- `0.0` means the mean parity-survival error is at least the tolerance;
- intermediate values report bounded hardware fidelity for this circuit
  family.

## Required Secondary Metrics

Every result should also report:

- exact-state retention error;
- transpiled depth per circuit;
- two-qubit gate count per circuit;
- readout-baseline parity-flip estimate if calibration circuits are run;
- backend name and simulator/hardware flag;
- timestamp and provider metadata available from the benchmark runner;
- raw counts or a reproducible pointer to raw counts;
- SHA-256 hash of the result payload.

## Recommended Default Configuration

```json
{
  "benchmark_name": "Kuramoto-XY parity leakage",
  "n_qubits": 4,
  "depths": [2, 4, 6, 8, 10],
  "shots": 2048,
  "initial_states": ["0011", "0001"],
  "coupling_profile": "exp_decay_0p45_0p3",
  "t_step": 0.3,
  "reference_mode": "exact_statevector",
  "transpilation_policy": "provider_default"
}
```

## Acceptance Gates for an Upstream Schema

Before this should be submitted as an upstream Metriq-Gym benchmark, the
following local gates should pass:

- deterministic reference values for `n=4` are generated from committed
  code;
- local simulator execution produces a score near `1.0` within shot
  noise;
- the JSON result schema can represent raw counts, score, resources, and
  reference metadata;
- the benchmark can be run without SCPN-private artefacts or credentials;
- documentation clearly states that this is a parity-leakage fidelity
  benchmark, not a quantum-advantage benchmark.

## Non-Claim Boundary

This proposed schema must not be used to claim:

- broad quantum advantage;
- DLA-parity-only causality;
- universal decoherence protection;
- clinical, biological, or consciousness-related conclusions;
- hardware superiority outside the measured circuit family.

## Upstream Draft Text

```text
I would like to propose a Metriq-Gym benchmark for heterogeneous
Kuramoto--XY parity leakage. The first workload prepares fixed n=4
nearest-neighbour Kuramoto--XY Trotter circuits, measures computational-basis
counts, and reports the fraction of shots leaving the prepared parity sector.
The ideal parity-survival reference is exactly one at every depth. The goal
is a bounded symmetry-leakage diagnostic for this circuit family, not a
full-state fidelity or quantum-advantage claim.

The related scpn-quantum-control campaign has committed n=4 exact-reference
code and public hardware raw-count artefacts. A standalone Metriq-Gym
implementation and accepted schema do not yet exist. Such an implementation
should report raw counts, exact references, depth, two-qubit gates, backend
metadata, and SHA-256 payload hashes.
```
