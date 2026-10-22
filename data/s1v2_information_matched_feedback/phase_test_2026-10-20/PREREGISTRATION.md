# S1v2 follow-up: decision-window phase test — preregistration

Written 2026-10-20 before any job of this test was submitted. Owner GO in chat (2026-10-20,
"priamy test fázy na Garnete a IBM; - áno"). Seat SCPN-QUANTUM-COMPUTING-SYSTEMS/claude-90ad.

## Question
Do the classical-decision intervals of dynamic circuits (mid-circuit reset, `if_test`) rotate the
phase of idle qubits, and does this differ between IQM Garnet and IBM Heron? Motivation: the S1v2
pilots showed a near-linear idle-qubit phase rotation per never-firing `if_test` event (Garnet
≈ −25°/event, Marrakesh ≈ −4°/event, Fez larger and irregular) and a flat phase for measurement
alone; the Garnet main NONE arm (only measurements and resets, no feedback) showed per-qubit phases
far from the ideal simulation, and on IQM a reset is a measurement plus a classically controlled X.

## Circuits (logic frozen in `phase_test.py`, this folder)
Nine qubits in the S1v2 register order (system s0–s2, monitors m0–m2, coins c0–c2), placed on each
backend's frozen S1v2 main reference layout (from that backend's pilot journal). The system qubits
start on the equator with RY(π/2); monitors and coins stay in |0⟩. Each "round" acts on the six
ancillas only:

- `meas`: measure the six ancillas;
- `reset`: measure the six ancillas, then reset them;
- `if`: measure the six ancillas, then one `if_test` per system qubit on its monitor's bit with an
  RX(π) body that never fires (bit is 0).

Rounds n ∈ {1, 2, 4} per variant plus a shared n = 0 baseline; final X or Y readout of the system
qubits. 20 circuits, 1024 shots each.

## Observable and analysis
Per system qubit i and circuit: ⟨X_i⟩, ⟨Y_i⟩ from the readout bit; phase φ_i(n) = atan2(⟨Y_i⟩, ⟨X_i⟩),
unwrapped along n; slope s_i = least-squares slope of φ_i(n) over n ∈ {0,1,2,4} per variant, in
degrees per round; uncertainty from 2000 binomial bootstrap draws per circuit (seed 20261020), 95 %
percentile interval.

## Predictions (decision rules)
- P1 (decision windows rotate phase on Garnet): on Garnet, for `reset` and for `if`, at least one
  system qubit has a 95 % interval of s_i excluding 0 with |s_i| ≥ 5°/round.
- P2 (measurement alone does not): on every backend, `meas` slopes have |s_i| < 5°/round or an
  interval containing 0, for all three system qubits.
- P3 (provider difference for reset): on IBM, `reset` slopes are smaller in magnitude than Garnet's
  `reset` slope on the same register position for at least two of the three system qubits.

Outcomes are reported whichever way they fall. This test does not change the preregistered S1v2
endpoints or their results; it can only inform the interpretation of the Garnet polarity result.

## Execution
IBM: one SamplerV2 job per backend (ibm_fez, ibm_marrakesh), max_execution_time 120 s. IQM Garnet:
one job per circuit (the main-stage batch was rejected three times by IQM station control on
2026-10-19/20). Records: one JSON per backend in this folder.
