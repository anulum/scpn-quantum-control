<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S1v2 information-matched feedback preregistration -->

# S1v2: information-matched dynamic-circuit feedback on IBM Heron and IQM Garnet — preregistration

Date: 2026-10-19

Status: frozen before any hardware data of this campaign. This document authorises
no submission. Every paid stage needs the owner's GO for that submission. This
preregistration must be committed **and pushed** before the first submission.

## Background and question

The published S1 paper (Zenodo DOI 10.5281/zenodo.20382099, `paper/submissions/submission_006_s1_feedback_control/`)
found that its monitored-feedback controller was not backend-general. It asked the
next controller to use direct Pauli observables, a matched open-loop arm,
polarity controls and a same-layout repeat. The template preregistered as RC-1
(`docs/campaigns/onqpu_dynamic_feedback_prereg_2026-07-16.md`) cannot test this. In
ideal simulation its feedback-on and feedback-off distributions differ by a
total-variation distance of at most about 0.003, while the noise floor at 4096
shots is about 0.03. RC-1 is therefore superseded by this design and is not run.

**Question.** Does measurement-conditioned feedback lock the phases of three detuned
XY-coupled qubits more strongly than the same corrections applied without the
measurement information, on IBM Heron and on IQM Garnet?

## Frozen design

All circuits come from `scripts/s1v2_protocol.py`.

| Item | Value |
|---|---|
| Logical digest, main matrix | `bf06f7e7c7e39af73540fd46b7efcf1658ba70d44039d4dc982e51bc72decf92` |
| Logical digest, latency pilot | `11cbc277cea29b704cd4b92f7b04317c1af59b42de6b1d775ee10f4ff5ca8998` |

Both digests are SHA-256 hashes of OpenQASM 3.

- **System.** Three system qubits start on the equator (`RY(pi/2)`). They have
  `omega = linspace(-0.4, 0.4, 3)` and evolve under
  `H = -K sum(X_i X_j + Y_i Y_j) - sum omega_i Z_i`, with one Lie–Trotter step of
  `dt = 1` per round.
  - **c0 (primary):** `K = 0.15`, 2 rounds.
  - **c2 (uncoupled reference):** `K = 0`, 3 rounds.
- **Monitor (per round, per system qubit).** The monitor is prepared with `RY(pi/2)`.
  A `CX–RY(pi/3)–CX` interaction in the system qubit's Y basis gives a weak `Y`
  readout. All monitors are then measured together.
- **Correction.**
  - **Applied gates:** an unconditional `RZ(-sign*0.5)` on each system qubit,
    then `RZ(+2*sign*0.5)` when that qubit's control bit is 1. Each `RZ` is built
    from `rx`/`ry` only, because IQM conditions only `x, y, rx, ry, r`; `if_test`
    blocks are not nested.
  - **Reset:** every monitor and coin is reset after each round.
- **Arms of c0:**
  - `CL+` (closed loop, polarity +1);
  - `CL−` (polarity −1);
  - `OL+` (information-matched open loop);
  - `NONE` (monitors measured, no correction).
- **Arms of c2:** `CL+`, `OL+`, `NONE`.
- **Open-loop control.** The corrections are the same, but each is conditioned on an
  independent coin qubit. The coin's per-slot `P(1)` equals the ideal closed-loop
  monitor rate. These rates are frozen in `COIN_PROBABILITIES`; they range from 0.19
  to 0.81 because the initial phase drift is predictable. The open loop therefore
  receives every correction the predictable drift implies, and differs from the
  closed loop only in the measurement information.
- **Readout.** Every arm is run once in the X basis and once in the Y basis. The
  matrix also has two system-readout calibration circuits (`000`, `111`).
  - Main matrix: 16 circuits at 4096 shots each, 65,536 shots in total.
- **Latency pilot.** One system qubit on the equator and one monitor in `|0>`. The
  pilot applies `k ∈ {0, 1, 2, 4, 8}` events of monitor measurement, with
  (`conditional`) or without (`measure_only`) an `if_test` that never fires. It is
  read out in X and Y.
  - Pilot: 20 circuits at 2048 shots each, 40,960 shots in total.

## Order of stages per backend

Backends:

- `ibm_fez` (primary IBM);
- `ibm_marrakesh` (IBM replication);
- IQM `garnet`.

The steps, in order:

1. `readiness`: live target check, transpilation and budget estimate, with no job.
   The readiness of 2026-10-19 passed on all three backends: every required dynamic
   operation is present and the main matrix needs at most 45 two-qubit gates.
2. `submit --stage pilot`, then `retrieve` and `analyse`.
3. **Pilot rule.** Let `rho = C_conditional(4) / C_conditional(0)`, where
   `C = |<X> + i<Y>|` of the pilot qubit. The 4 events equal the decision events of
   one c0 circuit (2 rounds × monitor decision + reset). The main stage on that
   backend runs only if `rho ≥ 0.15`. Otherwise the campaign on that backend stops
   and the pilot is reported as the result.
4. `submit --stage main --pilot-analysis <that backend's pilot analysis>`, then
   `retrieve` and `analyse`.

The scripts are `scripts/submit_s1v2_ibm.py` and `scripts/submit_s1v2_iqm.py`
(IQM runs in `.venv-iqm`). A journal is written before each provider call. A stage
with a journal is never submitted again. An ambiguous provider call is resolved by
a person, never by a retry.

## Endpoints and analysis

The analysis is in `analyse_main` and `analyse_pilot` of `scripts/s1v2_protocol.py`
and is applied per backend.

- **Observables.** From the system readout, `<X_i>` and `<Y_i>`. The order parameter
  is `R = |mean(<X_i> + i<Y_i>)|` and the reference alignment is `A = mean(<X_i>)`.
  Uncertainty comes from a multinomial bootstrap of both basis circuits (4000 draws,
  seed 20261019).
- **Primary endpoint.** `ΔR = R(c0, CL+) − R(c0, OL+)`. It passes on a backend if the
  one-sided 99 % lower bootstrap bound is above 0 **and** `ΔR ≥ 0.05`. This uses
  unmitigated readout.
- **Polarity check.** It passes if the 99 % upper bound of `A(c0, CL+)` is below 0
  **and** the 99 % lower bound of `A(c0, CL−)` is above 0. The lock point must
  follow the policy polarity.
- **Secondary, reported in every outcome:**
  - `R(c0, CL+) − R(c0, NONE)`;
  - `R(c2, CL+) − R(c2, OL+)`;
  - the primary contrast with per-qubit symmetric readout correction from the
    calibration circuits;
  - measured monitor rates of the closed-loop arms next to the frozen coin rates.
- **Cross-provider statement.** Feedback benefit is called *replicated across
  providers* only if the primary endpoint and the polarity check both pass on IQM
  Garnet and on at least one IBM backend.

## Predictions from the simulation gate

The gate was run before this document was frozen, using `scripts/s1v2_simulation_gate.py`.

- **Inputs:** vendor reference noise models (`FakeFez`, `FakeMarrakesh`,
  `IQMFakeGarnet`), 4096 shots, and idle decision time modelled as T1/T2 relaxation.
- **Results:** in `data/s1v2_information_matched_feedback/simulation_gate_*_2026-10-19.json`.

Predicted `ΔR(c0)` with its one-sided 99 % lower bound, at modelled decision time
per event:

| Model | 0 | 0.5 µs | 1 µs | 2 µs | 5 µs |
|---|---|---|---|---|---|
| ideal | 0.48 | | | | |
| FakeFez | 0.47 (0.45) | 0.47 | 0.47 | 0.46 | 0.43 (0.40) |
| FakeMarrakesh | 0.47 (0.44) | 0.47 | 0.46 | 0.46 | 0.44 (0.41) |
| IQMFakeGarnet | 0.38 (0.36) | 0.36 | 0.32 | 0.25 | 0.11 (0.08) |

- The primary endpoint and the polarity check pass in every row.
- Ideal lock points: `A(c0, CL+) = −0.58` and `A(c0, CL−) = +0.53`.
- Ideal `R`: CL+ 0.58, OL+ 0.10, NONE 0.04.
- The Garnet model loses the effect with decision time because its median T2 is
  about 9.6 µs. IQM does not publish its feedback latency; the pilot measures it.

The noise models are reference profiles, not the calibration of the day.

## Disclosure of design selection

The parameters were chosen on 2026-10-19 by an ideal-simulation sweep before any
hardware run of this design:

- monitor angle `pi/3`;
- correction `0.5`;
- `K`, `delta` and the round numbers;
- polarity labels.

The sweep covered `K ∈ {0, 0.15}`, `delta ∈ {0.4, 0.8}`, rounds `∈ {2, 3}`,
monitor angle `∈ {pi/6, pi/3}`, correction `∈ {0.25, 0.5}` and polarity ±1.

- **Why the selection is not hidden:** both polarities are part of the matrix.
- **Candidate dropped:** c1 (K = 0.15, 3 rounds) was dropped for depth, at 69–72
  two-qubit gates against 42–45.
- **Coin control changed:** the coin control first used a flat `P(1) = 0.5`. It was
  replaced by the per-slot rates before freezing, because that is the stronger
  control.

## Budget and caps

- **IBM:** 973 s remained on 2026-10-19 in the period ending 2027-04-24. The estimate
  per backend is 66 s for the pilot and 53 s for the main stage, assuming 1.1 s per
  circuit × 3 for dynamic circuits. The cap is 300 s per backend stage; the script
  refuses above it.
- **IQM Garnet:** 298 credits remained on 2026-10-19, as the owner read from the
  dashboard. The **preliminary estimate** is 2–8 credits for pilot plus main, based
  on earlier Garnet runs of about 0.04–0.07 credits per circuit. The API does not
  report credit burn. The owner reads the dashboard after the pilot. The campaign
  stops if the pilot alone used more than 10 credits.

## Claim boundary

- **Supported after execution:** for each backend and calibration window, whether
  the measurement information improves phase locking over an information-matched
  open loop, and whether the lock point follows the policy polarity. The
  measured idle cost of a mid-circuit decision is also supported.
- **Not supported:**
  - latency numbers of the vendor controller beyond the pilot's coherence ratio;
  - any claim of a general controller, of synchronisation protection beyond the
    measured observables, or of quantum advantage;
  - pooling raw counts across providers.

A failed pilot or a failed primary endpoint is a publishable result of equal
standing.

## Review

This campaign is reviewed before any submission by the group's designated reviewer
(RIGOR-FOUNDRY). The result manuscript is planned as submission 008. Its draft is
private until it is published.
