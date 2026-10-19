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

**Question.** On IBM Heron and on IQM Garnet, does a measurement-conditioned
correction policy lock the phases of three detuned XY-coupled qubits more strongly
than an open-loop policy? The open-loop policy has the same circuits, qubits,
measurements, resets and layout, and its corrections have the closed loop's
*ideal* per-slot frequencies.

**Estimand.** This is a comparison of two correction policies with equal resources.
It is not a proof that only the measurement information differs on noisy hardware:
the closed-loop correction frequencies on hardware may differ from the ideal ones.
The measured monitor rates of the closed-loop arms are reported next to the frozen
coin rates, so that difference stays visible.

## Frozen design

All circuits come from `scripts/s1v2_protocol.py`.

| Item | Value |
|---|---|
| Logical digest, main matrix | `4de9fd99aa2aec5c08f686dacb6167bbac2e1e66e6a39ef6ee40d663ff337fdc` |
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
- **Equal resources.** Every arm (CL, OL, NONE) has the same nine qubits: three
  system qubits, three monitors and three coins. In every arm the coins are
  prepared with the frozen per-slot probabilities, then measured and reset.
  Closed and open loop therefore have identical operation counts and differ only
  in the control register of the `if_test`: monitor bits or coin bits. NONE has
  no corrections; it is an unmatched secondary reference.
- **Common layout.** On each backend, one physical layout (s0..s2, m0..m2,
  c0..c2) is chosen once by transpiling the c0 OL circuit with seed 20261019 and
  is used for every main circuit. The pilot runs on the physical s0 and m0 of that
  layout. The layout is recorded in the pilot journal, and the main stage uses
  the recorded layout.
- **Arms of c0:**
  - `CL+` (closed loop, polarity +1);
  - `CL−` (polarity −1);
  - `OL+` (information-matched open loop);
  - `NONE` (monitors measured, no correction).
- **Arms of c2:** `CL+`, `OL+`, `NONE`.
- **Open-loop control.** The corrections are the same, but each is conditioned on an
  independent coin. The coin's per-slot `P(1)` equals the *ideal* closed-loop
  monitor rate (Aer statevector, 200,000 shots). These rates are frozen in
  `COIN_PROBABILITIES`; they range from 0.19 to 0.81 because the initial phase
  drift is predictable. The matching is ideal-only, as stated under Estimand.
- **Readout.** Every arm is run once in the X basis and once in the Y basis. The
  matrix also has two system-readout calibration circuits (`000`, `111`).
  - Main matrix: 16 circuits at 4096 shots each, 65,536 shots in total.
- **Coherence pilot.** One system qubit on the equator and one monitor in `|0>`. The
  pilot applies `k ∈ {0, 1, 2, 4, 8}` events of monitor measurement, with
  (`conditional`) or without (`measure_only`) an `if_test` that never fires. It is
  read out in X and Y. It is an operational screen of how much coherence survives
  mid-circuit measurement and decision events on the chosen qubits. It is **not** a
  measurement of the vendor latency, and not of the coherence of the main circuits,
  which also contain interactions and resets.
  - Pilot: 20 circuits at 2048 shots each, 40,960 shots in total.

## Order of stages per backend

Backends:

- `ibm_fez` (primary IBM);
- `ibm_marrakesh` (IBM replication);
- IQM `garnet`.

The steps, in order:

1. `readiness`: live target check, transpilation and budget estimate, with no job.
   The readiness of 2026-10-19 (after review 284, common layout) passed on all three
   backends: every required dynamic operation is present and the main matrix needs
   at most 45 two-qubit gates.
2. `submit --stage pilot`, then `retrieve` and `analyse`.
3. **Pilot screen.** Let `C = |<X> + i<Y>|` of the pilot qubit and
   `rho = C_conditional(4) / C_conditional(0)`. The 4 events match the number of
   decision events of one c0 circuit (2 rounds × monitor decision + reset). The main
   stage on that backend runs only if **both** conditions hold, with bounds from a
   multinomial bootstrap of the X and Y circuits (4000 draws):
   - the one-sided 99 % lower bound of `C_conditional(0)` is at least 0.5, so a
     coherent baseline is clearly present;
   - the one-sided 99 % lower bound of `rho` is at least 0.15.

   A noise-dominated baseline therefore cannot pass through a ratio of two small
   numbers. A fully depolarised Aer pilot is refused (test
   `test_depolarised_pilot_with_zero_coherence_is_refused`). If the screen fails,
   the campaign on that backend stops and the pilot is reported as the result.
4. `submit --stage main --pilot-analysis <that backend's pilot analysis>`, then
   `retrieve` and `analyse`.

The scripts are `scripts/submit_s1v2_ibm.py` and `scripts/submit_s1v2_iqm.py`
(IQM runs in `.venv-iqm`). Their safety mechanisms:

- **Exclusive journal.** Before each provider call, the stage is reserved by
  creating its journal with an exclusive atomic link and a directory fsync. Of
  any number of concurrent callers, exactly one obtains the stage. This is tested
  with 8 separate interpreters in 4 trials.
- **No resubmission.** A stage with a journal is never submitted again. An
  ambiguous provider call is resolved by a person, never by a retry.
- **Pilot chain check.** The main stage verifies the complete pilot chain:
  analysis → raw counts → journal. It checks experiment, provider, backend,
  stage, frozen pilot digest, hashes, labels and shots, and recomputes the pilot
  screen from the raw counts.
- **IBM job cap.** Each IBM job carries the provider `max_execution_time` equal to
  `--max-qpu-seconds` (default 300 s, a whole number in 1–10,800). The value is
  read back from the runtime and recorded in the journal before the call.

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
- **Statistical limits.**
  - The bounds are per-backend percentile-bootstrap bounds. Their exact 99 %
    coverage near `R = 0` is not demonstrated.
  - The cross-provider statement combines per-backend decisions (Garnet and at
    least one IBM backend). It is not a separately calibrated global 99 % test.
  - The readout-corrected secondary uses the calibration as a plug-in scale and
    does not propagate its uncertainty.
  - Raw counts are never pooled across providers.

## Predictions from the simulation gate

The gate was run before this document was frozen, using `scripts/s1v2_simulation_gate.py`.

- **Inputs:** vendor reference noise models (`FakeFez`, `FakeMarrakesh`,
  `IQMFakeGarnet`), 4096 shots, and idle decision time modelled as T1/T2 relaxation.
- **Results:** in `data/s1v2_information_matched_feedback/simulation_gate_*_2026-10-19.json`.

Predicted `ΔR(c0)` with its one-sided 99 % lower bound, at modelled decision time
per event:

| Model | 0 | 0.5 µs | 1 µs | 2 µs | 5 µs |
|---|---|---|---|---|---|
| ideal | 0.46 | | | | |
| FakeFez | 0.46 (0.44) | 0.48 | 0.47 | 0.46 | 0.44 (0.41) |
| FakeMarrakesh | 0.47 (0.44) | 0.48 | 0.47 | 0.46 | 0.44 (0.42) |
| IQMFakeGarnet | 0.39 (0.37) | 0.37 | 0.33 | 0.25 | 0.12 (0.09) |

All rows use the final circuits: coins in every arm, one common layout per backend.

- The primary endpoint and the polarity check pass in every row.
- Ideal lock points: `A(c0, CL+) = −0.56` and `A(c0, CL−) = +0.55`.
- Ideal `R`: CL+ 0.56, OL+ 0.10, NONE 0.04.
- The Garnet model loses the effect with decision time because its median T2 is
  about 9.6 µs. IQM does not publish its feedback latency; the pilot screens the
  retained coherence on the chosen qubits.

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
- **Changes after review 284 (RIGOR-FOUNDRY, 2026-10-19), all before any hardware
  data:**
  - coins added to every arm, giving equal resources;
  - one common layout per backend;
  - the pilot screen gained an absolute baseline floor and bootstrap bounds;
  - exclusive journal reservation and pilot-chain binding;
  - the IBM provider cap is forwarded to the runtime;
  - the estimand was narrowed.

## Budget and caps

- **IBM:** 973 s remained on 2026-10-19 in the period ending 2027-04-24. The estimate
  per backend is 66 s for the pilot and 53 s for the main stage, assuming 1.1 s per
  circuit × 3 for dynamic circuits. Each job carries a provider cap
  `max_execution_time = 300 s`. The script refuses a stage whose estimate exceeds
  the cap.
- **IQM Garnet:** 298 credits remained on 2026-10-19, as the owner read from the
  dashboard. The **preliminary estimate** is 2–8 credits for pilot plus main, based
  on earlier Garnet runs of about 0.04–0.07 credits per circuit. The API does not
  report credit burn, and **there is no automatic credit cap**. The owner reads the
  dashboard after the pilot before giving the main-stage GO. The campaign stops if
  the pilot alone used more than 10 credits.

## Claim boundary

- **Supported after execution:** for each backend and calibration window, whether
  the measurement-conditioned policy locks the phases more strongly than the
  resource-equal open-loop policy with ideal-matched correction frequencies, and
  whether the lock point follows the policy polarity. The pilot's coherence screen
  on the chosen qubits is also supported.
- **Not supported:**
  - latency numbers of the vendor controller;
  - the claim that only the measurement information differs between the arms on
    hardware;
  - the claim that the pilot's retained coherence equals that of the main circuits;
  - any claim of a general controller, of synchronisation protection beyond the
    measured observables, or of quantum advantage;
  - pooling raw counts across providers.

A failed pilot or a failed primary endpoint is a publishable result of equal
standing.

## Review

This campaign is reviewed before any submission by the group's designated reviewer
(RIGOR-FOUNDRY). The result manuscript is planned as submission 008. Its draft is
private until it is published.
