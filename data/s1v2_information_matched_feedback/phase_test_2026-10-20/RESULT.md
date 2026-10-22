# S1v2 follow-up: decision-window phase test — result

Run 2026-10-20 14:20–14:26Z after `PREREGISTRATION.md` (same folder). Jobs: ibm_fez
dbbnh07i0auc73940bpg; ibm_marrakesh (job id in its record file); IQM Garnet 20 single-circuit
jobs (all completed; ids in `phase_test_garnet_20261020T142020Z.json`). Ideal Aer null check:
all slopes within ±0.13°/round, 95 % intervals within ±0.4°.

Slopes of the idle system-qubit phase, degrees per round (95 % bootstrap interval), register
positions s0, s1, s2 on each backend's frozen S1v2 main layout:

| Backend | meas | reset (measure + reset of 6 ancillas) | if (measure + never-firing if_test) |
|---|---|---|---|
| ibm_fez | +0.25 [−0.95, 1.42]; +0.55 [−0.61, 1.75]; +0.11 [−1.04, 1.32] | +0.49 [−0.75, 1.69]; +0.01 [−1.20, 1.21]; +0.36 [−0.87, 1.55] | **+27.30** [22.89, 31.64]; −2.04 [−3.23, −0.76]; **−13.41** [−14.51, −12.32] |
| ibm_marrakesh | +0.86 [−0.27, 2.02]; +0.54 [−0.62, 1.75]; −0.23 [−1.46, 1.00] | +0.49 [−0.69, 1.66]; −0.96 [−2.16, 0.30]; −0.23 [−1.37, 0.98] | **−54.47** [−55.67, −53.22]; +5.72 [−0.17, 11.45]; **−5.02** [−6.22, −3.75] |
| IQM Garnet | +1.31 [0.18, 2.42]; +1.31 [0.09, 2.44]; +1.40 [0.20, 2.60] | **+4.39** [3.37, 5.47]; **+5.99** [4.98, 6.94]; **+7.09** [6.10, 8.04] | **−10.37** [−12.98, −7.62]; +2.77 [0.16, 5.28]; **+12.44** [10.85, 14.09] |

## Predictions
- **P1 — PASS.** Garnet `reset`: s1 +5.99 and s2 +7.09 °/round, intervals exclude 0, |s| ≥ 5.
  Garnet `if`: s0 −10.37 and s2 +12.44.
- **P2 — PASS.** `meas` slopes below 5°/round on all backends (Garnet's are +1.3–1.4 with intervals
  just excluding 0; IBM's intervals contain 0).
- **P3 — PASS.** IBM `reset` slopes are smaller in magnitude than Garnet's at all three positions on
  both IBM processors (|s| ≤ 0.96 vs 4.39–7.09).

## Reading
- A classical decision interval (`if_test`) rotates idle-qubit phases on both providers, with
  qubit-specific rates of up to 54°/round on IBM and 12°/round on Garnet.
- A reset rotates idle-qubit phases on Garnet (where a reset is a measurement plus a classically
  controlled X) and not on IBM (native reset). Measurement alone does not on either.
- This explains the direction of the provider difference in the S1v2 NONE arm (resets only), but
  not its size: two reset rounds give ≈ 9–14° here, while the Garnet NONE-arm qubit phases differed
  from the ideal by −23°, +51° and +97°. The reset-window rotation is a contributing mechanism, not a
  sufficient one; the remainder (two-qubit-gate phases, crosstalk, calibration drift between runs)
  is untested.
- The test does not change any preregistered S1v2 result. The Garnet polarity check stays failed.
