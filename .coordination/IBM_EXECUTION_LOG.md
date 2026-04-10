# IBM Quantum Execution Log

**Append-only log. NEVER overwrite. NEVER delete entries.**

Format: one entry per submission. Timestamp in ISO 8601 UTC.
Read `IBM_CAMPAIGN_STATE.md` first for context.

---

## Historical (before this log existed)

- **~Mar 2026:** Early Open Plan runs on ibm_fez for FIM dual protection
  (F_FIM=0.916 vs F_XY=0.849, p<10⁻¹²). Results in notebooks NB14-47.
- **2026-03-29:** IBM credits application submitted (rejected).
- **Used before 2026-04-10:** 6m 45s cumulative on current cycle.

---

## Run Log

<!-- New entries below. Each entry block MUST include:
  - timestamp_utc
  - experiment_name
  - backend
  - cycle_before / cycle_after (time_remaining)
  - circuits: list of (name, n_qubits, depth, shots)
  - job_ids
  - results_file
  - notes
-->

## 2026-04-10T182029Z — RETRIEVED

- **Experiment:** pipe_cleaner_ibm_kingston
- **Backend:** ibm_kingston
- **Job ID:** `d7cju9u5nvhs73a4ngn0`
- **Qubits:** 4, Shots per circuit: 1024
- **Circuits:** 2
- **even:** mean M = 3.7520, even frac = 0.9268
- **odd:** mean M = 1.8770, even frac = 0.0732
- **Results file:** `.coordination/ibm_runs/pipe_cleaner_retrieved_2026-04-10T182029Z.json`
- **Outcome:** Pipeline verified. Pipe cleaner submitted + parsed successfully on ibm_kingston.
