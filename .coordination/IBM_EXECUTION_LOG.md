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

## 2026-04-10T183728Z — PHASE 1 MINI-BENCH

- **Experiment:** phase1_dla_parity_mini_bench
- **Backend:** ibm_kingston
- **Circuits:** 42 (A: 32, B: 6, C: 4)
- **Job IDs:** `d7ck79m5nvhs73a4nr10`, `d7ck7hb0g7hs73dqvbg0`
- **Wall time:** 44.1s
- **Results file:** `.coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json`
- **DLA parity summary (exp A):**
  - depth   2: leak_even=0.0725, leak_odd=0.0828, asym_rel=-0.1239
  - depth   4: leak_even=0.0881, leak_odd=0.0811, asym_rel=+0.0873
  - depth   6: leak_even=0.0942, leak_odd=0.1008, asym_rel=-0.0654
  - depth   8: leak_even=0.1201, leak_odd=0.1223, asym_rel=-0.0180
  - depth  10: leak_even=0.1562, leak_odd=0.1299, asym_rel=+0.2030
  - depth  14: leak_even=0.1626, leak_odd=0.1707, asym_rel=-0.0472
  - depth  20: leak_even=0.2026, leak_odd=0.2131, asym_rel=-0.0493
  - depth  30: leak_even=0.2517, leak_odd=0.2559, asym_rel=-0.0162
- **Outcome:** Phase 1 primary DLA parity data on Heron r2.
