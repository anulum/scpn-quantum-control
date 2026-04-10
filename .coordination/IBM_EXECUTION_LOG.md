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

## 2026-04-10T184909Z — PHASE 1.5 REINFORCEMENT

- **Experiment:** phase1_5_reinforce
- **Backend:** ibm_kingston
- **Circuits:** 72 (D: 64, E: 8)
- **Job ID:** `d7ckcrh5a5qc73dosbmg`
- **Wall time:** 56.7s
- **Results file:** `.coordination/ibm_runs/phase1_5_reinforce_2026-04-10T184909Z.json`
- **Joint DLA parity (Phase 1 + 1.5):**
  - depth   2: leak_even=0.0773, leak_odd=0.0856, asym_rel=-0.0970 (n=6 reps)
  - depth   4: leak_even=0.0995, leak_odd=0.0866, asym_rel=+0.1494 (n=6 reps)
  - depth   6: leak_even=0.1173, leak_odd=0.1136, asym_rel=+0.0322 (n=6 reps)
  - depth   8: leak_even=0.1453, leak_odd=0.1284, asym_rel=+0.1312 (n=6 reps)
  - depth  10: leak_even=0.1679, leak_odd=0.1463, asym_rel=+0.1474 (n=6 reps)
  - depth  14: leak_even=0.1863, leak_odd=0.1803, asym_rel=+0.0329 (n=6 reps)
  - depth  20: leak_even=0.2314, leak_odd=0.2174, asym_rel=+0.0644 (n=6 reps)
  - depth  30: leak_even=0.2839, leak_odd=0.2652, asym_rel=+0.0703 (n=6 reps)
- **Outcome:** Phase 1 data reinforced with 4 extra reps; n=8 scaling data collected.

## 2026-04-10T185634Z — PHASE 2 CYCLE EXHAUST

- **Experiment:** phase2_exhaust_cycle
- **Backend:** ibm_kingston
- **Circuits:** 138 (F: 96, G: 18, H: 8, I: 16)
- **Job ID:** `d7ckft95a5qc73doseu0`
- **Wall time:** 97.5s
- **Results file:** `.coordination/ibm_runs/phase2_exhaust_2026-04-10T185634Z.json`
- **Joint DLA parity (all phases):**
  - depth   2: leak_even=0.0806, leak_odd=0.0827, asym_rel=-0.0251 (n=12 reps)
  - depth   4: leak_even=0.0963, leak_odd=0.0850, asym_rel=+0.1325 (n=12 reps)
  - depth   6: leak_even=0.1228, leak_odd=0.1116, asym_rel=+0.1007 (n=12 reps)
  - depth   8: leak_even=0.1410, leak_odd=0.1245, asym_rel=+0.1324 (n=12 reps)
  - depth  10: leak_even=0.1624, leak_odd=0.1478, asym_rel=+0.0986 (n=12 reps)
  - depth  14: leak_even=0.1857, leak_odd=0.1788, asym_rel=+0.0382 (n=12 reps)
  - depth  20: leak_even=0.2295, leak_odd=0.2114, asym_rel=+0.0855 (n=12 reps)
  - depth  30: leak_even=0.2771, leak_odd=0.2576, asym_rel=+0.0758 (n=12 reps)
- **Purpose:** Exhaust current Open Plan cycle to trigger 180-minute promo unlock.
- **Next action:** Check IBM dashboard. If usage ≥ 10m, opt in to 180-min promo.

## 2026-04-10T190136Z — PHASE 2.5 FINAL BURN

- **Experiment:** phase2_5_final_burn
- **Backend:** ibm_kingston
- **Circuits:** 90 (n=4 strongest depths, 9 new reps)
- **Job ID:** `d7ckide5nvhs73a4o780`
- **Wall time:** 65.1s
- **Results file:** `.coordination/ibm_runs/phase2_5_final_burn_2026-04-10T190136Z.json`
- **Final joint DLA parity (all phases):**
  - depth   2: leak_even=0.0806, leak_odd=0.0827, asym_rel=-0.0251 (n=12 reps)
  - depth   4: leak_even=0.0982, leak_odd=0.0862, asym_rel=+0.1398 (n=21 reps) [REINFORCED]
  - depth   6: leak_even=0.1291, leak_odd=0.1099, asym_rel=+0.1748 (n=21 reps) [REINFORCED]
  - depth   8: leak_even=0.1443, leak_odd=0.1284, asym_rel=+0.1241 (n=21 reps) [REINFORCED]
  - depth  10: leak_even=0.1658, leak_odd=0.1495, asym_rel=+0.1091 (n=21 reps) [REINFORCED]
  - depth  14: leak_even=0.1898, leak_odd=0.1797, asym_rel=+0.0558 (n=21 reps) [REINFORCED]
  - depth  20: leak_even=0.2295, leak_odd=0.2114, asym_rel=+0.0855 (n=12 reps)
  - depth  30: leak_even=0.2771, leak_odd=0.2576, asym_rel=+0.0758 (n=12 reps)
- **Purpose:** Complete cycle exhaust → 180-min promo unlock.
