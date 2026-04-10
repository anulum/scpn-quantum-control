# Phase 1 Mini-Bench — Experiment Design

**Date:** 2026-04-10
**Budget:** ~190s of remaining QPU runtime (target ~120s, keep ~70s margin)
**Backend:** ibm_kingston (156-qubit Heron r2)

---

## Scientific Goal

Test the DLA parity asymmetry hypothesis on real Heron r2 hardware with
minimum viable circuit depth and repetitions.

**Hypothesis (from SCPN simulator findings):** Under H_XY time evolution,
the odd-parity sector (P_odd = -1, where P = ∏Z_i) is 4–10% more robust
to hardware decoherence than the even-parity sector (P_even = +1).

**Metric:** `parity_leakage = P(final state has opposite parity from initial)`
- Ideal: 0 (parity is conserved by H_XY because [H_XY, P] = 0)
- Noisy: >0, and we predict `leakage_even > leakage_odd`

## DLA Parity Primer

For the XY Hamiltonian on n qubits:
- **Parity operator:** P = Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1}
- **Eigenvalue +1 (even):** states with even popcount (even number of |1⟩s)
- **Eigenvalue -1 (odd):** states with odd popcount
- **Conservation:** [H_XY, P] = 0 (XX+YY flips pairs, preserves parity)
- **DLA decomposition:** DLA(H_XY) = su(even) ⊕ su(odd)

The hypothesis is that the even and odd blocks respond differently to
depolarising noise under hardware operation.

## Initial States (equal-dynamics protocol)

To compare sectors fairly, we use initial states with **equally
nontrivial dynamics** — both have multiple states to mix among in
their conserved block.

| Sector | Initial state | Parity | M value | States in M-block |
|--------|---------------|--------|---------|-------------------|
| Even | `\|0011⟩` (qubits 0,1 flipped) | +1 | 0 | 6 |
| Odd | `\|0001⟩` (qubit 0 flipped) | -1 | +2 | 4 |

Both initial states have nontrivial XY dynamics (pairs can flip). Neither
is a trivial eigenstate.

---

## Experiment A — DLA Parity at n=4

```yaml
n_qubits: 4
t_step: 0.3         # θ ≈ 11.5°/gate — meaningful evolution
depths: [2, 4, 6, 8, 10, 14, 20, 30]
sectors: [even, odd]
initial_states:
  even: "0011"      # popcount=2, parity=+1
  odd:  "0001"      # popcount=1, parity=-1
reps: 2             # basic error bars
shots_per_circuit: 2048
```

**Circuit count:** 8 depths × 2 sectors × 2 reps = **32 circuits**
**Estimated QPU cost:** 32 × ~2s = ~64 s

## Experiment B — Scaling check at n=6

```yaml
n_qubits: 6
t_step: 0.3
depths: [4, 8, 16]
sectors: [even, odd]
initial_states:
  even: "000011"    # popcount=2, parity=+1
  odd:  "000001"    # popcount=1, parity=-1
reps: 1
shots_per_circuit: 2048
```

**Circuit count:** 3 × 2 × 1 = **6 circuits**
**Estimated QPU cost:** 6 × ~4s = ~24 s

## Experiment C — Readout baseline at n=4

No Trotter evolution, just state preparation + measurement.
Characterises readout error separately from decoherence.

```yaml
n_qubits: 4
initial_states: ["0000", "1111", "0101", "1010"]
shots_per_circuit: 4096
```

**Circuit count:** 4
**Estimated QPU cost:** 4 × ~0.5s = ~2 s

---

## Totals

| Experiment | Circuits | Est. QPU | Purpose |
|-----------|----------|----------|---------|
| A (DLA parity n=4) | 32 | 64 s | Primary hypothesis test |
| B (Scaling n=6) | 6 | 24 s | Scale probe |
| C (Readout baseline) | 4 | 2 s | Noise calibration |
| **TOTAL** | **42** | **~90 s** | |

**Budget margin:** 193 s remaining − 90 s estimated = **~103 s safety margin**.

Submitted as a single SamplerV2 job with 42 pubs. One job ID, one
retrieval, batch-efficient.

---

## Analysis Protocol

For each pub result:
1. Extract counts from DataBin (use `_extract_counts`)
2. For each bitstring, compute popcount parity (even/odd)
3. Compute `parity_leakage = N_opposite_parity / total_shots`
4. Aggregate per (sector, depth) across reps
5. Compare `leakage_even(depth)` vs `leakage_odd(depth)`
6. Report asymmetry signal: `(leakage_even - leakage_odd) / leakage_odd`

**Expected outcome:** Depth-dependent positive asymmetry at n=4, crossing
zero at small depth (no signal), growing with depth (decoherence accumulates).

---

## Outputs

- JSON: `.coordination/ibm_runs/phase1_bench_<timestamp>.json`
  - Raw counts per circuit
  - Computed metrics per (experiment, sector, depth, rep)
  - Aggregated asymmetry curves
- Log entry in `IBM_EXECUTION_LOG.md`
- Update `IBM_CAMPAIGN_STATE.md` with cumulative usage

## Risks

- **Queue delay:** ibm_kingston may have wait time. Timeout set to 20 min.
- **Job failure:** If SamplerV2 errors, retrieve script can recover by
  job_id. Retry logic: record job_id immediately, don't lose it.
- **Over-budget:** If individual circuits take longer than estimated,
  we burn more than 90s. Still within 193s margin, but watch dashboard.
- **Cycle boundary:** Current cycle ends ~UTC midnight. Submit well
  before to avoid partial-cycle reset confusion.
