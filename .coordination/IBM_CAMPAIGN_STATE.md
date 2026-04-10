# IBM Quantum Campaign — Persistent State

**Last updated:** 2026-04-10
**Owner:** Miroslav Šotek (protoscience@anulum.li)
**Purpose:** Survive context compaction. Read first in any session touching IBM.

---

## Background

- **2026-03-29:** Applied for IBM Quantum Credits (5h ibm_fez, 3 experiments).
- **~2026-04-09:** Credits application REJECTED (no academic affiliation).
- **2026-04-09:** Dr Berk Kovos (IBM Quantum Solutions Strategy Lead)
  personally emailed offering to discuss alternatives.
- **2026-04-10 ~15:30 CEST:** 30-minute video call with Berk.
- **Outcome:** No credits grant. Instead, Berk explained the existing
  Open Plan **180-minute promotion path** — actionable and scalable.

---

## The Path Forward (Berk's Guidance)

### Step 1 — Accumulate 20 min on current account
Spend 20 minutes of Open Plan runtime within any rolling 12-month window.
That triggers eligibility for a one-time 180-minute promo bonus.

### Step 2 — Opt in to the 180-min promo
Once eligible, activate the promo. You get 180 minutes to use however you
like (e.g. 180 minutes in one day, or spread across 12 months).

### Step 3 — After promo period
Account returns to standard 10 min / 28-day cycle. You DO NOT lose the
account; only the bonus allowance expires.

### Step 4 — Scale via multiple accounts
Each additional real person can register their own Open Plan account and
repeat the cycle. No hard cap; the constraint is "must be different
persons" (different emails + presumably identity).

---

## Current Account Status (scpn-quantum-control instance)

**Snapshot 2026-04-10 (from IBM dashboard):**

```
Instance cycle usage:  6m 45s  spent
Time remaining:        3m 15s
Time allocated:        10m 0s
Current cycle (UTC):   Mar 13, 2026 – Apr 10, 2026  ← ENDS TODAY
```

**12-month cumulative:** 6m 45s (of 20 min required for promo)
**Deficit to trigger promo:** 13m 15s

### Plan to trigger promo

| Cycle | Dates | Burn target | Cumulative |
|---|---|---|---|
| Mar 13 – Apr 10 | (today) | 3m 15s (all of it) | 10 min |
| Apr 10 – May 7 (est.) | next | 10 min | 20 min ← TRIGGER |
| — | | | opt in to 180-min promo |

Conservative estimate: **20 min cumulative reached in ~4 weeks**.

---

## Authoritative References

- [IBM Open Plan Updates Blog](https://www.ibm.com/quantum/blog/open-plan-updates)
- [Quantum Computing Report — IBM Open Plan Promo](https://quantumcomputingreport.com/ibm-offers-special-promotion-to-open-plan-users/)
- [IBM Plans Overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview)

**Verified facts:**
- Open Plan: 10 min per rolling 28-day window (base)
- Promo eligibility: 20 min used within any 12-month window
- Promo reward: 180 min over next 12 months (one-time, opt-in)
- Region: us-east only
- Hardware: ibm_fez, ibm_marrakesh, ibm_kingston, ibm_torino (156q Heron r2)
- Post-promo: return to 10 min / 28-day standard (no penalty)

---

## Experiment Priority Queue (for the 180-minute window)

**Total budget after promo:** 180 min ≈ 10,800 seconds of QPU time.

### Experiment 1 — DLA Parity Asymmetry (HIGHEST PRIORITY)
**Budget:** ~90 min
**Goal:** Hardware confirmation that odd (feedback) DLA sector is 4-10%
more robust to decoherence than even (projection) sector.
**Protocol:** Equal-depth circuits in even/odd magnetisation sectors,
n_qubits ∈ {4, 6, 8, 10, 12}, 10 reps × 8192 shots per circuit.
**Stretch:** Extend to n=14, n=16.
**Mitigation stack:** GUESS (new, arXiv:2603.13060) + DynQ placement.

### Experiment 2 — FIM Scaling Law (SECOND PRIORITY)
**Budget:** ~60 min
**Goal:** Extend dual protection result (F_FIM=0.916 > F_XY=0.849 at n=4)
to n=8, n=12, n=16. Measure λ_c(N).
**Protocol:** λ ∈ {0, 1, 3, 5, 8} × N ∈ {4, 8, 12}.

### Experiment 3 — M-sector Decoherence Profile (THIRD PRIORITY)
**Budget:** ~30 min
**Goal:** Depth sweep 50–400 CZ gates in each magnetisation sector to
characterise the noise profile for GUESS model calibration.
**Protocol:** Depth ∈ {50, 100, 200, 300, 400}, single coupling strength.

---

## Immediate Action (today, 2026-04-10)

**Target:** Burn remaining 3m 15s on current cycle before UTC midnight.

**Run:** Pipe cleaner end-to-end test on ibm_kingston (new hardware).
**Purpose:** Verify transpilation + pulse shaping + GUESS pipeline work
on Heron r2 156-qubit before Phase 2 real experiments.

**Script:** `scripts/pipe_cleaner_ibm_kingston.py`
**Results:** `.coordination/ibm_runs/pipe_cleaner_2026-04-10.json`
**Log:** `.coordination/IBM_EXECUTION_LOG.md`

---

## Credentials Location

- **Vault:** `/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md`
- **Channel:** `ibm_cloud` (ibm_quantum channel deprecated 2026-07-01)
- **Instance CRN:** stored in vault
- **API Key:** stored in vault

Never commit credentials. Always read from vault at runtime.

---

## DO NOT

- DO NOT spend more than 3m 15s today (you lose it at cycle reset anyway).
- DO NOT run anything on paid tier without explicit CEO approval.
- DO NOT create second account yet (wait until Phase 2 results validate
  the approach).
- DO NOT push IBM results publicly before internal review.
- DO NOT forget to log every run to IBM_EXECUTION_LOG.md.
