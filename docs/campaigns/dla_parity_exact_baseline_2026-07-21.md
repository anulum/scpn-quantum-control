# Exact statevector baseline for the kingston DLA-parity data — 2026-07-21

**Status:** completed. Addresses audit findings B-9/B-10 (exact classical reference
+ mitigation-ablation context for the promoted DLA-parity data).
**Code:** `src/scpn_quantum_control/analysis/dla_parity_exact_baseline.py`,
runner `scripts/run_dla_parity_exact_baseline.py`, tests
`tests/test_dla_parity_exact_baseline.py` (40).
**Artifact:** `data/dla_parity_exact_baseline/dla_parity_exact_baseline.json`.
**Source data:** `data/phase2_dla_parity/` (promoted Phase-2 reduced A+G, `ibm_kingston`, n=4).

## What the DLA-parity campaign measures

The promoted runs execute the Kuramoto-XY Trotter circuit

    H_XY = Σ K_nm (X_n X_m + Y_n Y_m) + Σ ω_n Z_n

(`K[i,j] = 0.45·exp(-0.3·|i−j|)`, `ω = linspace(0.8, 1.2, n)`, `t_step = 0.3`)
and report **parity leakage** — the fraction of shots that end in the opposite
excitation-number parity to the prepared state — for an even (`0011`) and an odd
(`0001`) initial state across ten depths (2…50).

## The exact reference

Every gate in the circuit conserves total excitation number: `rz` trivially, and
the paired `rxx(θ)·ryy(θ)` on each nearest-neighbour edge because `XX` and `YY`
commute on a pair and `XX+YY` is the number-conserving hopping generator. So the
ideal, noiseless final state never leaves the prepared parity sector and its
parity leakage is **exactly zero**.

The module computes this directly by statevector (feasible for `n ≤ ~14`). The
reconstruction is verified bit-for-bit against the campaign builder
(`scripts/phase1_mini_bench_ibm_kingston.build_xy_trotter_circuit`): statevector
fidelity = 1.0. The exact leakage is `< 1e-9` at every depth, for even and odd
sectors, and across widths n = 2…8.

## Exact vs hardware

| Depth | exact leakage | hardware leakage (even / odd) |
|---|---|---|
| 2  | 0 | 0.084 / 0.082 |
| 6  | 0 | 0.153 / 0.147 |
| 20 | 0 | ~0.30 |
| 50 | 0 | 0.422 / 0.422 |

Hardware leakage grows monotonically with circuit depth to **42 %** at depth 50,
while the ideal reference is pinned at 0. **All observed leakage is device noise**
(it accumulates with gate count), and the even/odd DLA-parity asymmetry
(`asymmetry_relative`, ~0.01–0.09 and vanishing at large depth) sits inside that
noise floor rather than reflecting coherent parity-selective dynamics.

## Mitigation-ablation context (B-10)

The promoted packs ran `resilience_level = 0` (no ZNE/DD) and the Phase-2 readout
mitigation self-reports `full_confusion_matrix_available = false` (an
approximation, not a full inversion). This exact baseline supplies the ideal
`leakage = 0` reference that a ZNE/DD ablation compares against; running the
ablation on real hardware is a separate owner-gated QPU submission (AUD-5/7),
now unblocked with a rigorous classical reference in place.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
