# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Falsification Protocol

# Falsification Protocol

A scientific claim is only meaningful if there is an experiment
whose outcome would refute it. This page collects the falsification
criteria for every non-trivial claim `scpn-quantum-control`
currently makes, so a reader can locate the break point without
reverse-engineering the source.

Each claim has four fields:

- **Claim** — what we assert.
- **Domain of validity** — the regime where the claim is supposed
  to hold.
- **Falsifier** — the observable result that would refute the claim.
- **Current evidence** — the experiment or computation on which the
  claim currently rests.

## C1 — DLA dimension formula

- **Claim.** For the heterogeneous XY Hamiltonian
  $H = -\sum K_{ij}(X_i X_j + Y_i Y_j) - \sum (\omega_i / 2) Z_i$
  with generic (non-degenerate) frequencies on $N$ qubits, the
  dynamical Lie algebra has dimension
  $\dim(\mathrm{DLA}) = 2^{2N-1} - 2$
  and decomposes as
  $\mathrm{DLA} = \mathfrak{su}(2^{N-1}) \oplus \mathfrak{su}(2^{N-1})$
  acting on the even- and odd-parity subspaces.
- **Domain.** $N \ge 2$, all $\omega_i$ pairwise distinct, all
  $K_{ij} \neq 0$ for $i \neq j$.
- **Falsifier.** Computing the DLA by nested commutator closure at
  any $N \ge 2$ and getting a dimension different from
  $2^{2N-1} - 2$. Or finding a non-trivial symmetry beyond $\mathbb{Z}_2$
  parity (which would split the DLA further).
- **Evidence.** Verified computationally for $N = 2, 3, 4, 5$ in
  `analysis/dla_parity_theorem.py` and `tests/test_dla_parity_theorem.py`.
  Representation-theoretic argument for all $N$ (not yet formalised
  in Lean 4 — `docs/internal/audit_2026-04-17T0800_claude_gap_audit.md`
  §C Lean 4 entry).

## C2 — DLA parity asymmetry on hardware

- **Claim.** On a real superconducting processor, the
  even-magnetisation sector's post-Trotter leakage is larger than
  the odd-magnetisation sector's, by a few per cent, and the gap
  grows with Trotter depth.
- **Domain.** IBM Heron r2 class hardware at $n = 4$ qubits, Trotter
  depths 2–14, XY Hamiltonian with the same $K_{nm}$ matrix as the
  classical simulator.
- **Falsifier.** Any of:
  (i) mean relative asymmetry for depths $\ge 4$ drops to $\le 2\%$
  on a new hardware run on the same backend;
  (ii) the sign flips (odd > even);
  (iii) Welch's two-sample $t$-test returns $p > 0.05$ on 7 of 8
  depths.
- **Evidence.** `data/phase1_dla_parity/*.json` (348 circuits across
  4 sub-phases on `ibm_kingston`, April 2026). Mean asymmetry
  $+10.8\,\%$ for depth $\ge 4$, peak $+17.48\,\%$ at depth 6, Welch
  $p < 0.05$ on 7/8 depths, Fisher combined $\chi^2 = 123.4$
  ($p \ll 10^{-16}$). Reproducer:
  `tests/test_phase1_dla_parity_reproduces.py`.

## C3 — $K_{nm}$ topological mapping

- **Claim.** The SCPN coupling matrix $K_{nm}$ (exponential-decay,
  all-to-all, with anchor overrides from Paper 27) correlates
  strongly with the effective coupling topology of at least two
  physical systems (photosynthesis FMO, Josephson junction arrays,
  EEG alpha-band, ITER MHD modes, IEEE power grid).
- **Domain.** Systems with a natural distance-dependent coupling on
  a complete graph.
- **Falsifier.** Spearman $\rho < 0.5$ on every listed system.
- **Evidence.** Josephson array $\rho = 0.990$, EEG alpha
  $\rho = 0.916$, IEEE 5-bus $\rho = 0.881$, ITER MHD $\rho = 0.944$,
  FMO $\rho = 0.304$. See
  [`GAP_CLOSURE_STATUS.md`](https://github.com/anulum/scpn-quantum-control/blob/main/GAP_CLOSURE_STATUS.md).

## C4 — Rust acceleration factors

- **Claim.** Measured Python↔Rust speedups for the functions in
  [`pipeline_performance.md §21`](pipeline_performance.md#21-measured-rust-speedups-vs-python-baseline)
  stay within a factor of 2 of the published values on a
  comparable-class runner (Linux x86_64, ≥ 8 cores, ≥ 16 GB RAM).
- **Domain.** The exact five paired benchmarks listed in §21
  (`build_knm`, `kuramoto_euler`, `correlation_matrix_xy`,
  `lindblad_jump_ops_coo`, `lindblad_anti_hermitian_diag`).
- **Falsifier.** The next green CI run of
  `tests/test_rust_path_benchmarks.py` reports any paired speedup
  drop of more than 50 % from the published figure.
- **Evidence.** Section §21 of `pipeline_performance.md` (measured
  2026-04-17 on ML350 Gen8 via `test_rust_path_benchmarks.py`).

## Open questions (no claim yet)

The following items are **not** claims — they are open problems.
Nothing in `scpn-quantum-control` depends on any of them being
true. They appear here so a reader knows they are known.

- **Gap 2 — quantum result beyond classical.** There is no formal
  argument that a classical simulator cannot reproduce the DLA
  parity asymmetry. The asymmetry is observed on hardware and on a
  Lindblad noise model; classical simulation cost scales as
  $O(\mathrm{poly}(N))$ for $N \le 16$ and is not the bottleneck.
- **Gap 3 — `p_h1 = 0.72` first principles.** The hypothesis that
  $p_{h1}$ equals $A_{\mathrm{HP}} \sqrt{2 / \pi}$ (Hasenbusch-Pinn
  amplitude times the Nelson-Kosterlitz ratio) is 3 % off the
  observed value and was initially motivated by a square-lattice
  coincidence that is independently falsified. It is listed in
  `bkt_universals.py` as the best numerical fit among seven
  candidate combinations; it is not a derived claim.

When either of these is promoted to a claim, an entry goes in the
**Claims** section above with its own falsifier.
