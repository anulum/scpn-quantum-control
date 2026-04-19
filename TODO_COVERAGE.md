<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Coverage gap — detailed per-file TODO

This file documents every source module that currently does not
reach 100 % line coverage, the specific line ranges that are
uncovered, and the classification that drives how each gap closes.
It is the work queue for the "Coverage push to 100 %" v1.0
milestone.

## Why this file exists at all

The only correct long-term prevention for a coverage gap is the
standing rule that has been in place for a long time:

> **NEW CODE = NEW TESTS (multifaceted) = NEW SUPERIOR DOCS.**

A module is not "done" until all three ship together. The 359-line
gap catalogued below exists because some modules were landed
without the tests side of that contract — usually because the
module depends on an optional backend that was not installed in
the local environment, or because the code contains defensive
branches (`if denom < 1e-30`) that were never test-triggered, or
because the Python fallback for a Rust-accelerated function was
never exercised while the Rust wheel was present.

Going forward: every new module ships with multi-angle tests that
exercise every branch, plus a page-or-section of user-facing docs.
This is not about the coverage metric — it is about the rule.

## Current baseline

Measured 2026-04-18 with `pytest --cov=scpn_quantum_control -m
"not slow" --ignore=tests/test_hardware_runner.py` after installing
`jax`, `dynamiqs` and `ripser`:

| Metric | Value |
| --- | --- |
| Statements | 13 076 |
| Uncovered | 315 (after `pulse_shaping` fallback tests) |
| Coverage | ~97.6 % |
| Failing tests under full-suite ordering | 10 (all pass in isolation) |

The line-count column below is **post-install** of the three
optional backends above. Earlier measurements that reported 520
uncovered lines were against an environment missing those extras.

## Classification taxonomy

Each row in the per-file table below carries one of the following
tags in the `Class` column:

* **Gap** — a genuine test gap. A real code path exists and must
  be exercised by a new test. Default expectation unless stated
  otherwise.
* **Fallback** — the Python fallback for a Rust-accelerated path.
  Tests should monkey-patch `_HAS_RUST = False` and verify parity
  or documented divergence. Example: `phase/pulse_shaping.py`
  (closed in commit `6c3e29e`).
* **Optional-backend** — the code only runs when an optional
  dependency is installed. If the extra is in `pyproject.toml` the
  line should be test-exercised (install the extra in CI). If the
  extra is truly environment-specific (GPU, vendor SDK) the line
  gets a `pragma: no cover` with an explanatory comment.
* **Import-guard** — a `try: import X ... except ImportError: ...`
  pattern where one branch is necessarily taken and the other
  cannot be reached at runtime in the current install. Annotate
  with `pragma: no cover` and cite which side is unreachable.
* **Defensive** — a numerical edge branch (`if denom < 1e-30`) or
  a typeguard (`isinstance`) that cannot be triggered by any
  legitimate caller. Decide per-case: either synthesise the edge
  input (better — keeps coverage honest), or delete the branch if
  it guards against a state that cannot occur by construction.
* **Env-blocked** — the local Ubuntu 24.04 system `mpl_toolkits`
  3.6.3 vs user-site `matplotlib` 3.10.8 conflict prevents
  `mitiq`/`cirq` from importing at all, so the related modules
  cannot be test-exercised on this workstation. Either fix the
  environment at the infrastructure layer, or annotate the
  affected code with `pragma: no cover` and document the block
  here.

## Heavy offenders (> 10 uncovered lines)

| File | Missing | Cov | Class | Target action |
| --- | ---: | ---: | --- | --- |
| `phase/pulse_shaping.py` | 48 → ~4 | 67 → 97% | Fallback | **CLOSED** in commit `6c3e29e` — 26 new tests; 4 remaining lines are the import-guard at 43-44 (unreachable while the Rust wheel is installed) + one path in hypergeometric that differs from Rust at non-zero (α, β). Annotate remaining as Import-guard. |
| `mitigation/mitiq_integration.py` | 32 | 30 % | Env-blocked | Either fix the Ubuntu `mpl_toolkits`/`matplotlib` conflict so `mitiq` imports locally, or `pragma: no cover` the whole `_MITIQ_AVAILABLE = True` branch with a cross-reference to `docs/internal/` pointing at this file. |
| `psi_field/lattice.py` | 29 | 79 % | Gap + Defensive | Lines 41-42 are an import guard. Lines 134, 151, 155-161, 193-196, 227-250 are lattice-operation branches not exercised by current tests. Add tests for non-trivial lattice sizes and boundary cases. |
| `bridge/snn_adapter.py` | 21 | 63 % | Gap | sc-neurocore bridge; test coverage skipped because `sc-neurocore` is pinned in CI via `xvalidate` extra but not exercised for every bridge path. Audit the missing lines and add unit tests for each adapter method. |
| `qec/multiscale_qec.py` | 19 | 83 % | Gap | Lines 61-62, 120-128, 152-158, 234 — rarer QEC schedules (multi-round, non-standard code distance). Add parametric tests. |
| `hardware/runner.py` | 16 | 94 % | Gap | IBM-hardware wrappers that the existing mocked tests don't cover. Extend `test_hardware_runner.py` (currently `--ignore`d in CI — needs revisiting). |
| `analysis/dynamical_lie_algebra.py` | 14 | 94 % | Gap | Specific DLA construction paths (non-connected Lie-algebra branches). Add tests for the adjacency cases that are not exercised. |
| `fep/variational_free_energy.py` | 14 | 77 % | Gap | Free-energy computation for optional input shapes (sparse coupling, non-uniform ω). Add targeted tests. |
| `fep/predictive_coding.py` | 14 | 64 % | Gap | Same pattern as above; triage which code paths have real users and cover those first. |
| `mitigation/symmetry_decay.py` | 12 | 79 % | Gap | Lines 31-32, 85-96 — alternative decay models. |
| `psi_field/observables.py` | 12 | 65 % | Gap | Observable computation paths for non-default lattice sizes. |
| `psi_field/infoton.py` | 11 | 76 % | Gap | Lines 40-41 (import guard), 105-115 (alternate operator path). |

## Medium offenders (5–10 uncovered lines)

| File | Missing | Cov | Class |
| --- | ---: | ---: | --- |
| `phase/xy_kuramoto.py` | 10 | 85 % | Gap |
| `phase/adapt_vqe.py` | 9 | 90 % | Gap |
| `hardware/gpu_accel.py` | 7 | 88 % | Optional-backend (CUDA) — consider `pragma: no cover` |
| `phase/lindblad_engine.py` | 5 | 96 % | Gap |
| `hardware/backends.py` | 5 | 95 % | Gap |
| `phase/mps_evolution.py` | 5 | 94 % | Gap |
| `hardware/qubit_mapper.py` | 5 | 93 % | Gap |
| `hardware/provenance.py` | 5 | 90 % | Gap |

## Small offenders (3–4 uncovered lines)

| File | Missing | Cov | Class |
| --- | ---: | ---: | --- |
| `hardware/experiment_mitigation.py` | 4 | 97 % | Gap |
| `analysis/finite_size_scaling.py` | 4 | 94 % | Gap |
| `hardware/classical.py` | 3 | 98 % | Gap |
| `hardware/pennylane_adapter.py` | 3 | 97 % | Gap |
| `benchmarks/quantum_advantage.py` | 3 | 97 % | Gap |
| `hardware/async_runner.py` | 3 | 96 % | Gap |
| `l16/quantum_director.py` | 3 | 95 % | Gap |
| `benchmarks/gpu_baseline.py` | 3 | 95 % | Optional-backend (CUDA) |
| `phase/contraction_optimiser.py` | 3 | 91 % | Gap |

## One-line and two-line offenders

42 files in total. Each entry is a single-line defensive branch,
an import guard, or a small edge case. They are grouped here by
subsystem for easier batch processing when their dedicated commit
lands.

### `analysis/`

* `entanglement_enhanced_sync.py:206` — Defensive.
* `koopman.py:202` — Gap.
* `loschmidt_echo.py:90` — Defensive.
* `magic_nonstabilizerness.py:77` — Defensive (max-magic edge).
* `magnetisation_sectors.py:231` — Gap.
* `monte_carlo_xy.py:269` — Gap (default-arg path).
* `otoc.py:178` — Gap (small-input branch).
* `qfi.py:113` — Defensive (denom < 1e-30).
* `qfi_criticality.py:94` — Defensive.
* `quantum_persistent_homology.py:199` — Gap.
* `quantum_speed_limit.py:124` — Gap.

### `applications/`

* `eeg_benchmark.py:113` — Defensive (NaN-check).
* `fmo_benchmark.py:129` — Defensive (NaN-check).
* `iter_benchmark.py:113` — Defensive (NaN-check).
* `power_grid.py:141` — Defensive (NaN-check).

### `control/`

* `vqls_gs.py:102` — Gap.
* `topological_optimizer.py:82` — Gap.

### `gauge/`

* `confinement.py:86` — Defensive (ratio ≤ 0).
* `wilson_loop.py:102` — Defensive (sparse conversion).

### `hardware/`

* `cirq_adapter.py:28` — Import-guard (Env-blocked).
* `experiment_dynamics.py:1` — Gap.
* `fast_classical.py:2` — Gap.

### `mitigation/`

* `symmetry_verification.py:40-41` — Import-guard (Rust wheel).

### `phase/`

* `ansatz_methodology.py:207` — Gap.
* `avqds.py:92, 125` — Gap.
* `backend_selector.py:110` — Gap.
* `floquet_kuramoto.py:167` — Gap.
* `tensor_jump.py:125` — Gap.
* `varqite.py:77` — Gap.

### `qec/`

* `control_qec.py:224` — Gap.

### `qsnn/`

* `dynamic_coupling.py:39-40` — Import-guard (sc-neurocore).

### Root

* `backend_dispatch.py:2` — Gap.

## Test-ordering flakiness (separate work item)

Ten tests pass in isolation but fail when the full suite runs in
its current ordering:

* `test_persistent_homology.py::TestComputePersistence::*` (4 cases)
* `test_persistent_homology.py::test_persistence_all_pi`
* `test_persistent_homology.py::test_persistence_random_phases`
* `test_persistent_homology.py::test_persistence_2_oscillators`
* `test_persistent_homology.py::test_persistence_threshold_effect`
* `test_quantum_persistent_homology.py::TestCompareQuantumClassical::test_returns_both`
* `test_qubit_mapper.py::TestPerformance::test_community_detection_fast`

Every failure is a `NameError` or timing-related assertion. Root
cause is almost certainly cross-test fixture or module-level
state pollution. This blocks a clean 100 % landing because the
coverage CI gate implicitly requires all tests green. Triage:

1. Reproduce under `pytest -p no:randomly` to eliminate
   randomised order as a cause.
2. Bisect which predecessor test contaminates the state.
3. Fix the offender with a proper fixture tear-down rather than
   silencing the affected test.

This work item is tracked separately from the coverage push
proper — it is a correctness issue, not a coverage one.

## Plan for the coverage push proper

The heavy and medium offenders are each a single commit. The
small offenders batch into five commits by subsystem. Each commit
follows the same gate audit as every other in the repo — SPDX,
Co-Authored-By, British English, no quality labels, no simplified
models, real tests not placeholder. No commit decreases coverage.
No commit introduces `# pragma: no cover` without a one-sentence
justification cross-linking this file.

Indicative commit budget to land 315 → 0 uncovered:

| Batch | Count | Commits |
| --- | ---: | ---: |
| Heavy offenders (> 10 lines each) | 12 | 12 |
| Medium offenders (5–10 lines) | 8 | 8 |
| Small offenders (3–4 lines) | 9 | 2–3 |
| One- and two-line offenders | 42 | 5 |
| Test-ordering flakiness | 10 tests | 1–3 |
| Final `pragma: no cover` sweep + docs update | — | 1 |
| **Total** | — | **~30** |

Each cycle includes a full-suite `pytest --cov` re-measure
(~25 minutes) to verify the intended delta.

## Standing rule going forward

New modules land with:

1. Multi-angle tests that exercise every branch. Happy path
   **plus** every `if`/`elif`/`try`/`except` branch, **plus**
   every numerical edge case the code defends against. No
   module reaches merge without this.
2. A docs page (or section in an existing page) describing the
   module's purpose, inputs, outputs, worked example, and the
   tolerance / approximation regime it operates in.
3. A pre-commit gate that refuses to accept a diff that decreases
   coverage on the module it touches. (Infrastructure task — not
   yet shipped; tracked as part of the broader CI tightening.)

Failure to ship all three at module landing is how the 315-line
gap accumulated. No future commit should reproduce that failure.

Mutation testing — the mechanical verification of rule (1) — is
documented separately in
[`docs/mutation_testing.md`](docs/mutation_testing.md). Line
coverage says "this line ran"; mutation testing says "this
line's behaviour was actually constrained by an assertion". A
module with 100 % line coverage and 80 % survived mutants is
coverage-theatre. Both numbers matter.
