# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Mutation Testing

# Mutation Testing

Line-coverage tells you which statements ran. It does **not** tell
you which logic changes your tests would catch. Mutation testing
flips operators and constants in the source one at a time, reruns
the suite, and measures how many mutants "survived" — a surviving
mutant is a behaviour the tests do not constrain.

`scpn-quantum-control` uses [mutmut
2.5](https://mutmut.readthedocs.io/) and gates a representative
module on every release cycle. The long-term aim is to extend the
target set, not to chase a perfect mutation score on any single
file.

## Current targets

| Module | LOC | Test files | Runner |
| --- | ---: | --- | --- |
| `analysis/koopman.py` | ~250 | `tests/test_koopman.py` | `tools/mutmut_runner.sh` |
| `bridge/knm_hamiltonian.py` | 236 | `tests/test_knm_hamiltonian.py` + mutation kills + parity + properties | `tools/mutmut_runner_knm.sh` |
| `analysis/otoc.py` | 197 | `tests/test_otoc.py` + mutation kills + sync probe | `tools/mutmut_runner_otoc.sh` |

Each target has its own shell-script runner in `tools/` that runs
only the tests covering that module, keeping per-mutant wall time
at a few seconds rather than the 25-minute full-suite round.

Config in `pyproject.toml [tool.mutmut]`. Invoked per module via:

```bash
mutmut run --paths-to-mutate src/scpn_quantum_control/bridge/knm_hamiltonian.py \
           --tests-dir tests/ \
           --runner="$(pwd)/tools/mutmut_runner_knm.sh"
```

The runner must be a **shell script with an absolute path**; mutmut
2.5 splits `--runner` args on whitespace via
`subprocess.Popen(args_list)` without `shell=True`, so passing
`--runner="python -m pytest ..."` fails with `FileNotFoundError:
[Errno 2] No such file or directory: '-m'`. Wrap the command in a
script — see the three runners in `tools/`.

## Baseline (2026-04-17, v0.9.6)

Partial run — 24 of ~95 generated mutants tested before CI-time
budget expired. Sample is representative of the first ~40 lines
(imports + `_validate_inputs`).

| Outcome | Count | Notes |
| --- | --: | --- |
| Killed | 12 | Tests caught the mutation (expected behaviour) |
| Survived | 12 | See classification below |
| Untested | 71 | Ran out of wall-clock on the mutmut baseline; covered by the weekly CI job |

Survived mutants, classified by hand (`mutmut show <id>`):

- **10 string-content mutations** — ids 9, 14, 16, 22, 24, 32, 34,
  35, 37 + one more. Example: mutmut rewrote
  `raise ValueError("K must be a square 2-D matrix, ...")` to
  `raise ValueError("XXK must be a square 2-D matrix, ...XX")`.
  The tests assert via `pytest.raises(ValueError, match="square 2-D matrix")`
  which still passes because "XX" wrapping preserves the substring.
  Known limitation of substring matching; intentional trade-off —
  error messages evolve, strict-string assertions would add
  busy-work on every refactor.
- **2 equivalent mutations** — ids 10 and 39. Both substitute
  `K.shape[0]` for `K.shape[1]`. Earlier validation already rejects
  non-square `K`, so the two values are identically equal at the
  mutation point. No observable behaviour change.

**Real miss rate on tested mutants: 0 %.** The 12 "survived"
results are all either string-content drift or provable equivalents.

## Baseline (2026-04-18) — `bridge/knm_hamiltonian.py`

Full run of the 102 generated mutants via
`tools/mutmut_runner_knm.sh` (108 knm-focused tests, ~5 s/mutant).

| Stage | Killed | Survived | Timed out | Suspicious |
| --- | ---: | ---: | ---: | ---: |
| Before new tests | 21 | 81 | n/a | n/a |
| After `tests/test_knm_hamiltonian_mutation_kills.py` (16 new tests) | **46 → 67** | **81 → 35** | 3 | 18 |

Kill delta: +46 mutants (~57 % of the first-baseline survivors).
The 35 still-surviving are dominated by:

* Equivalent mutants at L-dependent thresholds — `if L > 15` vs
  `if L >= 15` coincide for L ≠ 15; similar pattern at `L > 6`.
* Paper-27 sub-hierarchy boost paths that are dead code when
  `max(K[i, j], floor)` is dominated by `floor` at the tested
  K_base / K_alpha defaults.
* One real unreachable: `H_op = knm_to_xxz_hamiltonian(...) →
  H_op = None` inside `knm_to_dense_matrix`, after a
  `try: import scpn_quantum_engine` that succeeds in the dev
  environment. The mutated Python-fallback branch is never
  entered. Would be killed by monkey-patching the Rust engine off
  — deferred as a follow-up consistent with the `pulse_shaping`
  Rust-fallback tests.

Semantically meaningful mutants now killed by the new suite:

* Sign flips on XX / YY / ZZ coefficients — the Kuramoto-XY
  mapping convention is now enforced.
* Sign flip on Z onsite term — natural-frequency direction is
  pinned.
* Off-by-one in the pair loop — every (i, j) with j = i + 1 …
  n − 1 is asserted present.
* None propagation on every public return.
* Default L value, `L > 15` threshold, and cross-hierarchy boost
  index (K[4, 6] vs K[5, 6]).
* Each element of the 16-entry Paper 27 Table 1 natural-frequency
  array and each of the four Table 2 anchor values.

## Baseline (2026-04-18) — `analysis/otoc.py`

Full run of 38 generated mutants via `tools/mutmut_runner_otoc.sh`
(three otoc-focused test files).

| Stage | Killed | Survived |
| --- | ---: | ---: |
| Before new tests | 0 | 38 |
| After `tests/test_otoc_mutation_kills.py` (9 new tests) | **9** | **29** |

Kill delta: +9 mutants (~24 % of the first baseline). The 29
still-surviving are dominated by boundary-condition equivalents
(`len(x) < 3` vs `len(x) <= 3` at exactly-three-point inputs) and
string-content mutations in docstrings and error messages. Further
kills require either richer OTOC numerical oracles or accepting
these as classified equivalents — tracked as follow-up.

Semantically meaningful kills:

* Pauli matrix element perturbations — X/Y/Z matrices now have
  their diagonals, off-diagonals, and squared-identity relation
  enforced.
* First-crossing vs later-crossing in scrambling-time estimator —
  `below[0]` semantics pinned.
* `f0 = 0` zero-guard — both estimators return `None` rather
  than dividing by zero.

## Policy

- **Release gate (soft):** every release cycle runs the full
  `mutmut run` on each target and records the updated score here.
  A real regression (drop in killed-vs-survived ratio on
  semantically-meaningful mutants) blocks the release; a shift in
  string-content survivors is expected and does not.
- **Follow-up work (audit item B7):**
  1. Extend the target list to `analysis/krylov_complexity.py`
     and the `phase/` core after otoc and knm_hamiltonian stay
     stable across two CI cycles.
  2. Write `tools/mutmut_equivalents.py` so the CI run exits 0
     on the known-safe set without human classification each
     time.
  3. Evaluate `pytest-mutagen` / `cosmic-ray` parallel executors
     for larger sweeps.

## CI integration

`.github/workflows/mutation-testing.yml` runs `mutmut run` weekly
on Monday 04:00 UTC (before the commit-trailer audit at 05:00 and
the link-check at 05:30). Results land as a workflow artifact and
drive a succinct job summary. The workflow is non-blocking
(`continue-on-error: true`) until three consecutive weekly runs
complete without a surprise semantic survivor — then it promotes
to a blocking gate.

Manual dispatch is always available for on-demand runs during
test-hardening work.

## How to interpret the output

- `mutmut results` — list of mutant ids by outcome category.
- `mutmut show <id>` — diff of the mutation and the affected file
  location.
- `mutmut html` — browsable HTML report under `mutmut-html/`
  (gitignored).

When a new survivor shows up on the weekly run:
1. `mutmut show <id>` — classify: killed, equivalent, or real miss.
2. If real miss: write the test that would have caught it, rerun
   locally to confirm killed, PR.
3. If equivalent: add to `tools/mutmut_equivalents.py` (follow-up).
4. If string-content: no action unless the target-message wording
   is itself a public interface.

Audit item **B7** in the internal gap audit closes when the CI
workflow has completed three consecutive green weekly runs and
the target-module list has expanded beyond `koopman.py`. As of
2026-04-18, `knm_hamiltonian.py` and `otoc.py` are in the target
list and the CI workflow remains the three-weeks-green gate.

## Connection to the "new code = new tests" rule

The rule recorded in `TODO_COVERAGE.md` — **new code = new
tests (multifaceted) = new superior docs** — is what mutation
testing mechanically verifies. Line coverage proves that a line
ran; mutation testing proves that a line's *behaviour* was
constrained by an assertion. A module with 100 % line coverage
but 80 % survived mutants is a module whose tests are
coverage-theatre. Both numbers matter; the CI workflow defends
both.
