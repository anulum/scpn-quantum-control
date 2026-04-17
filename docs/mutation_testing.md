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

## Current target

- **Module:** `src/scpn_quantum_control/analysis/koopman.py` (~250
  lines, 34 tests in `tests/test_koopman.py`).
- **Rationale:** small, self-contained, freshly hardened by commit
  `c7d4ccd` (input validation + 13 new tests). A representative
  surface for a first baseline.

Config in `pyproject.toml [tool.mutmut]`. Invoked via:

```bash
mutmut run --paths-to-mutate src/scpn_quantum_control/analysis/koopman.py \
           --tests-dir tests/ \
           --runner=/absolute/path/to/tools/mutmut_runner.sh
```

The runner must be a **shell script with an absolute path**; mutmut
2.5 splits `--runner` args on whitespace via `subprocess.Popen(args_list)`
without `shell=True`, so passing `--runner="python -m pytest ..."`
fails with `FileNotFoundError: [Errno 2] No such file or directory:
'-m'`. Wrap the command in a script — see `tools/mutmut_runner.sh`.

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

## Policy

- **Release gate (soft):** every release cycle runs the full
  `mutmut run` on the current target and records the score here.
  A real regression (drop in killed-vs-survived ratio on
  semantically-meaningful mutants) blocks the release; a shift in
  string-content survivors is expected and does not.
- **Follow-up work (audit item B7):**
  1. Extend `paths_to_mutate` to include
     `analysis/otoc.py`, `analysis/krylov_complexity.py`,
     `bridge/knm_hamiltonian.py`, then the `phase/` core. Each
     extension lands only after the existing target is
     consistently green.
  2. Consider pytest's `pytest-mutagen` or `cosmic-ray` for
     parallel execution of the test suite — mutmut 2.5 runs serially
     and a ~100-mutant sweep on the current suite takes ~2 h on a
     single core.
  3. Write an `equivalent-mutants` ignore list in
     `tools/mutmut_equivalents.py` so a scheduled CI run can exit 0
     on the known-safe set without human classification each time.

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

Audit item **B7** in
the internal gap audit closes
when the CI workflow has completed three consecutive green weekly
runs and the target-module list has expanded beyond `koopman.py`.
