# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Pre-registration protocol

# Pre-registration Protocol for Hardware Campaigns

## Why pre-register

Hardware experiments on a shared QPU are expensive and
under-specified by nature — the submitter decides post hoc what
counts as a "successful" run. Without a protocol frozen before the
first circuit is submitted, hypothesis-after-the-results
(HARKing), garden-of-forking-paths analysis, and selective
reporting become indistinguishable from honest discovery. The
Phase 1 DLA parity campaign (`data/phase1_dla_parity/`) pre-dates
this policy but was documented retroactively in
`docs/falsification.md` C2 with the specific falsifiers we would
now have named in a pre-registration.

Every scpn-quantum-control hardware campaign from Phase 2 onward
must be pre-registered on OSF (Open Science Framework) before the
first circuit is submitted to any backend. The pre-registration
URL goes into the campaign's entry in `docs/results.md` at the
moment the campaign starts, not after it finishes.

## What gets pre-registered

A single markdown-size document covering:

1. **Hypothesis.** A claim of the form used in
   `docs/falsification.md` — one sentence, one observable, one
   domain of validity.
2. **Predicted signal size** with a 95 % apriori confidence
   interval from the classical (noiseless) simulator.
3. **Decision procedure.** Exactly one primary statistic (e.g.
   Welch's two-sample $t$-test on per-depth parity leakage) and
   the decision rule that turns its value into
   confirm / falsify / inconclusive.
4. **Circuit budget** — total circuits, per-depth reps, backend
   name, shot count. Changing any of these during the run
   requires a protocol amendment that is itself timestamped.
5. **Pre-specified subgroup analyses** — every slice of the data
   we already plan to report (e.g. "depth 4 only"). Subgroups
   added after seeing the data are reported separately as
   exploratory with that label.
6. **Known confounds + pre-specified robustness checks.** Example
   from Phase 1 that would now be pre-registered: popcount-matched
   control circuits, randomised schedule order.
7. **Analysis script.** Path in the repo, frozen at a specific
   commit hash, that will take the raw result JSON and produce
   the decision statistic.
8. **Data deposit target.** Zenodo DOI (new version of the
   dataset DOI) before publication; GitHub path during the
   submission window.
9. **Authors.** Who is responsible for each section; who has
   backend access.

Template lives at `docs/preregistration_template.md`; fill in a
copy under `.coordination/preregistrations/<campaign>.md`
(gitignored local working copy) and upload to OSF for the frozen
copy.

## When the freeze happens

The pre-registration is considered frozen at the moment the OSF
record is created with the **registered** flag (OSF supports
both "draft" and "registered" states). Subsequent edits leave a
timestamp and diff; the decision statistic cannot be swapped
silently.

In the repository workflow:

- **Before freeze:** all iteration lives in
  `.coordination/preregistrations/<campaign>.md` (local). No
  artefacts are uploaded to the backend.
- **At freeze:** OSF record is created, the DOI for the
  registration is added to the campaign's row in
  `docs/results.md`, and the `registered_at` timestamp lands in
  the per-campaign header of the submission script.
- **After freeze:** the submission script is run. Results are
  written under `data/<campaign>/` with the provenance block
  (`hardware/provenance.py`) including the OSF DOI so readers
  can cross-verify.
- **After analysis:** the analysis script runs, the decision
  statistic is computed, the `CHANGELOG.md` entry cites both the
  OSF DOI and the result git hash.

## Amendments

Pre-specified changes to the protocol are expected (e.g. IBM
changes the backend queue depth while we are running). Every
amendment:

1. Lives in a new file
   `.coordination/preregistrations/<campaign>_amendment_<N>.md`.
2. Cites the original OSF DOI and explains the trigger.
3. Creates a new OSF registration (amendment, not overwrite).
4. The amendment DOI is added to the campaign's row in
   `docs/results.md` alongside the original DOI.

Any amendment that changes the primary statistic or the
confirm / falsify rule is effectively a new study. Report it as
such.

## Retroactive record for Phase 1

The Phase 1 DLA parity campaign (April 2026, `ibm_kingston`) was
run before this policy existed. A retroactive pre-registration is
not scientifically meaningful — we cannot un-see the results. What
we do instead:

- `docs/falsification.md` C2 records the falsifier we would have
  pre-registered (mean asymmetry $\ge 2\%$ for depths $\ge 4$,
  sign not reversed, $\ge 7/8$ depths Welch-significant).
- `tests/test_phase1_dla_parity_reproduces.py` gates every future
  re-run on those numbers.
- No amendment is possible; the campaign is sealed.

## Upcoming campaigns to pre-register

- **Phase 2 DLA parity ($n = 4$ scaling, popcount control).** Draft
  exists in `.coordination/` per memory
  `reference_ibm_credits_application`. Freeze target: before the
  180-minute/year IBM Quantum Open Plan allocation activates.
- **Phase 3 Heron r2 cross-backend replication** (kingston +
  marrakesh). Freeze target: when both backends have matching
  calibration passes.

## References

- OSF pre-registration — <https://osf.io/prereg/>
- Simmons, Nelson, Simonsohn (2011), "False-Positive Psychology:
  Undisclosed Flexibility in Data Collection and Analysis Allows
  Presenting Anything as Significant" — <https://doi.org/10.1177/0956797611417632>
- Chambers, D. (2017), "The Seven Deadly Sins of Psychology" —
  Princeton University Press.

Audit item **C9** in
`docs/internal/audit_2026-04-17T0800_claude_gap_audit.md` closes
when the first campaign (Phase 2) completes this workflow end to
end, including the OSF DOI in `docs/results.md`.
