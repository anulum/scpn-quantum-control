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
must commit its protocol to this repository and push that commit to the
public remote before the first circuit is submitted to any backend. The public statement is "protocol
committed before the run". Git author and commit timestamps are set by
the author; the externally checkable time is the push to the public
remote. No campaign has been registered with a third-party registry
such as OSF.

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

Template lives at `docs/preregistration_template.md`. The frozen
protocol is committed as `docs/campaigns/<campaign>_prereg_<date>.md`
together with the campaign manifest (observable, circuit family, shots,
abort criteria, statistics) and the analysis script it names.

## When the freeze happens

The protocol is frozen when its manifest and analysis script are
committed and that commit is pushed to the public remote. Later edits
are new commits with their own history; the decision statistic cannot be
swapped silently.

In the repository workflow:

- **Before freeze:** iteration happens in a local working copy. No
  circuits are submitted to a backend.
- **At freeze:** the protocol file is committed and pushed; the
  campaign's entry in `docs/results.md` cites that file.
- **After freeze:** the submission script runs. Results are written
  under `data/<campaign>/` with the provenance block
  (`hardware/provenance.py`), which records the exact repository commit
  that ran the submission and therefore contains the frozen protocol.
- **After analysis:** the committed reproducer recomputes every promoted
  statistic from raw counts and exits non-zero on failure; the
  `docs/results.md` entry cites the protocol file and the analysis.

## Amendments

Pre-specified changes to the protocol are expected (e.g. IBM
changes the backend queue depth while we are running). Every
amendment:

1. Is a new commit that leaves the original protocol intact.
2. Cites the original protocol file and explains the trigger.
3. Is pushed before any circuit it governs is submitted.
4. Is cited in the campaign's entry in `docs/results.md` alongside the
   original protocol file.

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

## Campaign status

The status of every campaign, including its protocol file, lives in
`docs/results.md`; this page describes the procedure only.

## References

- Simmons, Nelson, Simonsohn (2011), "False-Positive Psychology:
  Undisclosed Flexibility in Data Collection and Analysis Allows
  Presenting Anything as Significant" — <https://doi.org/10.1177/0956797611417632>
- Chambers, D. (2017), "The Seven Deadly Sins of Psychology" —
  Princeton University Press.

Third-party registration (for example on OSF) remains an option for a
future campaign; until one is made, no document should state that a
campaign was registered externally.
