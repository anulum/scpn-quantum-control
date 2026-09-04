<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM DLA submission-journal runbook -->

# IQM DLA Submission-Journal Runbook

Date: 2026-07-26

This runbook documents the crash-safe submission boundary implemented by
`scripts/run_iqm_dla_powered_block.py`. It does not change any frozen circuit,
shot, layout, timing, budget, or statistical rule in the campaign
preregistrations. It changes only how the runner records and recovers provider
calls.

## Safety contract

Before each IQM provider call, the runner atomically writes and `fsync`s a
versioned JSON journal whose job group is in state `submitting`. The journal
contains:

- the campaign, backend, date, repetition/window, and selected layout;
- every transpiled depth;
- the exact labels, circuit names, and shot count for each job group;
- a SHA-256 digest of the complete serialised IQM circuit payload and all
  execution-affecting parameters;
- one state and provider job ID per `main` or `readout` group.

After IQM returns a job ID, that group is atomically advanced to `submitted`
before the next provider call. A process restart:

- returns without provider contact when every group is already `submitted`;
- skips a completed group and submits only a remaining `prepared` group;
- refuses all submission when any group is `submitting` or
  `recovery_required` without a job ID.

The last case is intentionally fail-closed. A timeout can occur after the
provider accepted a job but before the client received its ID. Retrying that
call would risk a duplicate paid job.

## Normal window-variability submission

After the preregistered wall-clock, calibration, layout, depth, budget, and
owner-GO gates pass, use the existing campaign command and a new output path:

```bash
.venv-iqm/bin/python scripts/run_iqm_dla_powered_block.py submit \
  --campaign window-variability \
  --quantum-computer garnet \
  --layout primary \
  --window 3 \
  --date 2026-07-26 \
  --out data/iqm_paper_replication/iqm_dla_window_variability_submission_observation_window_03_2026-07-26.json \
  --i-have-owner-go
```

For the window-variability campaign, `--window` is restricted to the frozen
range 1–10. A pre-existing
legacy record, mismatched campaign identity, altered payload, duplicate job ID,
or malformed journal is never overwritten.

## Ambiguous-call recovery

If submission exits with code `4` or a process interruption leaves a group in
`submitting`, do not run a new submission immediately.

1. Inspect the IQM Resonance dashboard and identify the exact job created by
   the interrupted call.
2. Bind that provider job to the matching group:

```bash
.venv-iqm/bin/python scripts/run_iqm_dla_powered_block.py recover \
  --record data/iqm_paper_replication/iqm_dla_window_variability_submission_observation_window_03_2026-07-26.json \
  --group main \
  --job-id EXACT-IQM-JOB-UUID \
  --i-confirm-provider-job
```

Recovery retrieves the provider-side submitted payload and accepts the binding
only when its full digest equals the frozen pre-submit digest for that group.
A wrong job, wrong circuit order, wrong calibration set, wrong shot count, or
different compilation option is rejected. The command does not submit a new
job.

After every ambiguous group is bound, rerun the original `submit` command. It
reuses the recovered job ID and submits only any still-`prepared` group. Then
retrieve counts from the completed journal with the existing `retrieve`
subcommand.

## Claim boundary

The journal proves client-side intent, provider payload identity, and durable
job-ID custody. It does not provide provider-side request idempotency, estimate
credit consumption, replace dashboard confirmation after an ambiguous network
response, or relax any campaign gate. Earlier observation-window 1 and 2 submission records remain
valid for retrieval and are not rewritten into the new schema.
