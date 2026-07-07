# Actions History Dashboard

This page documents the read-only workflow-history audit layer used to keep
GitHub Actions history interpretable without deleting evidence for unresolved
defects.

The audit helpers classify workflow runs only. They never cancel runs, delete
runs, close issues, edit branches, or mutate GitHub state.

## Dashboard commands

Run the full Actions history classifier:

```bash
python tools/audit_github_actions_history.py \
  --repo anulum/scpn-quantum-control \
  --limit 500
```

Write the same report as JSON:

```bash
python tools/audit_github_actions_history.py \
  --repo anulum/scpn-quantum-control \
  --limit 500 \
  --json > actions-history-audit.json
```

Run the Link Check history classifier:

```bash
python tools/audit_link_check_history.py \
  --repo anulum/scpn-quantum-control \
  --limit 500
```

Use a captured `gh run list` fixture instead of querying GitHub:

```bash
gh run list \
  --repo anulum/scpn-quantum-control \
  --limit 500 \
  --json databaseId,conclusion,status,workflowName,headSha,headBranch,createdAt,event \
  > actions-history.json

python tools/audit_github_actions_history.py --input actions-history.json
python tools/audit_link_check_history.py --input actions-history.json
```

## Scheduled artefacts

The `Actions History Audit` workflow runs weekly and can also be triggered
manually. It uploads:

| Artefact | Meaning |
| --- | --- |
| `actions-history.json` | Raw `gh run list` output for the sampled history window. |
| `actions-history-audit.json` | Classified run history with buckets, reasons, and safe-delete candidates. |

The workflow is intentionally read-only. It uses `actions: read` and
`contents: read` permissions and uploads audit artefacts even when unresolved
history is detected.

## Classification buckets

| Bucket | Meaning | Safe to delete? |
| --- | --- | --- |
| `clean_success` | Completed successfully and retained as current evidence. | No |
| `in_progress` | Still running or not completed. | No |
| `resolved_failure` | Failed run with a later successful run for the same workflow and branch. | Candidate |
| `unresolved_failure` | Failed run without later successful evidence. | No |
| `superseded_cancelled` | Cancelled run with later successful evidence for the same workflow and branch. | Candidate |
| `unresolved_cancelled` | Cancelled run without later successful evidence. | No |
| `other_completed` | Completed with a non-standard conclusion. | No |

The Link Check helper uses the same principle, with link-specific buckets:

| Bucket | Meaning | Safe to delete? |
| --- | --- | --- |
| `clean_success` | Successful Link Check evidence. | No |
| `resolved_link_failure` | Link Check failure followed by a later successful Link Check on the same branch. | Candidate |
| `live_link_failure` | Link Check failure with no later successful evidence. | No |
| `accepted_external_transient` | Explicitly accepted external-service failure such as rate-limit or HEAD-request blocking. | No |
| `superseded_cancelled` | Cancelled Link Check followed by later successful evidence. | Candidate |
| `unresolved_cancelled` | Cancelled Link Check without later successful evidence. | No |

## Safe-delete rule

A failed or cancelled workflow run is only a deletion candidate when all of
these are true:

1. The run is completed.
2. The run is classified as `resolved_failure`, `resolved_link_failure`, or
   `superseded_cancelled`.
3. A later successful run exists for the same workflow and branch.
4. The run is not the only available evidence for an unresolved defect.
5. The deletion is performed manually after reviewing the audit report.

The helpers deliberately do not execute `gh run delete`.

## Accepted external-transient Link Check failures

Accepted transient failures should be explicit and narrow. Use JSON like:

```json
[
  {
    "databaseId": 123456789,
    "reason": "External journal returned HTTP 503 to anonymous HEAD requests."
  }
]
```

Then run:

```bash
python tools/audit_link_check_history.py \
  --input actions-history.json \
  --accepted accepted-link-failures.json
```

This keeps a visible distinction between real live link failures and external
services that intermittently reject automated checks.

## Release-safety boundary

The dashboard is a release-safety aid, not a replacement for engineering
judgement. A clean dashboard means the sampled history window has no
unresolved buckets under the classifier rules. It does not prove that every
possible historical defect has been fixed unless the sampled window covers the
entire relevant history and the latest CI/security/link-check runs are green.
