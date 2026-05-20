<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S1 Feedback Readiness Index -->

# S1 Feedback Readiness Index

Date: 2026-05-06

This index records the complete no-QPU readiness state for the S1 hybrid
classical--quantum feedback track. It is a review document, not a submission
approval.

## Current Status

S1 is technically prepared up to the no-submit readiness layer. A live QPU run is
not yet authorised because the manual live-submission preflight, real backend
metadata capture, live transpilation record, and explicit approval record are not
complete.

## Canonical One-command Reproduction

```bash
scpn-bench s1-feedback-ready
```

This regenerates:

- S1 control-loop latency artefacts;
- S1 preregistration JSON and Markdown;
- provider dry-run payloads;
- template capability-probe examples;
- synthetic raw-count analysis summary.

## Artefact Inventory

| Artefact | Purpose |
|----------|---------|
| `docs/campaigns/hybrid_feedback_loop_s1_2026-05-06.md` | S1 technical and scientific boundary note. |
| `docs/campaigns/s1_live_submission_preflight_2026-05-06.md` | Mandatory manual gate before live submission. |
| `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json` | Machine-readable preregistration package. |
| `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.md` | Human-readable preregistration summary. |
| `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json` | No-QPU control-plane latency summary. |
| `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.csv` | CSV form of the latency summary. |
| `data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json` | Synthetic non-hardware raw-count fixture. |
| `data/s1_feedback_loop/s1_feedback_analysis_summary_2026-05-06.json` | Analysis output from the synthetic fixture. |
| `data/s1_feedback_loop/s1_ibm_metadata_template_2026-05-06.json` | Offline IBM/Qiskit metadata template. |
| `data/s1_feedback_loop/s1_ibm_metadata_probe_ibm_dynamic_metadata_template_2026-05-06.json` | No-submit IBM template capability decision. |
| `data/s1_feedback_loop/s1_generic_gate_metadata_template_2026-05-06.json` | Provider-neutral gate metadata template. |
| `data/s1_feedback_loop/s1_generic_gate_metadata_probe_openqasm3_dynamic_metadata_template_2026-05-06.json` | No-network generic gate capability decision. |

## Commands

Regenerate full no-QPU readiness bundle:

```bash
scpn-bench s1-feedback-ready
```

Regenerate preregistration only:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/export_s1_feedback_preregistration.py
```

Regenerate latency only:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_s1_feedback_loop.py
```

Rehearse analysis on synthetic fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyse_s1_feedback_hardware.py \
  data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json
```

Probe offline IBM metadata template:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_ibm_metadata.py \
  --metadata-json data/s1_feedback_loop/s1_ibm_metadata_template_2026-05-06.json
```

Probe provider-neutral gate metadata template:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_generic_gate_metadata.py \
  data/s1_feedback_loop/s1_generic_gate_metadata_template_2026-05-06.json
```

## Current Preregistered Job Shape

- Arms: monitored feedback and matched open-loop control.
- System qubits: 3.
- Monitor qubits: 1.
- Total qubits: 4.
- Classical bits: 6.
- Dynamic rounds: 3.
- Shots per circuit: 1024.
- Repetitions: 12.
- Circuits / arms: 2.
- Estimated execution seconds: 24.

## Claim Boundary

The S1 job can test whether a monitored feedback policy improves the
preregistered synchronisation target error relative to a matched open-loop
control under the same circuit family, shots, repetitions, and layout target.

It cannot, by itself, establish:

- quantum advantage;
- backend-independent behaviour;
- sub-microsecond real-time feedback unless provider-side logic implements it;
- analogue-native feedback suitability;
- general synchronisation protection beyond the tested payload.

## Remaining Live-submission Blockers

- [ ] Run `scpn-bench s1-feedback-ready` immediately before live review.
- [ ] Capture real backend metadata without submission.
- [ ] Run the capability probe against the real backend metadata.
- [ ] Perform live transpilation without submission.
- [ ] Record transpiled depth and operation counts.
- [ ] Complete `docs/campaigns/s1_live_submission_preflight_2026-05-06.md`.
- [ ] Create a new session log for the live-submission decision.
- [ ] Create a `HardwareApprovalRecord` matching the exact package hash.
- [ ] Confirm remaining QPU budget and approved QPU-second ceiling.
- [ ] Prepare raw-count archival path before submission.
- [ ] Wire provider submitter only after all gates above are complete.

## Platform Interpretation

IBM-style dynamic-circuit backends and generic dynamic-circuit gate providers are
the correct first live candidates for this specific payload. Analogue/native XY
platforms remain scientifically important, but they need a separate native
feedback formulation and dossier rather than this mid-circuit conditional
payload.
