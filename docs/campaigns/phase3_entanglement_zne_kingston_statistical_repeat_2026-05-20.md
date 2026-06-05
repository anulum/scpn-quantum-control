<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Phase 3 Kingston ZNE statistical repeat -->

# Phase 3 Kingston ZNE Statistical Repeat

Date: 2026-05-20

## Purpose

This run is a same-backend, same-layout statistical repeat for the Phase 3
reduced-Pauli ZNE paper result. It is not a new exploratory protocol. It keeps
the completed Kingston third-backend ZNE design fixed and tests whether the
DLA-worsens / FIM-improves pattern survives an independent execution under
same-day backend and calibration drift.

## Fixed Protocol

| Field | Value |
|---|---|
| Backend | `ibm_kingston` |
| Physical qubits | `[141,142,143,144]` |
| Channels | `dla_odd_signal/XXII`, `dla_odd_signal/YYII`, `dla_odd_shallow/IIXX`, `dla_odd_shallow/IIYY`, `fim_lambda4_feedback/IZZI` |
| Noise scales | `1,3,5` |
| Repetitions | `3` per channel and scale |
| Main shots | `2048` |
| Readout shots | `8192` |
| Readout calibration | full 16-state computational-basis calibration |
| Budget ceiling | `25.0` QPU minutes |

## Readiness

No-submit readiness command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/phase3_entanglement_tomography_ibm.py \
  --backend ibm_kingston \
  --physical-qubits 141,142,143,144 \
  --zne-subset-rows data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez_pinned_full_readout.csv \
  --zne-noise-scales 1,3,5 \
  --max-depth 1900 \
  --max-total-gates 5000
```

Readiness artefact:

- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T114659Z.json`
- SHA256: `6b06eeba71d7ff12b3990e9eaaa6ec68457de33eb9c1c45232506b3c9826df7b`

Readiness result:

| Field | Value |
|---|---:|
| Status | `readiness_passed` |
| Main circuits | 45 |
| Readout circuits | 16 |
| Estimated QPU minutes | 0.5591666666666667 |
| Budget ceiling minutes | 25.0 |
| Maximum transpiled depth | 1683 |
| Maximum basis-expansion ratio | 4.836206896551724 |

## Submission And Completion

Approved async submission command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/phase3_entanglement_tomography_ibm.py \
  --backend ibm_kingston \
  --physical-qubits 141,142,143,144 \
  --zne-subset-rows data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez_pinned_full_readout.csv \
  --zne-noise-scales 1,3,5 \
  --max-depth 1900 \
  --max-total-gates 5000 \
  --submit-async \
  --confirm-budget
```

Canonical pending artefact:

- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T114719Z.json`
- SHA256 after pending-job registration:
  `060718902e20d92e910470637f01fe3d58b67d483d22618dbd0ae611bd06459e`

IBM jobs:

| Role | Job ID | Status at submission verification |
|---|---|---|
| Main ZNE circuits | `d86pul0p0eas73dla3dg` | `RUNNING` |
| Full 16-state readout calibration | `d86pul8p0eas73dla3eg` | `RUNNING` |

Submission readiness snapshot:

| Field | Value |
|---|---:|
| Main circuits | 45 |
| Readout circuits | 16 |
| Estimated QPU minutes | 0.5591666666666667 |
| Maximum transpiled depth | 1657 |
| Maximum basis-expansion ratio | 4.735632183908046 |

Both jobs completed and were hydrated into the canonical artefact.

Reducer command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyse_phase3_entanglement_zne.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T114719Z.json \
  --result-tag 2026-05-20_ibm_kingston_zne_repeat
```

Reducer outputs:

- `data/phase3_entanglement_tomography/entanglement_zne_summary_2026-05-20_ibm_kingston_zne_repeat.json`
- `data/phase3_entanglement_tomography/entanglement_zne_scale_rows_2026-05-20_ibm_kingston_zne_repeat.csv`
- `data/phase3_entanglement_tomography/entanglement_zne_channel_summary_2026-05-20_ibm_kingston_zne_repeat.csv`
- `docs/campaigns/phase3_entanglement_zne_manifest_2026-05-20_ibm_kingston_zne_repeat.md`

Result snapshot:

| Field | Value |
|---|---:|
| Scale rows | 15 |
| Channels | 5 |
| Scale-1 mean absolute deviation | 0.4655910323155505 |
| Linear ZNE mean absolute deviation | 0.48161750800999487 |
| Readout-mitigated linear ZNE mean absolute deviation | 0.48931903445861447 |
| Quadratic ZNE mean absolute deviation | 0.48251293904941395 |

Channel-level result: all four DLA transverse channels again move farther from
exact under linear ZNE, while the FIM control improves. This satisfies the
paper-use criterion for same-layout drift robustness.

## Claim Boundary

This is a completed statistical repeat for drift robustness of the already
completed Kingston ZNE stress test. It does not expand the system size, does
not add new channels, does not claim backend-general dynamics, and does not
change the preregistered five-channel ZNE claim boundary.

## Recovery

Check live job status:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
import importlib.util
from qiskit_ibm_runtime import QiskitRuntimeService

spec = importlib.util.spec_from_file_location(
    "phase1", "scripts/phase1_mini_bench_ibm_kingston.py"
)
phase1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase1)
credential_value, instance = phase1.parse_vault(
    Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
)
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=credential_value,
    instance=instance,
)
for role, job_id in {
    "main": "d86pul0p0eas73dla3dg",
    "readout": "d86pul8p0eas73dla3eg",
}.items():
    job = service.job(job_id)
    print(f"{role}\t{job_id}\t{job.status()}")
PY
```

The completed reduction command was:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyse_phase3_entanglement_zne.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T114719Z.json \
  --result-tag 2026-05-20_ibm_kingston_zne_repeat
```

Paper-use criterion:

- Satisfied: the repeat again shows all four DLA transverse channels worsening
  under linear ZNE while the FIM control improves.
