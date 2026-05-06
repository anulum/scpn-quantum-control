<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Metriq Local Smoke -->

# Metriq Local Smoke

Date: 2026-05-06

This note records a no-QPU Metriq-Gym local simulator smoke run for the
Metriq submission readiness gate.

## Scope

The run used a standard Metriq-Gym benchmark definition and the local Qiskit
Aer simulator. It did not use IBM Quantum hardware, did not upload a result,
and did not submit an SCPN-specific benchmark to Metriq.

## Environment

Metriq-Gym was executed from the isolated environment:

```text
/home/anulum/.venvs/scpn-metriq
```

The project environment remained separate:

```text
.venv-linux
```

## Benchmark Configuration

The temporary configuration used for the smoke run was:

```json
{
  "benchmark_name": "Bernstein-Vazirani",
  "shots": 100,
  "min_qubits": 2,
  "max_qubits": 2,
  "skip_qubits": 1,
  "max_circuits": 1,
  "method": 1,
  "input_value": 1
}
```

The configuration was kept in `/tmp` for execution because it is a disposable
Metriq-Gym smoke input, not a scientific SCPN benchmark artefact.

## Commands

```bash
/home/anulum/.venvs/scpn-metriq/bin/mgym job dispatch \
  /tmp/scpn_metriq_bv_local_smoke_2026-05-06.json \
  --provider local \
  --device aer_simulator

/home/anulum/.venvs/scpn-metriq/bin/mgym job poll \
  b96914ac-2e2f-461e-84a8-b61d81300fb2
```

## Result

Metriq-Gym job ID:

```text
b96914ac-2e2f-461e-84a8-b61d81300fb2
```

Polled result summary:

```text
job_type: Bernstein-Vazirani
provider: local
device: aer_simulator
simulator: true
device_qubits: 30
runtime_seconds: 0.017248485004529357
accuracy_score: 1.0 ± 0.0
score: 1.0 ± 0.0
```

## Interpretation

This is a CLI and environment readiness check only. It confirms that the
installed Metriq-Gym path can dispatch and poll a local simulator benchmark
without QPU spend.

It does not establish:

- a Metriq upload;
- a Metriq-accepted SCPN benchmark;
- a QPU result;
- an SCPN package performance score;
- a comparison against the repository's Rust, VQE, DLA, FIM, or GPU
  benchmark artefacts.

## Remaining Gate

The remaining Metriq task is one of:

1. upload a valid Metriq-native benchmark result through the documented
   Metriq-Gym path after deliberate review; or
2. propose an SCPN/Kuramoto-XY benchmark schema upstream and wait for
   acceptance before any SCPN-specific result upload.
