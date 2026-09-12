# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — offline HAL source custody
"""Capture actual deterministic HAL execution without physical simulation claims."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from scpn_quantum_control import stable_core_product as codec
from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumWorkload,
)


def hal_evidence_source() -> dict[str, Any]:
    """Capture the full input and output of a fresh offline HAL job.

    Returns
    -------
    dict
        Original profile, workload, job and count result with qualified native
        identities. The injected adapter produces deterministic fixture counts;
        profile capabilities do not qualify SDK, statevector or QPU execution.

    """
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile("local_statevector")
    adapter = LocalDeterministicSimulator(profile)
    hal.register_backend(adapter)
    inputs: dict[str, Any] = {
        "workload_id": "source-offline-job",
        "ir_format": "openqasm3",
        "program": "OPENQASM 3; qubit[2] q; bit[2] c; c = measure q;",
        "n_qubits": 2,
        "shots": 16,
        "metadata": {"seed": 17},
    }
    workload = QuantumWorkload(**inputs)
    job = hal.submit(profile.backend_id, workload)
    result = hal.result(job)
    job_fields = {
        "job_id": job.job_id,
        "backend_id": job.backend_id,
        "workload_id": job.workload_id,
        "status": job.status,
        "metadata": dict(job.metadata),
    }
    result_fields = {
        "job": dict(job_fields),
        "status": result.status,
        "counts": dict(result.counts),
        "shots": result.shots,
        "metadata": dict(result.metadata),
    }
    source = {
        "producer": "scpn_quantum_control.hardware.hal.HardwareAbstractionLayer.result",
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "profile": asdict(profile),
        "workload": inputs,
        "job": job_fields,
        "result": result_fields,
        "observed_status": hal.status(job),
        "native_types": {
            name: f"{type(value).__module__}.{type(value).__qualname__}"
            for name, value in (
                ("profile", profile),
                ("workload", workload),
                ("job", job),
                ("result", result),
            )
        },
        "claim_boundary": "deterministic HAL contract adapter; not statevector evolution, provider SDK or hardware evidence",
        "unavailable": [
            "statevector_amplitudes",
            "calibration",
            "hardware_observation",
            "companion_validation",
        ],
    }
    detached: dict[str, Any] = json.loads(codec.canonical_json_bytes(source))
    return detached


def non_count_qualification_proposal() -> dict[str, Any]:
    """Propose refusal of amplitudes inferred from the actual count-only result.

    Returns
    -------
    dict
        Unexecuted qualification scenario with unchanged original HAL evidence.
        No counts are padded and no amplitudes or conversion are fabricated.

    """
    source = hal_evidence_source()
    return {
        "status": "proposed_refusal",
        "executed": False,
        "requested_quantity": "statevector_amplitudes",
        "source": source,
        "source_sha256": codec.digest_stable_core_payload(source),
        "qualification": "unavailable",
        "padding_or_conversion_performed": False,
        "reason": "profile declares support but actual adapter result contains counts only; amplitudes cannot be inferred",
    }
