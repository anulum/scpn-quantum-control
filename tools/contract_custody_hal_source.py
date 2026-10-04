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
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumWorkload,
)


def hal_evidence_source(*, profile: BackendProfile | None = None) -> dict[str, Any]:
    """Capture a fresh offline HAL job in the original corpus v1 field layout.

    Parameters
    ----------
    profile
        Supplied local native declaration, or the built-in local statevector
        profile. Its capabilities govern the real sixteen-shot submission.

    Returns
    -------
    dict
        Original v1 profile, workload, job and count result with qualified native
        identities. The injected adapter produces deterministic fixture counts;
        profile capabilities do not qualify SDK, statevector or QPU execution.

    Raises
    ------
    ValueError
        A finite native shot limit cannot be represented by the original v1
        capture, or the supplied profile cannot admit the actual local job.

    Notes
    -----
    The original layout predates optional native shot limits and provider
    semantics. Only an unset shot limit is omitted; a declared limit refuses
    before execution rather than disappearing from evidence. The fresh local
    job has no workload semantics, submission companion or provider observation.
    Native HAL admission and dataclass fields retain their current contracts.

    """
    if profile is None:
        hal = HardwareAbstractionLayer.with_builtin_profiles()
        profile = hal.profile("local_statevector")
    else:
        hal = HardwareAbstractionLayer((profile,))
    if profile.capabilities.max_shots is not None:
        raise ValueError("HAL evidence v1 cannot represent declared max_shots")
    profile_fields = asdict(profile)
    del profile_fields["capabilities"]["max_shots"]
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
        "profile": profile_fields,
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
