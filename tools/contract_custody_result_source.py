# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable result source custody
"""Capture a stable Result codec roundtrip with an explicit fixture projection."""

from __future__ import annotations

import json
from typing import Any

from scpn_quantum_control import stable_core_product as codec
from scpn_quantum_control.stable_core import build_result
from tools.contract_custody_hal_source import hal_evidence_source


def stable_result_evidence_source() -> dict[str, Any]:
    """Construct and roundtrip a real Result without claiming a native HAL adapter.

    Returns
    -------
    dict
        Full original HAL source, explicit projected inputs and native v2
        result envelope, rebuilt result and hashes. The projection is test
        construction, not shipped adapter support or hardware qualification.

    """
    source = hal_evidence_source()
    counts = source["result"]["counts"]
    inputs: dict[str, Any] = {
        "experiment_id": "source-offline-projection",
        "backend_id": source["job"]["backend_id"],
        "status": "succeeded",
        "observables": {"zero_fraction": counts["00"] / source["result"]["shots"]},
        "artifacts": [],
        "blockers": [],
        "metadata": {
            "source_job_id": source["job"]["job_id"],
            "claim_boundary": "explicit fixture projection of local deterministic counts; no native HAL-to-Result adapter claimed",
        },
    }
    result = build_result(**inputs)
    envelope = codec.serialise_result(result)
    rebuilt = codec.deserialise_result(envelope)
    detached: dict[str, Any] = json.loads(
        codec.canonical_json_bytes(
            {
                "producer": "scpn_quantum_control.stable_core.build_result",
                "reader": "scpn_quantum_control.stable_core_product.deserialise_result",
                "inputs": inputs,
                "source": source,
                "source_sha256": codec.digest_stable_core_payload(source),
                "native_type": f"{type(result).__module__}.{type(result).__qualname__}",
                "native": result.to_dict(),
                "envelope": envelope,
                "rebuilt": rebuilt.to_dict(),
                "reserialised": codec.serialise_result(rebuilt),
                "envelope_sha256": codec.digest_stable_core_payload(envelope),
                "binding_status": "explicit_test_projection_not_production_adapter",
                "unavailable": [
                    "companion_validation",
                    "hardware_observation",
                    "native_hal_result_adapter",
                ],
            }
        )
    )
    return detached
