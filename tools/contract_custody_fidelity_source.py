# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — proposed fidelity boundaries
"""Preserve native uncertainty in explicitly unexecuted qualification scenarios.

The dimensionless units below are caller declarations for this synthetic fixture,
not metadata supplied by the native producer and not inferred physical units.
Scenario envelopes are review inputs, not additions to the companion wire schema.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scpn_quantum_control import stable_core_product as scp
from tools.contract_custody_derivative_source import derivative_evidence_source
from tools.contract_custody_design_vectors import valid_companion


def fidelity_design_scenarios(raw_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build source-backed positive and isolated unsupported-operation proposals.

    Parameters
    ----------
    raw_record
        Existing experiment envelope retained unchanged for the conversion case.

    Returns
    -------
    dict
        Three frozen review scenarios. Native standard errors and confidence
        radii remain separate with full evidence references. Explicit fixture
        unit declarations make the aggregation fault independent of missing
        native unit metadata. No conversion, aggregation or companion reader
        executes; expected outcomes are proposals, not qualification evidence.

    """
    source = derivative_evidence_source("parameter_shift")
    result = source["result"]
    components = []
    for kind in ("standard_error", "confidence_radius"):
        components.append(
            {
                "kind": kind,
                "estimand": "gradient in native parameter_names order",
                "method": result["method"],
                "value": deepcopy(result[kind]),
                "unit": "1",
                "assumptions": [
                    "caller declares this synthetic objective and both parameters dimensionless",
                    "supplied shifted estimates; not hardware observations",
                    "radius and standard error describe the same covariance, not independent errors",
                ],
                "evidence_ref": {
                    "source": "derivative",
                    "sha256": source["result_sha256"],
                    "field_path": f"result.{kind}",
                    "covariance_path": "result.covariance",
                    "confidence_level_path": "result.confidence_level",
                    "confidence_z_path": "result.confidence_interval.confidence_z",
                },
            }
        )
    positive = {
        "inputs": {
            "source_records": {"derivative": source},
            "unit_declaration": {
                "origin": "explicit synthetic fixture caller; not native producer metadata",
                "objective": "1",
                "parameters": {name: "1" for name in result["parameter_names"]},
            },
            "fidelity_components": components,
            "aggregation_request": None,
            "calibration_reference": None,
            "supported_transform_composition": [],
        },
        "expected_outcome": {
            "raw_readable": True,
            "source_result_sha256": source["result_sha256"],
            "preserve_components_separately": True,
            "aggregate_value": None,
            "decision": "accept_explicit_fixture_components",
            "executed": False,
        },
    }
    aggregation = deepcopy(positive)
    aggregation["inputs"]["aggregation_request"] = {
        "operation": "sum",
        "components": ["standard_error", "confidence_radius"],
        "justification": None,
    }
    aggregation["expected_outcome"]["decision"] = "refuse_unjustified_error_aggregation"
    raw_digest = scp.digest_stable_core_payload(raw_record)
    conversion = {
        "inputs": {
            "raw_record": deepcopy(raw_record),
            "companion": valid_companion(raw_digest),
            "transform_request": {
                "operation": "unit_conversion",
                "field_path": "fields.omega",
                "source_unit": "rad/s",
                "target_unit": "Hz",
                "accepted_transform_ref": None,
            },
        },
        "expected_outcome": {
            "raw_readable": True,
            "raw_digest": raw_digest,
            "decision": "refuse_unsupported_conversion",
            "converted_value": None,
            "persist_qualified_record": False,
            "executed": False,
        },
    }
    return {
        "fidelity_components_preserve_native_uncertainty": positive,
        "unjustified_error_aggregation_refused": aggregation,
        "unsupported_frequency_conversion_refused": conversion,
    }
