#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm replay contract export
"""Export the fail-closed contract for the Paper 0 K_nm replay artefact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.run_paper0_knm_preregistered_replay import SCHEMA

CONTRACT_SCHEMA = "paper0_knm_preregistered_replay_contract_v1"
PAPER0_NAME = "GOTM-SCPN Paper 0: The Foundational Framework"


def build_contract_payload() -> dict[str, Any]:
    """Return the normative fail-closed replay contract."""

    return {
        "schema": CONTRACT_SCHEMA,
        "paper": PAPER0_NAME,
        "replay_schema": SCHEMA,
        "status": "normative_fail_closed_contract",
        "claim_boundary": (
            "The contract defines offline reproducibility and promotion safety only. "
            "It does not authorise hardware execution, measured-system validation, "
            "or stronger K_nm claims."
        ),
        "required_top_level_keys": [
            "schema",
            "paper",
            "status",
            "claim_boundary",
            "reproducibility",
            "inputs",
            "primary_candidate",
            "negative_control",
            "gates",
            "promotion_decision",
            "next_required_artifacts",
        ],
        "locked_inputs": {
            "primary_candidate": "data/public_application_benchmarks/eeg_alpha_plv_8ch.json",
            "negative_control": "data/public_application_benchmarks/ieee5bus_power_grid.json",
            "negative_measured_couplings": (
                "data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json"
            ),
        },
        "required_reproducibility_keys": [
            "generator",
            "comparator",
            "gate",
            "input_manifest",
            "floating_point_policy",
            "randomness_policy",
        ],
        "required_input_manifest_fields": ["path", "sha256"],
        "digest_algorithm": "sha256",
        "required_primary_candidate_keys": [
            "source_name",
            "domain",
            "matrix_shape",
            "diagnostics",
            "null_model",
            "blockers",
        ],
        "required_negative_control_keys": [
            "source_name",
            "domain",
            "matrix_shape",
            "diagnostics",
            "null_model",
            "measured_couplings",
            "interpretation",
        ],
        "required_diagnostic_keys": [
            "pearson_upper",
            "spearman_upper",
            "frobenius_relative_error",
            "density",
            "candidate_edge_count",
            "reference_edge_count",
            "shared_edge_count",
        ],
        "required_null_model_keys": [
            "seed",
            "permutations",
            "observed_pearson_upper",
            "null_mean_pearson_upper",
            "null_std_pearson_upper",
            "observed_vs_null_z",
            "two_sided_empirical_p",
        ],
        "required_gates": {
            "named_system": "pass",
            "units_and_normalisation": "blocked_primary_dimensionless_plv",
            "pairwise_uncertainty": "blocked_primary_missing_per_edge_uncertainty",
            "negative_control": "pass_non_promotional_sparse_control_remains_non_closing",
            "qpu_submission": "blocked_no_qpu_preregistration_lane",
            "claim_promotion": "blocked_measured_system_gate_open",
        },
        "required_promotion_decision": {
            "decision": "do_not_promote",
            "hardware_submission_authorised": False,
            "claim_promotion_authorised": False,
            "minimum_required_evidence_items": 4,
            "minimum_falsifier_items": 4,
        },
        "promotion_blockers": [
            "primary EEG candidate remains dimensionless PLV rather than calibrated coupling",
            "primary candidate lacks per-edge uncertainty",
            "QPU submission lane remains closed",
            "claim-promotion gate remains closed",
        ],
    }


def write_contract_payload(output_json: Path) -> dict[str, Any]:
    """Write the replay contract JSON and return the payload."""

    payload = build_contract_payload()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_replay_against_contract(
    replay_payload: dict[str, Any],
    contract_payload: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return fail-closed replay contract blockers."""

    contract = contract_payload or build_contract_payload()
    blockers: list[str] = []

    if replay_payload.get("schema") != contract["replay_schema"]:
        blockers.append("replay schema does not match the contract replay_schema")

    missing_top_level = [
        key for key in contract["required_top_level_keys"] if key not in replay_payload
    ]
    if missing_top_level:
        blockers.append(f"replay payload is missing top-level keys: {missing_top_level}")

    if replay_payload.get("status") != "blocked_non_closing_preregistered_replay":
        blockers.append("replay status must remain blocked_non_closing_preregistered_replay")

    claim_boundary = str(replay_payload.get("claim_boundary", ""))
    if "does not authorise hardware submission" not in claim_boundary:
        blockers.append("claim boundary must explicitly block hardware submission")

    reproducibility = replay_payload.get("reproducibility")
    if not isinstance(reproducibility, dict):
        blockers.append("replay reproducibility block must be present")
    else:
        missing_reproducibility = [
            key for key in contract["required_reproducibility_keys"] if key not in reproducibility
        ]
        if missing_reproducibility:
            blockers.append(
                f"replay reproducibility block is missing keys: {missing_reproducibility}"
            )
        _validate_input_manifest_contract(
            reproducibility.get("input_manifest"), contract, blockers
        )

    _validate_metric_block(
        replay_payload.get("primary_candidate"),
        required_entity_keys=contract["required_primary_candidate_keys"],
        required_diagnostic_keys=contract["required_diagnostic_keys"],
        required_null_model_keys=contract["required_null_model_keys"],
        blockers=blockers,
        label="primary_candidate",
    )
    _validate_metric_block(
        replay_payload.get("negative_control"),
        required_entity_keys=contract["required_negative_control_keys"],
        required_diagnostic_keys=contract["required_diagnostic_keys"],
        required_null_model_keys=contract["required_null_model_keys"],
        blockers=blockers,
        label="negative_control",
    )
    _validate_required_gates(replay_payload.get("gates"), contract, blockers)
    _validate_required_promotion_decision(
        replay_payload.get("promotion_decision"),
        contract,
        blockers,
    )
    return tuple(blockers)


def _validate_input_manifest_contract(
    manifest: Any,
    contract: dict[str, Any],
    blockers: list[str],
) -> None:
    """Validate manifest shape and locked input paths against the contract."""

    if not isinstance(manifest, dict):
        blockers.append("replay input_manifest must be present")
        return
    locked_inputs = contract["locked_inputs"]
    if set(manifest) != set(locked_inputs):
        blockers.append("replay input_manifest keys do not match locked contract inputs")
        return
    for name, locked_path in locked_inputs.items():
        entry = manifest.get(name)
        if not isinstance(entry, dict):
            blockers.append(f"input manifest entry {name} must be an object")
            continue
        for field in contract["required_input_manifest_fields"]:
            if field not in entry:
                blockers.append(f"input manifest entry {name} is missing {field}")
        if entry.get("path") != locked_path:
            blockers.append(f"input manifest entry {name} path does not match contract")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            blockers.append(f"input manifest entry {name} must expose a 64-character sha256")


def _validate_metric_block(
    entity: Any,
    *,
    required_entity_keys: list[str],
    required_diagnostic_keys: list[str],
    required_null_model_keys: list[str],
    blockers: list[str],
    label: str,
) -> None:
    """Validate required replay entity, diagnostic, and null-model fields."""

    if not isinstance(entity, dict):
        blockers.append(f"{label} block must be present")
        return
    missing_entity = [key for key in required_entity_keys if key not in entity]
    if missing_entity:
        blockers.append(f"{label} block is missing keys: {missing_entity}")
    diagnostics = entity.get("diagnostics")
    if not isinstance(diagnostics, dict):
        blockers.append(f"{label} diagnostics block must be present")
    else:
        missing_diagnostics = [key for key in required_diagnostic_keys if key not in diagnostics]
        if missing_diagnostics:
            blockers.append(f"{label} diagnostics block is missing keys: {missing_diagnostics}")
    null_model = entity.get("null_model")
    if not isinstance(null_model, dict):
        blockers.append(f"{label} null_model block must be present")
    else:
        missing_null = [key for key in required_null_model_keys if key not in null_model]
        if missing_null:
            blockers.append(f"{label} null_model block is missing keys: {missing_null}")


def _validate_required_gates(
    gates: Any,
    contract: dict[str, Any],
    blockers: list[str],
) -> None:
    """Validate required fail-closed gate states."""

    if not isinstance(gates, dict):
        blockers.append("replay gates block must be present")
        return
    for name, expected_state in contract["required_gates"].items():
        if gates.get(name) != expected_state:
            blockers.append(f"replay gate {name} must remain {expected_state}")


def _validate_required_promotion_decision(
    decision: Any,
    contract: dict[str, Any],
    blockers: list[str],
) -> None:
    """Validate the do-not-promote decision boundary."""

    if not isinstance(decision, dict):
        blockers.append("replay promotion_decision block must be present")
        return
    required = contract["required_promotion_decision"]
    for name in ("decision", "hardware_submission_authorised", "claim_promotion_authorised"):
        if decision.get(name) != required[name]:
            blockers.append(f"promotion decision field {name} must remain {required[name]}")
    evidence = decision.get("required_evidence_before_reconsideration")
    if (
        not isinstance(evidence, list)
        or len(evidence) < required["minimum_required_evidence_items"]
    ):
        blockers.append("promotion decision must preserve required evidence items")
    falsifiers = decision.get("falsifiers")
    if not isinstance(falsifiers, list) or len(falsifiers) < required["minimum_falsifier_items"]:
        blockers.append("promotion decision must preserve falsifier items")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--check-replay",
        type=Path,
        help="validate an existing replay JSON against the fail-closed contract",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for exporting the Paper 0 K_nm replay contract."""

    ns = _parse_args(argv)
    contract = (
        write_contract_payload(ns.output_json)
        if ns.output_json is not None
        else build_contract_payload()
    )
    if ns.check_replay is not None:
        blockers = validate_replay_against_contract(_load_json(ns.check_replay), contract)
        if blockers:
            print("paper0 K_nm replay contract valid: False")
            for blocker in blockers:
                print(f"  blocker: {blocker}")
            return 1
        print("paper0 K_nm replay contract valid: True")
        return 0
    if ns.output_json is None:
        print(json.dumps(contract, indent=2, sort_keys=True))
    else:
        print(f"paper0 K_nm replay contract exported: {ns.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
