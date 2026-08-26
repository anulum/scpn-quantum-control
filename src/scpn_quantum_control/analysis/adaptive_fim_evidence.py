# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM calibration and replay evidence
"""Digest-bound calibration controls and offline FIM custody replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from .adaptive_fim_feedback import (
    ADAPTIVE_FIM_CLAIM_BOUNDARY,
    ADAPTIVE_FIM_SCHEMA,
    AdaptiveFIMConfig,
    FIMWitness,
    adaptive_count_aware_schedule,
    plan_adaptive_fim_schedule,
)

ADAPTIVE_FIM_EVIDENCE_SCHEMA: Final[str] = "adaptive_fim_evidence.v2"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
HISTORICAL_SOURCE: Final[Path] = REPO_ROOT / (
    "data/scpn_fim_hamiltonian/"
    "fim_ibm_repeated_followup_raw_counts_2026-05-05_ibm-run-cf4835290f607387.json"
)
HISTORICAL_SOURCE_ID: Final[str] = (
    "fim_ibm_repeated_followup_raw_counts_2026-05-05_ibm-run-cf4835290f607387"
)
REPLAY_CIRCUIT_INDICES: Final[tuple[int, ...]] = (0, 3, 7)
ADAPTIVE_FIM_LITERATURE: Final[tuple[dict[str, str], ...]] = (
    {
        "authors": "C. Ferrie, C. E. Granade, and D. G. Cory",
        "title": "Adaptive Hamiltonian Estimation Using Bayesian Experimental Design",
        "year": "2011",
        "doi": "10.1063/1.3703632",
        "arxiv": "1111.0935",
        "scope": "later experiments are selected from prior outcomes in a bounded model",
    },
    {
        "authors": "I. Hincks, T. Alexander, M. Kononenko, B. Soloway, and D. G. Cory",
        "title": "Hamiltonian Learning with Online Bayesian Experiment Design in Practice",
        "year": "2018",
        "doi": "",
        "arxiv": "1806.02427",
        "scope": "online designs are compared with fixed sweeps at matched data volume",
    },
    {
        "authors": "L. D. Brown, T. T. Cai, and A. DasGupta",
        "title": "Interval Estimation for a Binomial Proportion",
        "year": "2001",
        "doi": "10.1214/ss/1009213286",
        "arxiv": "",
        "scope": "Wilson and Jeffreys intervals avoid known Wald-interval pathologies",
    },
)


def canonical_adaptive_fim_json(value: object) -> str:
    """Return stable compact JSON bytes for evidence digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of one custody artefact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Load exactly the frozen completed repeated-follow-up rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("historical source must contain a JSON object")
    if payload.get("status") != "completed" or payload.get("job_id") != "ibm-run-cf4835290f607387":
        raise ValueError("historical source custody identifiers do not match the frozen replay")
    raw_rows = payload.get("result_rows")
    if not isinstance(raw_rows, list):
        raise ValueError("historical source result_rows must be an array")
    selected: dict[int, dict[str, object]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or not isinstance(raw_row.get("metadata"), dict):
            continue
        metadata = cast(dict[str, object], raw_row["metadata"])
        index = metadata.get("circuit_index")
        if index in REPLAY_CIRCUIT_INDICES:
            if not isinstance(index, int) or index in selected:
                raise ValueError("historical replay circuit indices must be unique integers")
            if metadata.get("lambda_fim") != 4.0 or metadata.get("depth") != 2:
                raise ValueError("historical replay row no longer matches frozen lambda/depth")
            selected[index] = raw_row
    if tuple(sorted(selected)) != REPLAY_CIRCUIT_INDICES:
        raise ValueError("historical source is missing a frozen replay circuit")
    return tuple(selected[index] for index in REPLAY_CIRCUIT_INDICES)


def _witness_from_historical_row(row: Mapping[str, object]) -> FIMWitness:
    """Derive disjoint exact-state retention and magnetisation-leakage counts."""
    metadata = row.get("metadata")
    counts = row.get("counts")
    if not isinstance(metadata, dict) or not isinstance(counts, dict):
        raise ValueError("historical replay rows require metadata and counts objects")
    shots = metadata.get("shots")
    initial = metadata.get("initial_bitstring")
    popcount = metadata.get("popcount")
    circuit_index = metadata.get("circuit_index")
    if (
        not isinstance(shots, int)
        or not isinstance(initial, str)
        or not isinstance(popcount, int)
        or not isinstance(circuit_index, int)
    ):
        raise ValueError("historical replay metadata has invalid count fields")
    typed_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str) or not isinstance(count, int) or count < 0:
            raise ValueError("historical counts must map bitstrings to non-negative integers")
        typed_counts[bitstring] = count
    if sum(typed_counts.values()) != shots:
        raise ValueError("historical counts do not sum to declared shots")
    retention_events = typed_counts.get(initial[::-1], 0)
    leakage_events = sum(
        count for bitstring, count in typed_counts.items() if bitstring.count("1") != popcount
    )
    return FIMWitness.from_counts(
        leakage_events=leakage_events,
        retention_events=retention_events,
        shots=shots,
        depth=cast(int, metadata["depth"]),
        source="hardware_replay",
        artifact_id=f"{HISTORICAL_SOURCE_ID}:circuit-{circuit_index}",
    )


def historical_replay_witnesses(path: Path = HISTORICAL_SOURCE) -> tuple[FIMWitness, ...]:
    """Return three count-bound witnesses from immutable committed custody."""
    return tuple(_witness_from_historical_row(row) for row in _load_source_rows(path))


def synthetic_calibration_witnesses() -> tuple[FIMWitness, ...]:
    """Return high-signal, boundary, and underpowered calibration controls."""
    return (
        FIMWitness.from_counts(
            leakage_events=60,
            retention_events=400,
            shots=512,
            depth=2,
            source="synthetic",
            artifact_id="adaptive_fim.synthetic.high_leakage",
        ),
        FIMWitness.from_counts(
            leakage_events=25,
            retention_events=430,
            shots=512,
            depth=2,
            source="synthetic",
            artifact_id="adaptive_fim.synthetic.boundary_hold",
        ),
        FIMWitness.from_counts(
            leakage_events=3,
            retention_events=25,
            shots=32,
            depth=2,
            source="synthetic",
            artifact_id="adaptive_fim.synthetic.underpowered_hold",
        ),
    )


def _config() -> AdaptiveFIMConfig:
    """Return the frozen calibration/replay decision policy."""
    return AdaptiveFIMConfig(
        lambda_min=0.0,
        lambda_max=8.0,
        step_gain=4.0,
        max_delta_per_batch=0.5,
        target_leakage=0.05,
        deadband=0.0,
        confidence_z=1.959963984540054,
        min_shots=256,
        mode="leakage_suppression",
    )


def adaptive_fim_evidence_payload(
    source_path: Path = HISTORICAL_SOURCE,
) -> dict[str, object]:
    """Build the deterministic adaptive-FIM calibration and replay payload."""
    config = _config()
    synthetic = synthetic_calibration_witnesses()
    calibration = plan_adaptive_fim_schedule(
        4.0,
        synthetic,
        policy_id="ci_dry_run_only",
        shots_per_arm=128,
        config=config,
    )
    replay_witnesses = historical_replay_witnesses(source_path)
    replay_first = adaptive_count_aware_schedule(4.0, replay_witnesses, config)
    replay_second = adaptive_count_aware_schedule(4.0, replay_witnesses, config)
    replay_deterministic = canonical_adaptive_fim_json(
        [step.to_dict() for step in replay_first]
    ) == canonical_adaptive_fim_json([step.to_dict() for step in replay_second])
    budget_refusal = plan_adaptive_fim_schedule(
        4.0,
        synthetic,
        policy_id="ci_dry_run_only",
        shots_per_arm=4096,
        config=config,
    )
    hardware_refusal = plan_adaptive_fim_schedule(
        4.0,
        synthetic,
        policy_id="default_no_submit",
        shots_per_arm=128,
        config=config,
        request_hardware=True,
    )
    calibration_actions = [step.decision for step in calibration.steps]
    replay_actions = [step.decision for step in replay_first]
    functional_passed = (
        calibration.allowed
        and calibration_actions == ["decrease", "hold", "hold"]
        and replay_actions == ["decrease", "decrease", "decrease"]
        and replay_deterministic
        and not budget_refusal.allowed
        and not hardware_refusal.allowed
        and not budget_refusal.steps
        and not hardware_refusal.steps
    )
    payload: dict[str, object] = {
        "schema": ADAPTIVE_FIM_EVIDENCE_SCHEMA,
        "claim_boundary": ADAPTIVE_FIM_CLAIM_BOUNDARY,
        "literature": [dict(item) for item in ADAPTIVE_FIM_LITERATURE],
        "decision_policy": {
            "method": "two_sided_wilson_interval_harmful_direction_gate",
            "config": {
                "lambda_min": config.lambda_min,
                "lambda_max": config.lambda_max,
                "step_gain": config.step_gain,
                "max_delta_per_batch": config.max_delta_per_batch,
                "target_leakage": config.target_leakage,
                "deadband": config.deadband,
                "confidence_z": config.confidence_z,
                "min_shots": config.min_shots,
                "mode": config.mode,
            },
            "increase_action_available": False,
        },
        "historical_source": {
            "path": source_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(source_path),
            "job_id": "ibm-run-cf4835290f607387",
            "circuit_indices": list(REPLAY_CIRCUIT_INDICES),
            "use": "offline_proposal_replay_only",
        },
        "synthetic_calibration": calibration.to_dict(),
        "historical_offline_replay": {
            "witnesses": [witness.to_dict() for witness in replay_witnesses],
            "steps": [step.to_dict() for step in replay_first],
            "deterministic_replay": replay_deterministic,
            "closed_loop_efficacy_tested": False,
        },
        "budget_refusal": budget_refusal.to_dict(),
        "hardware_refusal": hardware_refusal.to_dict(),
        "functional_passed": functional_passed,
        "provider_submission": False,
        "hardware_execution": False,
        "closed_loop_validated": False,
        "fim_protection_claimed": False,
        "optimal_policy_claimed": False,
        "quantum_advantage_claimed": False,
    }
    digest = hashlib.sha256(canonical_adaptive_fim_json(payload).encode("utf-8")).hexdigest()
    return payload | {"content_digest": digest}


def validate_adaptive_fim_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for one adaptive-FIM evidence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    data = cast(dict[str, object], payload)
    findings: list[str] = []
    expected = {
        "schema": ADAPTIVE_FIM_EVIDENCE_SCHEMA,
        "claim_boundary": ADAPTIVE_FIM_CLAIM_BOUNDARY,
        "functional_passed": True,
        "provider_submission": False,
        "hardware_execution": False,
        "closed_loop_validated": False,
        "fim_protection_claimed": False,
        "optimal_policy_claimed": False,
        "quantum_advantage_claimed": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            findings.append(f"{key} must equal {value!r}")
    if data.get("literature") != [dict(item) for item in ADAPTIVE_FIM_LITERATURE]:
        findings.append("literature must retain the three frozen primary-source records")
    source = data.get("historical_source")
    if not isinstance(source, dict):
        findings.append("historical_source must be an object")
    else:
        if source.get("job_id") != "ibm-run-cf4835290f607387":
            findings.append("historical_source job_id must match frozen custody")
        if source.get("circuit_indices") != list(REPLAY_CIRCUIT_INDICES):
            findings.append("historical_source circuit_indices must match frozen replay")
        if source.get("use") != "offline_proposal_replay_only":
            findings.append("historical_source use must remain offline replay only")
    calibration = data.get("synthetic_calibration")
    if not _plan_contract_matches(calibration):
        findings.append("synthetic calibration plan contracts must remain exact")
    if not _actions_match(calibration, ["decrease", "hold", "hold"]):
        findings.append("synthetic calibration actions must be decrease, hold, hold")
    replay = data.get("historical_offline_replay")
    if not isinstance(replay, dict):
        findings.append("historical_offline_replay must be an object")
    else:
        steps = replay.get("steps")
        if not _step_contracts_match(steps):
            findings.append("historical replay step contracts must remain exact")
        actions = _step_actions(steps)
        if actions != ["decrease", "decrease", "decrease"]:
            findings.append("historical replay actions must be three decreases")
        if replay.get("deterministic_replay") is not True:
            findings.append("historical replay must be deterministic")
        if replay.get("closed_loop_efficacy_tested") is not False:
            findings.append("historical replay cannot claim closed-loop efficacy")
    for key in ("budget_refusal", "hardware_refusal"):
        refusal = data.get(key)
        if not _plan_contract_matches(refusal):
            findings.append(f"{key} plan contracts must remain exact")
        if not isinstance(refusal, dict) or refusal.get("allowed") is not False:
            findings.append(f"{key} must remain refused")
        elif refusal.get("steps") != [] or refusal.get("observers") != []:
            findings.append(f"{key} must emit no proposals or observers")
    digest = data.get("content_digest")
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    expected_digest = hashlib.sha256(
        canonical_adaptive_fim_json(unsigned).encode("utf-8")
    ).hexdigest()
    if digest != expected_digest:
        findings.append("content_digest does not match canonical payload")
    return tuple(findings)


def _step_actions(value: object) -> list[object] | None:
    """Return action labels from JSON-like step rows."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    return [cast(dict[str, object], item).get("decision") for item in value]


def _step_contracts_match(value: object) -> bool:
    """Return whether serialized proposal steps retain the exact contract."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return False
    return all(
        cast(dict[str, object], item).get("schema") == ADAPTIVE_FIM_SCHEMA
        and cast(dict[str, object], item).get("claim_boundary") == ADAPTIVE_FIM_CLAIM_BOUNDARY
        for item in value
    )


def _observer_contracts_match(value: object) -> bool:
    """Return whether serialized observers retain the exact claim boundary."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return False
    return all(
        cast(dict[str, object], item).get("claim_boundary") == ADAPTIVE_FIM_CLAIM_BOUNDARY
        for item in value
    )


def _plan_contract_matches(value: object) -> bool:
    """Return whether a serialized plan and its rows retain exact contracts."""
    if not isinstance(value, dict):
        return False
    plan = cast(dict[str, object], value)
    return (
        plan.get("schema") == ADAPTIVE_FIM_SCHEMA
        and plan.get("claim_boundary") == ADAPTIVE_FIM_CLAIM_BOUNDARY
        and _step_contracts_match(plan.get("steps"))
        and _observer_contracts_match(plan.get("observers"))
    )


def _actions_match(plan: object, expected: Sequence[str]) -> bool:
    """Return whether one JSON-like plan has exactly the expected actions."""
    if not isinstance(plan, dict):
        return False
    return _step_actions(plan.get("steps")) == list(expected)


def render_adaptive_fim_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render the bounded evidence summary as deterministic Markdown."""
    findings = validate_adaptive_fim_evidence(dict(payload))
    if findings:
        raise ValueError("invalid adaptive FIM evidence: " + "; ".join(findings))
    calibration = cast(dict[str, object], payload["synthetic_calibration"])
    replay = cast(dict[str, object], payload["historical_offline_replay"])
    source = cast(dict[str, object], payload["historical_source"])
    calibration_steps = cast(list[dict[str, object]], calibration["steps"])
    replay_steps = cast(list[dict[str, object]], replay["steps"])
    lines = [
        "# Adaptive FIM proposal evidence",
        "",
        f"Schema: `{payload['schema']}`",
        "",
        "## Frozen results",
        "",
        "| Lane | Actions | Interpretation |",
        "| --- | --- | --- |",
        "| Synthetic calibration | "
        + " -> ".join(str(step["decision"]) for step in calibration_steps)
        + " | High-signal decrease; boundary and underpowered controls hold |",
        "| Historical offline replay | "
        + " -> ".join(str(step["decision"]) for step in replay_steps)
        + " | Replays already committed adverse lambda=4 witnesses; no efficacy test |",
        "| Hardware-safe over-budget request | refused | No schedule or observer emitted |",
        "| Hardware request | refused | No provider submission or execution |",
        "",
        "## Custody",
        "",
        f"- Source: `{source['path']}`",
        f"- SHA256: `{source['sha256']}`",
        f"- Job ID: `{source['job_id']}`",
        "- Use: offline proposal replay only.",
        "",
        "## Claim boundary",
        "",
        str(payload["claim_boundary"]),
        "",
        "The replay shows that the rule is deterministic and conservative for the",
        "selected committed witnesses. It does not test whether later proposed",
        "batches improve leakage or retention, and therefore does not validate a",
        "closed-loop controller, FIM protection, an optimal policy, or advantage.",
        "",
        f"Content digest: `{payload['content_digest']}`",
        "",
    ]
    return "\n".join(lines)


def write_adaptive_fim_evidence(
    json_path: Path,
    markdown_path: Path,
    *,
    payload: dict[str, object] | None = None,
    source_path: Path = HISTORICAL_SOURCE,
) -> dict[str, object]:
    """Validate and atomically write adaptive-FIM JSON and Markdown evidence."""
    evidence = adaptive_fim_evidence_payload(source_path) if payload is None else payload
    findings = validate_adaptive_fim_evidence(evidence)
    if findings:
        raise ValueError("invalid adaptive FIM evidence: " + "; ".join(findings))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    markdown_temp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    json_temp.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_temp.write_text(render_adaptive_fim_evidence_markdown(evidence), encoding="utf-8")
    json_temp.replace(json_path)
    markdown_temp.replace(markdown_path)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """Write the default or caller-selected adaptive-FIM evidence artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/adaptive_fim_product/adaptive_fim_evidence.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/adaptive_fim_product/adaptive_fim_evidence.md"),
    )
    parser.add_argument("--source", type=Path, default=HISTORICAL_SOURCE)
    args = parser.parse_args(argv)
    payload = write_adaptive_fim_evidence(
        args.json_output,
        args.markdown_output,
        source_path=args.source,
    )
    print(
        f"wrote {args.json_output} and {args.markdown_output}; "
        f"functional_passed={str(payload['functional_passed']).lower()}; "
        "hardware/closed-loop/protection/advantage claims=false"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through repository runner
    raise SystemExit(main())


__all__ = [
    "ADAPTIVE_FIM_EVIDENCE_SCHEMA",
    "ADAPTIVE_FIM_LITERATURE",
    "HISTORICAL_SOURCE",
    "HISTORICAL_SOURCE_ID",
    "REPLAY_CIRCUIT_INDICES",
    "adaptive_fim_evidence_payload",
    "canonical_adaptive_fim_json",
    "historical_replay_witnesses",
    "main",
    "render_adaptive_fim_evidence_markdown",
    "synthetic_calibration_witnesses",
    "validate_adaptive_fim_evidence",
    "write_adaptive_fim_evidence",
]
