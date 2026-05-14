#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Layer 5 Active Inference math spec builder
"""Promote Paper 0 Layer 5 Active Inference math records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6450, 6485))
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06465",
    "P0R06466",
    "P0R06469",
    "P0R06471",
    "P0R06473",
    "P0R06475",
    "P0R06476",
    "P0R06477",
    "P0R06479",
    "P0R06481",
    "P0R06482",
)

MECHANISMS_BY_SPEC = {
    "l5_active_inference_math.generative_hierarchy": (
        (
            "P0R06453:p_cosmos",
            "P0R06455:p_dimensions_cosmos",
            "P0R06459:p_self_body_world",
            "P0R06463:p_quantum_classical",
        ),
        (
            "Layer 15: p(cosmos)",
            "Layer 14: p(dimensions|cosmos)",
            "Layer 5: p(self|body,world)",
            "Layer 1: p(quantum|classical)",
        ),
        (
            "15-layer hierarchy is represented as conditional generative models",
            "Layer 5 self state is conditioned on body and world",
            "Layer 1 quantum state is conditioned on classical state",
        ),
    ),
    "l5_active_inference_math.layer_free_energy": (
        ("P0R06465:F_L_expectation", "P0R06466:F_L_kl_accuracy"),
        (
            "F_L = E_q(psi_L)[log q(psi_L) - log p(psi_L, o_L)]",
            "F_L = D_KL[q(psi_L)||p(psi_L)] - E_q[log p(o_L|psi_L)]",
        ),
        (
            "each layer carries a variational free-energy objective",
            "free energy decomposes into complexity KL and expected log-likelihood term",
        ),
    ),
    "l5_active_inference_math.message_passing_update": (
        ("P0R06469:epsilon_up", "P0R06471:epsilon_down", "P0R06473:delta_mu"),
        (
            "epsilon_L^up = partial F_L / partial mu_L = o_L - g(mu_L)",
            "epsilon_L^down = partial F_{L+1} / partial mu_L = mu_L - f(mu_{L+1})",
            "Delta mu_L = -kappa(epsilon_L^up + epsilon_L^down)",
        ),
        (
            "upward inference flow is observation minus generated prediction",
            "downward prediction flow is local state minus parent prediction",
            "belief state update descends the summed prediction errors",
        ),
    ),
    "l5_active_inference_math.action_and_precision_control": (
        (
            "P0R06475:G_pi",
            "P0R06476:G_pi_decomposition",
            "P0R06477:argmin_action",
            "P0R06479:precision_weighted_update",
            "P0R06481:precision_matrix",
            "P0R06482:prediction_error",
        ),
        (
            "G(pi) = E_q[H[p(o|s,pi)]] + E_q[D_KL[q(s|pi)||p(s|C)]]",
            "G(pi) = -information_gain + divergence_from_prior",
            "a_star = argmin_pi G(pi)",
            "Delta mu = Pi^(-1) x epsilon",
            "Pi = precision matrix = inverse covariance",
            "epsilon = prediction error",
        ),
        (
            "action selection minimises expected free energy over policies",
            "expected free energy combines ambiguity and prior divergence",
            "source text combines inverse-precision update formula with higher-precision larger-update wording",
        ),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l5_active_inference_math.generative_hierarchy": {
        "validation_protocol": "paper0.l5_active_inference_math.generative_hierarchy",
        "canonical_statement": (
            "The mathematical implementation starts from an inter-layer generative "
            "hierarchy spanning p(cosmos), p(dimensions|cosmos), p(self|body,world), "
            "and p(quantum|classical)."
        ),
        "variables": ("cosmos", "dimensions", "self", "body", "world", "quantum", "classical"),
        "validation_targets": (
            "preserve all explicit hierarchy distributions",
            "preserve Layer 5 conditioning on body and world",
            "reject missing hierarchy anchors",
        ),
        "null_controls": (
            "missing-layer-15-prior control must be rejected",
            "missing-layer-5-self-conditioning control must be rejected",
            "missing-layer-1-quantum-classical control must be rejected",
        ),
    },
    "l5_active_inference_math.layer_free_energy": {
        "validation_protocol": "paper0.l5_active_inference_math.layer_free_energy",
        "canonical_statement": (
            "Each layer receives a variational-free-energy objective with an expectation "
            "form and a KL-minus-expected-log-likelihood decomposition."
        ),
        "variables": ("q_psi_L", "p_psi_L", "o_L", "F_L"),
        "validation_targets": (
            "preserve both free-energy equations",
            "require finite probability-simplex inputs",
            "reject zero-support likelihoods",
        ),
        "null_controls": (
            "shape-mismatch control must be rejected",
            "non-positive-likelihood control must be rejected",
            "missing-KL-decomposition control must be rejected",
        ),
    },
    "l5_active_inference_math.message_passing_update": {
        "validation_protocol": "paper0.l5_active_inference_math.message_passing_update",
        "canonical_statement": (
            "Inter-layer message passing is represented by upward prediction error, "
            "downward prediction error, and a kappa-scaled belief update."
        ),
        "variables": ("epsilon_up", "epsilon_down", "mu_L", "kappa"),
        "validation_targets": (
            "preserve upward error equation",
            "preserve downward error equation",
            "preserve negative-kappa update equation",
        ),
        "null_controls": (
            "dimension-mismatch control must be rejected",
            "non-positive-kappa control must be rejected",
            "missing-downward-error control must be rejected",
        ),
    },
    "l5_active_inference_math.action_and_precision_control": {
        "validation_protocol": "paper0.l5_active_inference_math.action_and_precision_control",
        "canonical_statement": (
            "Policy selection minimises expected free energy, and prediction-error "
            "updates are source-bounded to the manuscript's inverse-precision formula "
            "while flagging its higher-precision wording as a consistency warning."
        ),
        "variables": ("G_pi", "policy", "precision_matrix", "prediction_error", "Delta_mu"),
        "validation_targets": (
            "preserve expected-free-energy action selection",
            "preserve inverse-precision update formula",
            "flag precision-wording consistency rather than rewriting the source",
        ),
        "null_controls": (
            "singular-precision control must be rejected",
            "source-precision-wording warning must be emitted",
            "missing-argmin-policy control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceMathValidationSpec:
    """Validation spec promoted from Paper 0 Active Inference math records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    equation_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceMathValidationSpecBundle:
    """Layer 5 Active Inference math validation specs plus coverage summary."""

    specs: tuple[L5ActiveInferenceMathValidationSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_l5_active_inference_math_specs(
    source_records: list[dict[str, Any]],
) -> L5ActiveInferenceMathValidationSpecBundle:
    """Build source-covered specs for the Layer 5 Active Inference math block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[L5ActiveInferenceMathValidationSpec] = []
    for key, (equation_ids, formulae, mechanisms) in MECHANISMS_BY_SPEC.items():
        metadata = SPEC_METADATA[key]
        specs.append(
            L5ActiveInferenceMathValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(equation_ids),
                source_formulae=tuple(formulae),
                source_mechanisms=tuple(mechanisms),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                equation_ledger_ids=EQUATION_SOURCE_LEDGER_IDS,
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=(
                    "source-bounded Layer 5 Active Inference mathematical fixture; "
                    "not empirical evidence"
                ),
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 Layer 5 Active Inference Math Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": "source-bounded Layer 5 Active Inference mathematical fixture; not empirical evidence",
    }
    return L5ActiveInferenceMathValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: L5ActiveInferenceMathValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Layer 5 Active Inference Math Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae:",
                *[f"- {formula}" for formula in spec.source_formulae],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: L5ActiveInferenceMathValidationSpecBundle,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write JSON and Markdown artefacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": bundle.summary,
                "specs": [asdict(spec) for spec in bundle.specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")


def main() -> int:
    """Build the default Layer 5 Active Inference math validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_active_inference_math_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_active_inference_math_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_l5_active_inference_math_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
