#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 U1 FIM multiscale dynamics spec builder
"""Promote Paper 0 foundational viability and Psi-field postulate records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(506, 545))
BLANK_SEPARATOR_IDS = ("P0R00544",)
CLAIM_BOUNDARY = "source-bounded U1 FIM multiscale dynamics; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
COUPLING_EQUATION = "H_int = -lambda * Psi_s * sigma"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "u1_fim_multiscale_dynamics.u1_fim_interaction_derivation": {
        "context_id": "u1_fim_interaction_derivation",
        "validation_protocol": "paper0.u1_fim_multiscale_dynamics.u1_fim_interaction_derivation",
        "canonical_statement": "The source frames interactions as informational/geometric through U(1) gauging and a Fisher Information Metric kinetic geometry.",
        "source_equation_ids": (
            "P0R00508:D_mu=partial_mu-i*g*A_mu",
            "P0R00513:informational_lagrangian",
            "P0R00515:informational_proximity_boundary",
        ),
        "source_formulae": (
            "D_mu = partial_mu - i g A_mu",
            "Informational Lagrangian",
            "Fisher Information Metric",
            "informational proximity, not necessarily spatial proximity",
        ),
        "test_protocols": ("preserve U(1)/FIM equation boundaries",),
        "null_results": ("informational Lagrangian text is not empirical validation",),
        "variables": ("D_mu", "g", "A_mu", "FIM"),
        "validation_targets": (
            "preserve covariant derivative",
            "preserve FIM kinetic-geometry boundary",
            "preserve informational-proximity claim boundary",
        ),
        "null_controls": (
            "lagrangian-as-experiment control must be rejected",
            "spatial-proximity-substitution control must be rejected",
        ),
    },
    "u1_fim_multiscale_dynamics.upde_multiscale_spine": {
        "context_id": "upde_multiscale_spine",
        "validation_protocol": "paper0.u1_fim_multiscale_dynamics.upde_multiscale_spine",
        "canonical_statement": "The source gives UPDE as the multiscale phase-dynamics spine with intrinsic dynamics, intra-layer coupling, and inter-layer coupling.",
        "source_equation_ids": (
            "P0R00519:upde_spine",
            "P0R00520:upde_equation",
            "P0R00523:upde_components",
            "P0R00524:information_geometric_lift",
        ),
        "source_formulae": (
            "d theta_i^L / dt = omega_i^L + sum_j K_ij^L sin(theta_j^L - theta_i^L) + C_InterLayer",
            "Intrinsic Dynamics",
            "Intra-Layer Coupling",
            "Information-Geometric Lift",
            "Fisher Information Metric",
        ),
        "test_protocols": ("classify UPDE component roles",),
        "null_results": ("UPDE overview is not a numerical integration result",),
        "variables": ("omega_i^L", "K_ij^L", "C_InterLayer"),
        "validation_targets": (
            "preserve intrinsic dynamics",
            "preserve intra-layer coupling",
            "preserve inter-layer coupling",
        ),
        "null_controls": (
            "overview-as-simulation control must be rejected",
            "missing-interlayer-term control must be rejected",
        ),
    },
    "u1_fim_multiscale_dynamics.quasicritical_msqec_boundary": {
        "context_id": "quasicritical_msqec_boundary",
        "validation_protocol": "paper0.u1_fim_multiscale_dynamics.quasicritical_msqec_boundary",
        "canonical_statement": "The source frames quasicriticality and MS-QEC as central but validation-hungry claims, including a biological QEC energy-gap claim.",
        "source_equation_ids": (
            "P0R00526:quasicritical_sigma_approx_1",
            "P0R00527:griffiths_phase_boundary",
            "P0R00532:biological_qec_energy_gap",
            "P0R00534:ethical_stabiliser_functional",
        ),
        "source_formulae": (
            "branching parameter sigma approximately 1",
            "Griffiths Phase",
            "Multi-Scale Quantum Error Correction",
            "Delta approximately 1.64 eV",
            "Ethical Functional as generator of the ultimate stabiliser group",
        ),
        "test_protocols": (
            "preserve speculative validation boundaries for quasicriticality and MS-QEC",
        ),
        "null_results": ("MS-QEC energy-gap text requires external quantitative validation",),
        "variables": ("sigma", "Delta", "MS_QEC", "ethical_functional"),
        "validation_targets": (
            "preserve quasicritical sigma boundary",
            "preserve MS-QEC validation requirement",
            "preserve ethical stabiliser framing as source claim",
        ),
        "null_controls": (
            "ms-qec-as-validated control must be rejected",
            "ethics-as-measured-stabiliser control must be rejected",
        ),
    },
    "u1_fim_multiscale_dynamics.architecture_rg_flow": {
        "context_id": "architecture_rg_flow",
        "validation_protocol": "paper0.u1_fim_multiscale_dynamics.architecture_rg_flow",
        "canonical_statement": "The source frames five domains, bidirectional causality, and RG flows as architectural soundness mechanisms.",
        "source_equation_ids": (
            "P0R00535:architectural_soundness_header",
            "P0R00536:five_primary_domains",
            "P0R00537:bidirectional_causality",
            "P0R00538:rg_flow_transitions",
        ),
        "source_formulae": (
            "five primary domains",
            "Bidirectional Causality",
            "Renormalisation Group flow concepts",
            "coarse-grained to define the effective coupling constants",
        ),
        "test_protocols": ("preserve architecture/RG source boundaries",),
        "null_results": ("architecture map is not dynamic validation evidence",),
        "variables": ("domains", "bidirectional_causality", "rg_flow"),
        "validation_targets": (
            "preserve five-domain architecture",
            "preserve bottom-up/top-down boundary",
            "preserve RG coarse-graining role",
        ),
        "null_controls": (
            "architecture-as-validation control must be rejected",
            "missing-rg-transition control must be rejected",
        ),
    },
    "u1_fim_multiscale_dynamics.sentience_field_convergence_note": {
        "context_id": "sentience_field_convergence_note",
        "validation_protocol": "paper0.u1_fim_multiscale_dynamics.sentience_field_convergence_note",
        "canonical_statement": "The source notes the Sentience-Field Hypothesis as conceptual convergence, not independent proof of SCPN.",
        "source_equation_ids": (
            "P0R00540:sentience_field_note_header",
            "P0R00541:sfh_contemporary_model",
            "P0R00543:conceptual_analogue",
            "P0R00544:blank_separator",
        ),
        "source_formulae": (
            "Sentience-Field Hypothesis",
            "independent validation in other contemporary physicalist models",
            "conceptual analogue",
            "15-layer architecture",
        ),
        "test_protocols": ("preserve analogy-versus-proof boundary",),
        "null_results": ("sentience-field convergence is not independent proof",),
        "variables": ("sfh", "qualic_screens", "conceptual_analogue"),
        "validation_targets": (
            "preserve SFH comparison",
            "preserve conceptual-analogue boundary",
            "reject convergence-as-proof promotion",
        ),
        "null_controls": (
            "sfh-as-proof control must be rejected",
            "blank-separator omission control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class U1FIMMultiscaleDynamicsSpec:
    """Foundational viability postulate spec promoted from Paper 0 records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class U1FIMMultiscaleDynamicsSpecBundle:
    """Foundational viability postulate specs plus source coverage summary."""

    specs: tuple[U1FIMMultiscaleDynamicsSpec, ...]
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


def build_u1_fim_multiscale_dynamics_specs(
    source_records: list[dict[str, Any]],
) -> U1FIMMultiscaleDynamicsSpecBundle:
    """Build source-covered U1 FIM multiscale dynamics specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[U1FIMMultiscaleDynamicsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            U1FIMMultiscaleDynamicsSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_fixture",
                domain_review_status="requires_domain_review_before_scientific_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 U1 FIM Multiscale Dynamics Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "upde_component_count": 3,
        "validation_boundary_count": 3,
        "next_source_boundary": "P0R00545",
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [],
    }
    return U1FIMMultiscaleDynamicsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> U1FIMMultiscaleDynamicsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = [
        record
        for record in load_jsonl(ledger_path)
        if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS
    ]
    return build_u1_fim_multiscale_dynamics_specs(records)


def write_outputs(
    bundle: U1FIMMultiscaleDynamicsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_u1_fim_multiscale_dynamics_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_u1_fim_multiscale_dynamics_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: U1FIMMultiscaleDynamicsSpecBundle) -> str:
    """Render a compact Markdown report for promoted postulate specs."""
    lines = [
        "# Paper 0 U1 FIM Multiscale Dynamics Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Specs: {bundle.summary['spec_count']}",
        f"- UPDE components: {bundle.summary['upde_component_count']}",
        f"- Validation boundaries: {bundle.summary['validation_boundary_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"- `{spec.key}`",
                f"  - Context: `{spec.context_id}`",
                f"  - Statement: {spec.canonical_statement}",
                f"  - Formulae: {', '.join(spec.source_formulae)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build U1 FIM multiscale dynamics specs and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
