#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 foundational strengths and phase boundary spec builder
"""Promote Paper 0 foundational-strengths and phase-boundary records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1189, 1242))
CLAIM_BOUNDARY = (
    "source-bounded foundational-strength and phase-regime claims; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "foundational_strengths_phase_boundary.foundational_strengths": {
        "context_id": "foundational_strengths",
        "validation_protocol": "paper0.foundational_strengths_phase_boundary.foundational_strengths",
        "canonical_statement": (
            "The source states that the derived interaction Lagrangian is predictive, "
            "constrained, explanatory, and falsifiable, but this remains source claim language."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:foundational_strengths" for number in range(1189, 1192)
        ),
        "source_formulae": (
            "Foundational Strengths of the SCPN Lagrangian",
            "L_Int_prime is framed as an advancement for the SCPN framework",
            "interaction terms are described as symmetry and consistency consequences",
            "arbitrary functions are replaced with Psi-star-Psi terms",
            "phenomenological constants are identified with g and xi",
            "massless spin-1 informational gauge boson remains a prediction target",
        ),
        "test_protocols": ("preserve foundational-strength claims as source-bounded targets",),
        "null_results": ("predictive wording is not experimental confirmation",),
        "variables": ("L_Int_prime", "g", "xi", "Psi", "infoton"),
        "validation_targets": (
            "preserve predictive/constrained/explanatory/falsifiable labels",
            "preserve parameter-identification claim",
            "preserve infoton prediction boundary",
        ),
        "null_controls": (
            "foundational-strength prose must not be treated as validation evidence",
        ),
    },
    "foundational_strengths_phase_boundary.architecture_integration": {
        "context_id": "architecture_integration",
        "validation_protocol": "paper0.foundational_strengths_phase_boundary.architecture_integration",
        "canonical_statement": (
            "The source links the derived interaction Lagrangian to the 15-layer SCPN "
            "architecture and cross-layer causation claims."
        ),
        "source_equation_ids": (
            "P0R01192:architecture_integration_heading",
            "P0R01193:broader_scpn_architecture_integration",
        ),
        "source_formulae": (
            "Integration with the Broader SCPN Architecture",
            "15-layer SCPN architecture is claimed to be strengthened by the interaction Lagrangian",
            "quantum-biological phase-locking and Oversoul evolution are listed as downstream dynamics",
            "top-down and bottom-up causation are framed as manifestations of source-defined laws",
        ),
        "test_protocols": ("preserve architecture-integration claim boundary",),
        "null_results": ("architecture integration is not a pipeline-wide executable proof",),
        "variables": ("SCPN", "Psi", "L_Int_prime", "top_down", "bottom_up"),
        "validation_targets": (
            "preserve 15-layer integration claim",
            "preserve cross-layer causation wording",
            "preserve source-only status",
        ),
        "null_controls": (
            "architecture claim without downstream wiring remains source accounting only",
        ),
    },
    "foundational_strengths_phase_boundary.future_research_and_parameter_constraints": {
        "context_id": "future_research_and_parameter_constraints",
        "validation_protocol": (
            "paper0.foundational_strengths_phase_boundary.future_research_and_parameter_constraints"
        ),
        "canonical_statement": (
            "The source enumerates future-research paths and constraints for g, v, "
            "infoton mass, Psi-Higgs scale, fifth-force bounds, stellar cooling, and cosmology."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:future_research_and_parameter_constraints"
            for number in range(1194, 1206)
        ),
        "source_formulae": (
            "Avenues for Future Research",
            "spontaneous symmetry breaking can give mass to the infoton via a Higgs mechanism",
            "non-Abelian extensions can imply multiple interacting informational gauge bosons",
            "geometric coupling -xi R Psi-star-Psi is linked to inflation and dark-energy questions",
            "m_A = g v for the Infoton",
            "m_h is approximately v for the Psi-Higgs",
            "unitarity bounds, RG flow, fifth-force experiments, astrophysical cooling, BBN, and CMB constrain parameter space",
            "P0R01205 marks the next modulus-phase separation heading",
        ),
        "test_protocols": ("preserve parameter-constraint and future-research queue boundaries",),
        "null_results": ("future-research list is not completed parameter inference",),
        "variables": ("g", "v", "m_A", "m_h", "xi", "N_eff"),
        "validation_targets": (
            "preserve mass formulae",
            "preserve theoretical and observational constraint classes",
            "preserve transition to modulus-phase boundary",
        ),
        "null_controls": ("unconstrained parameter values must not be promoted",),
    },
    "foundational_strengths_phase_boundary.modulus_phase_decomposition": {
        "context_id": "modulus_phase_decomposition",
        "validation_protocol": "paper0.foundational_strengths_phase_boundary.modulus_phase_decomposition",
        "canonical_statement": (
            "The source resolves the real-scalar versus complex-field appearance of Psi "
            "through modulus-phase decomposition and phase-invariant observables."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:modulus_phase_decomposition" for number in range(1206, 1224)
        ),
        "source_formulae": (
            "The Apparent Inconsistency",
            "Psi is treated as one field with real-valued gravitational and complex phase-sensitive regimes",
            "observable quantities depend on phase-invariant combinations like |Psi|^2",
            "V(Psi) = -(mu^2/2)|Psi|^2 + (lambda/4)|Psi|^4, mu^2 > 0",
            "|Psi| = v = mu/sqrt(lambda)",
            "Psi(x) = (v + h(x)) exp(i theta(x))",
            "h(x) is the radial Psi-Higgs excitation",
            "theta(x) is the hidden phase degree of freedom",
        ),
        "test_protocols": ("preserve modulus-phase decomposition equations and regime split",),
        "null_results": ("decomposition is not a measured Psi-field observation",),
        "variables": ("Psi", "mu", "lambda", "v", "h", "theta"),
        "validation_targets": (
            "preserve phase-invariant observable rule",
            "preserve Mexican-hat potential",
            "preserve radial/phase mode separation",
        ),
        "null_controls": ("treating real and complex Psi as separate fields must be rejected",),
    },
    "foundational_strengths_phase_boundary.axion_analogy_and_em_interface": {
        "context_id": "axion_analogy_and_em_interface",
        "validation_protocol": "paper0.foundational_strengths_phase_boundary.axion_analogy_and_em_interface",
        "canonical_statement": (
            "The source uses an axion/ALP analogy to frame phase-sensitive electromagnetic "
            "coupling while keeping it as a source analogy and target."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:axion_analogy_and_em_interface" for number in range(1224, 1233)
        ),
        "source_formulae": (
            "The Axion Analogy",
            "axion phase can be hidden from gravity but important in electromagnetic context",
            "theta F F_tilde is cited as the electromagnetic coupling analogy",
            "Psi phase theta can modulate electromagnetic fields or quantum oscillators",
            "ALP-mediated Psi-EM interface is referenced to Chapter 8",
            "L_a_gamma_gamma = g_a_gamma_gamma a F_mu_nu F_tilde^mu_nu = g_a_gamma_gamma a (E dot B)",
            "Primakoff effect permits ALP-photon interconversion in background magnetic fields",
        ),
        "test_protocols": ("preserve ALP analogy and electromagnetic-interface claim boundary",),
        "null_results": ("axion analogy is not an observed Psi-EM conversion",),
        "variables": ("theta", "F", "F_tilde", "a", "E", "B", "g_a_gamma_gamma"),
        "validation_targets": (
            "preserve theta F F_tilde analogy",
            "preserve ALP-mediated interface source equation",
            "preserve Primakoff-effect source context",
        ),
        "null_controls": ("analogy-only evidence must not be promoted",),
    },
    "foundational_strengths_phase_boundary.gauge_choice_and_kinetic_phase_boundary": {
        "context_id": "gauge_choice_and_kinetic_phase_boundary",
        "validation_protocol": (
            "paper0.foundational_strengths_phase_boundary.gauge_choice_and_kinetic_phase_boundary"
        ),
        "canonical_statement": (
            "The source separates gravity calculations with fixed global phase from quantum-phase "
            "interaction regimes and records the kinetic stress-energy contribution of varying phase."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:gauge_choice_and_kinetic_phase_boundary"
            for number in range(1233, 1242)
        ),
        "source_formulae": (
            "Gauge Choice and Consistency",
            "global U(1) phase is fixed when solving Einstein equations",
            "Psi phase is allowed to vary for quantum-phase interactions",
            "No fundamental conservation law is broken",
            "Kinetic Phase Contribution -- A Subtlety",
            "Psi = R exp(i theta)",
            "rho_kinetic_phase = one-half R^2 theta_dot^2",
            "gravity analysis assumes theta_dot approximately zero",
            "changing theta(x,t) is a different regime of the same field",
        ),
        "test_protocols": ("preserve gravity/phase regime boundary and kinetic phase caveat",),
        "null_results": ("phase-varying regime is not included in fixed-phase gravity analysis",),
        "variables": ("Psi", "R", "theta", "theta_dot", "rho_kinetic_phase"),
        "validation_targets": (
            "preserve fixed-phase gravity assumption",
            "preserve released-phase quantum regime",
            "preserve kinetic phase energy caveat",
        ),
        "null_controls": (
            "mixing fixed-phase and phase-varying regimes without caveat must fail",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class FoundationalStrengthsPhaseBoundarySpec:
    """Foundational-strengths and phase-boundary spec promoted from Paper 0 records."""

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
class FoundationalStrengthsPhaseBoundarySpecBundle:
    """Foundational-strengths and phase-boundary specs plus source coverage summary."""

    specs: tuple[FoundationalStrengthsPhaseBoundarySpec, ...]
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


def build_foundational_strengths_phase_boundary_specs(
    source_records: list[dict[str, Any]],
) -> FoundationalStrengthsPhaseBoundarySpecBundle:
    """Build source-covered foundational-strengths and phase-boundary specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)
    math_ids = sorted(
        {str(math_id) for record in anchors for math_id in record.get("math_ids", [])}
    )
    image_ids = sorted(
        {str(image_id) for record in anchors for image_id in record.get("image_ids", [])}
    )
    table_ids = sorted(
        {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
    )

    specs: list[FoundationalStrengthsPhaseBoundarySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FoundationalStrengthsPhaseBoundarySpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0].get("section_path", "")),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_accounting_fixture",
                domain_review_status="requires_domain_review_before_public_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Foundational Strengths Phase Boundary Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": math_ids,
        "image_ids": image_ids,
        "table_ids": table_ids,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01242",
        "spec_keys": [spec.key for spec in specs],
    }
    return FoundationalStrengthsPhaseBoundarySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> FoundationalStrengthsPhaseBoundarySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_foundational_strengths_phase_boundary_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: FoundationalStrengthsPhaseBoundarySpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Foundational Strengths Phase Boundary Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Category counts: {bundle.summary['category_counts']}",
        f"- Block-type counts: {bundle.summary['block_type_counts']}",
        f"- Math IDs: {bundle.summary['math_ids']}",
        f"- Image IDs: {bundle.summary['image_ids']}",
        f"- Table IDs: {bundle.summary['table_ids']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Promoted Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae / source labels:",
            ]
        )
        for formula in spec.source_formulae:
            lines.append(f"- {formula}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: FoundationalStrengthsPhaseBoundarySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_foundational_strengths_phase_boundary_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_foundational_strengths_phase_boundary_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
