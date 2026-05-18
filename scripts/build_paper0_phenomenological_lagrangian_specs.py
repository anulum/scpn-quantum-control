#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 phenomenological Lagrangian spec builder
"""Promote Paper 0 phenomenological Master Interaction Lagrangian records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1333, 1384))
CLAIM_BOUNDARY = "source-bounded phenomenological Lagrangian scaffold; not derived gauge theory"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "phenomenological_lagrangian.section_opening_dual_coupling": {
        "context_id": "section_opening_dual_coupling",
        "validation_protocol": "paper0.phenomenological_lagrangian.section_opening_dual_coupling",
        "canonical_statement": (
            "The source presents the early phenomenological Master Interaction Lagrangian as a scaffold "
            "with Psi, physical-field, and interaction terms plus geometric and informational couplings."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:opening_dual_coupling" for number in range(1333, 1342)
        ),
        "source_formulae": (
            "The Phenomenological Formulation: An Evolutionary Starting Point",
            "L_total decomposes into L_Psi, L_Physical, and L_Int",
            "geometric coupling links Psi-field and spacetime curvature R",
            "informational coupling links Psi-field and det(Fisher Information Metric)",
            "stationary action delta S_Master = 0 is claimed to yield architecture under this scaffold",
            "Euler-Lagrange plus mean-field approximation is claimed to yield the UPDE",
        ),
        "test_protocols": ("preserve phenomenological scaffold opening",),
        "null_results": ("phenomenological scaffold is not the final gauge-theory derivation",),
        "variables": ("L_total", "L_Psi", "L_Physical", "L_Int", "R", "FIM", "S_Master"),
        "validation_targets": (
            "preserve three-term Lagrangian decomposition",
            "preserve dual coupling scaffold",
            "preserve stationary-action and UPDE scaffold boundary",
        ),
        "null_controls": (
            "derived gauge-theory claims must not be inferred from scaffold-only records",
        ),
    },
    "phenomenological_lagrangian.predictive_coding_free_energy": {
        "context_id": "predictive_coding_free_energy",
        "validation_protocol": "paper0.phenomenological_lagrangian.predictive_coding_free_energy",
        "canonical_statement": (
            "The source maps the early phenomenological Lagrangian onto predictive-coding/free-energy language "
            "as a first-pass heuristic cost function."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:predictive_coding_free_energy" for number in range(1343, 1348)
        ),
        "source_formulae": (
            "phenomenological Lagrangian is a first-pass heuristic cost function",
            "misalignment with spacetime geometry is one source of free energy",
            "lack of informational complexity is another source of free energy",
            "stationary action is mapped to the Free Energy Principle",
            "subsequent gauge-theoretic derivation refines this heuristic cost function",
        ),
        "test_protocols": ("preserve heuristic free-energy mapping",),
        "null_results": ("predictive-coding mapping is not an independent physical derivation",),
        "variables": ("L_Int", "free_energy", "stationary_action", "geometry", "complexity"),
        "validation_targets": (
            "preserve first-pass cost-function status",
            "preserve two heuristic free-energy sources",
            "preserve refinement-to-gauge-theory boundary",
        ),
        "null_controls": ("free-energy analogy must not be treated as measured surprise",),
    },
    "phenomenological_lagrangian.black_box_interaction": {
        "context_id": "black_box_interaction",
        "validation_protocol": "paper0.phenomenological_lagrangian.black_box_interaction",
        "canonical_statement": (
            "The source identifies the early H_int model as a black-box, less-constrained interaction "
            "with geometric and informational characters but no deeper-principle derivation."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:black_box_interaction" for number in range(1348, 1356)
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "black-box interaction with geometric and informational character",
            "geometric coupling: Psi_s, sigma as spacetime curvature R, lambda coupling constant",
            "informational coupling: Psi_s, sigma as informational volume det(gFIM), lambda coupling constant",
            "model lacks predictive power and theoretical constraint of gauge-theory-derived Lagrangian",
        ),
        "test_protocols": ("preserve black-box interaction limitations",),
        "null_results": ("black-box H_int is not a symmetry-derived interaction",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma", "R", "det_gFIM"),
        "validation_targets": (
            "preserve H_int source equation",
            "preserve geometric and informational sigma roles",
            "preserve missing-derivation limitation",
        ),
        "null_controls": ("black-box interaction must not satisfy gauge-derived classifier",),
    },
    "phenomenological_lagrangian.master_interaction_terms": {
        "context_id": "master_interaction_terms",
        "validation_protocol": "paper0.phenomenological_lagrangian.master_interaction_terms",
        "canonical_statement": (
            "The source formalises the early total and interaction Lagrangians, including geometric coupling "
            "to Ricci curvature and informational coupling to Fisher-metric volume."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:master_interaction_terms" for number in range(1356, 1372)
        ),
        "source_formulae": (
            "L_Total = L_Psi + L_Physical + L_Int",
            "L_Int = L_Geometric + L_Informational",
            "L_Geometric = (1/(2 kappa)) R + g_PsiG f(Psi) R",
            "kappa = 8 pi G / c^4",
            "L_Informational = g_PsiI Psi det(g_mu_nu(x))",
            "det(g_mu_nu) represents informational volume",
        ),
        "test_protocols": ("preserve early Master Interaction Lagrangian equations",),
        "null_results": ("formal equations remain proposed phenomenological terms",),
        "variables": (
            "L_Total",
            "L_Int",
            "L_Geometric",
            "L_Informational",
            "kappa",
            "R",
            "g_mu_nu",
        ),
        "validation_targets": (
            "preserve total Lagrangian equation",
            "preserve geometric term and kappa definition",
            "preserve informational determinant term",
        ),
        "null_controls": ("missing either coupling term must fail scaffold accounting",),
    },
    "phenomenological_lagrangian.architecture_stationary_action": {
        "context_id": "architecture_stationary_action",
        "validation_protocol": "paper0.phenomenological_lagrangian.architecture_stationary_action",
        "canonical_statement": (
            "The source links the early Lagrangian to architecture through a partition function and stationary-action "
            "principle, then states UPDE emergence under Euler-Lagrange plus mean-field approximation."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:architecture_stationary_action" for number in range(1372, 1384)
        ),
        "source_formulae": (
            "Z = integral D Psi D Phi_Physical exp(i S_Master / hbar)",
            "observed architecture corresponds to configurations that minimise the action",
            "Principle of Stationary Action: delta S_Master = 0",
            "Euler-Lagrange equations applied to L_Master with mean-field approximation derive UPDE",
            "Psi = |Psi| exp(i theta)",
            "delta S_Master / delta theta_iL = 0 implies d theta_iL / dt = omega_iL + sum K_ij sin(Delta theta) + ...",
            "phenomenological formulation is the foundational scaffold for later evolution",
        ),
        "test_protocols": ("preserve stationary-action architecture and UPDE scaffold",),
        "null_results": ("UPDE scaffold statement is not a complete derivation proof",),
        "variables": ("Z", "Psi", "Phi_Physical", "S_Master", "theta", "omega", "K_ij"),
        "validation_targets": (
            "preserve partition-function expression",
            "preserve stationary-action boundary",
            "preserve UPDE mean-field scaffold",
        ),
        "null_controls": (
            "stationary-action slogan alone must not satisfy UPDE derivation accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PhenomenologicalLagrangianSpec:
    """Phenomenological Lagrangian spec promoted from Paper 0 records."""

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
class PhenomenologicalLagrangianSpecBundle:
    """Phenomenological Lagrangian specs plus source coverage summary."""

    specs: tuple[PhenomenologicalLagrangianSpec, ...]
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


def build_phenomenological_lagrangian_specs(
    source_records: list[dict[str, Any]],
) -> PhenomenologicalLagrangianSpecBundle:
    """Build source-covered phenomenological Lagrangian specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[PhenomenologicalLagrangianSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PhenomenologicalLagrangianSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
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
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Phenomenological Lagrangian Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01384",
    }
    return PhenomenologicalLagrangianSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PhenomenologicalLagrangianSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_phenomenological_lagrangian_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PhenomenologicalLagrangianSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Phenomenological Lagrangian Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: PhenomenologicalLagrangianSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artifacts for promoted specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_phenomenological_lagrangian_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_phenomenological_lagrangian_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
