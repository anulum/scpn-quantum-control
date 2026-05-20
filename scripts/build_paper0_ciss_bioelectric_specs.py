#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 CISS-bioelectric spec builder
"""Promote Paper 0 Layer 3 CISS-bioelectric records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6560, 6582))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06560",
    "P0R06561",
    "P0R06562",
    "P0R06563",
    "P0R06567",
    "P0R06570",
    "P0R06573",
    "P0R06577",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06565",
    "P0R06568",
    "P0R06571",
    "P0R06574",
    "P0R06576",
)

CLAIM_BOUNDARY = (
    "source-bounded CISS-bioelectric feedback simulator contract; not empirical evidence"
)

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ciss_bioelectric.layer3_framing": {
        "validation_protocol": "paper0.ciss_bioelectric.layer3_framing",
        "canonical_statement": (
            "Layer 3 is promoted as coupled CISS-bioelectric dynamics joining CISS spin "
            "filtering, radical-pair dynamics, a bioelectric cascade, and membrane-feedback coupling."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "Layer 3 is framed as a CISS-bioelectric feedback loop",
            "dual mechanism integration joins CISS spin filtering and radical-pair dynamics",
            "bioelectric cascade and coupled feedback are retained as separate mechanisms",
        ),
        "variables": ("layer3", "CISS", "bioelectric_feedback", "radical_pair"),
        "validation_targets": (
            "preserve Layer 3 framing",
            "preserve dual-mechanism integration",
            "reject collapsed CISS-only or bioelectric-only accounts",
        ),
        "null_controls": (
            "missing-CISS-channel control must be rejected",
            "missing-bioelectric-channel control must be rejected",
            "unbounded-empirical-claim control must be rejected",
        ),
    },
    "ciss_bioelectric.ciss_spin_filter": {
        "validation_protocol": "paper0.ciss_bioelectric.ciss_spin_filter",
        "canonical_statement": (
            "The CISS spin-filter term is preserved as a source Hamiltonian with epsilon, "
            "splitting, spin-orbit, and spin-coupling contributions."
        ),
        "source_equation_ids": ("P0R06565:H_total", "P0R06566:B_eff"),
        "source_formulae": (
            "H_total = epsilon_0 + (Delta/2) sigma_z + (lambda / L^2)(sigma dot L) + g S dot sigma",
            "lambda is spin-orbit coupling and generates effective B_eff in the 10-100 T source range",
        ),
        "source_mechanisms": (
            "lambda is retained as the spin-orbit coupling parameter",
            "effective B_eff is source-bounded to the 10-100 T range",
            "spin-filter Hamiltonian contributions remain additive",
        ),
        "variables": (
            "epsilon_0",
            "Delta",
            "sigma_z",
            "lambda",
            "L",
            "sigma_dot_L",
            "g",
            "S_dot_sigma",
        ),
        "validation_targets": (
            "compute finite additive spin-filter Hamiltonian",
            "reject non-positive length scale",
            "retain effective-field source range",
        ),
        "null_controls": (
            "non-positive-length control must be rejected",
            "non-finite-Hamiltonian-input control must be rejected",
            "out-of-range-effective-field control must be labelled",
        ),
    },
    "ciss_bioelectric.radical_pair_modulation": {
        "validation_protocol": "paper0.ciss_bioelectric.radical_pair_modulation",
        "canonical_statement": (
            "Radical-pair dynamics are promoted with Zeeman, hyperfine, exchange, and CISS "
            "effective-field modulation of singlet/triplet ratio."
        ),
        "source_equation_ids": ("P0R06568:H_RP",),
        "source_formulae": (
            "H_RP = sum_i [omega_i S_iz + sum_k A_ik S_i dot I_k] + J(1/2 + 2 S_1 dot S_2)",
            "singlet/triplet ratio is modulated by B_eff from CISS",
        ),
        "source_mechanisms": (
            "Zeeman terms are retained in the radical-pair Hamiltonian",
            "hyperfine terms are retained in the radical-pair Hamiltonian",
            "exchange coupling contributes through J(1/2 + 2 S_1 dot S_2)",
            "CISS effective field modulates singlet/triplet ratio",
        ),
        "variables": ("omega_i", "S_iz", "A_ik", "I_k", "J", "S_1_dot_S_2", "B_eff"),
        "validation_targets": (
            "compute finite radical-pair Hamiltonian",
            "reject non-vector Zeeman input",
            "retain positive modulation label for applied field",
        ),
        "null_controls": (
            "shape-mismatch control must be rejected",
            "non-finite-hyperfine control must be rejected",
            "zero-field-modulation control must be bounded",
        ),
    },
    "ciss_bioelectric.bioelectric_cascade_feedback": {
        "validation_protocol": "paper0.ciss_bioelectric.bioelectric_cascade_feedback",
        "canonical_statement": (
            "Bioelectric target-gradient drive, calcium/CaMKII/chromatin cascade, membrane derivative, "
            "field-dependent CISS efficiency, and local-field radical-pair feedback are preserved."
        ),
        "source_equation_ids": (
            "P0R06571:E",
            "P0R06574:dV_mem_dt",
            "P0R06576:H_RP_feedback",
        ),
        "source_formulae": (
            "E = -grad V_target -> activates Ca_v channels -> intracellular Ca2+ spike",
            "dV_mem/dt = -I_ion(V_mem, B_eff(lambda(E))) + I_pump",
            "lambda(E) is a function of local electric field E",
            "H_RP = H_RP_base + B_local(V_mem) dot (g_1 S_1 + g_2 S_2)",
        ),
        "source_mechanisms": (
            "electric field is the negative target-potential gradient",
            "calcium, CaMKII, HDAC/HAT phosphorylation, and chromatin remodelling form the cascade",
            "membrane dynamics depend on ionic current under B_eff(lambda(E)) and pump current",
            "radical-pair Hamiltonian receives a local membrane-potential field coupling",
        ),
        "variables": ("E", "V_target", "Ca_v", "CaMKII", "V_mem", "I_ion", "I_pump", "B_local"),
        "validation_targets": (
            "compute signed bioelectric cascade drive",
            "compute finite membrane derivative",
            "reject unsupported morphogenesis evidence claims",
        ),
        "null_controls": (
            "non-finite-gradient control must be rejected",
            "non-finite-current control must be rejected",
            "unsupported-morphogenesis-evidence control must be rejected",
        ),
    },
    "ciss_bioelectric.observable_predictions": {
        "validation_protocol": "paper0.ciss_bioelectric.observable_predictions",
        "canonical_statement": (
            "Observable predictions are retained as proposed validation targets: optogenetic "
            "bioelectric perturbation, chiral CISS blockade, and nonlinear radical-pair yield."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "bioelectric field perturbation by optogenetics predicts epigenetic changes",
            "CISS blockade by chiral molecular disruption predicts loss of field-guided morphogenesis",
            "radical-pair yield versus applied E-field is expected to show non-linear modulation",
        ),
        "variables": (
            "optogenetics",
            "epigenetic_change",
            "CISS_blockade",
            "radical_pair_yield",
            "E_field",
        ),
        "validation_targets": (
            "preserve optogenetic perturbation target",
            "preserve chiral blockade target",
            "preserve nonlinear radical-pair-yield target",
        ),
        "null_controls": (
            "missing-optogenetic-perturbation control must be rejected",
            "missing-CISS-blockade control must be rejected",
            "linear-only-yield control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CISSBioelectricValidationSpec:
    """Validation spec promoted from Paper 0 CISS-bioelectric records."""

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
    structural_source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class CISSBioelectricValidationSpecBundle:
    """CISS-bioelectric validation specs plus coverage summary."""

    specs: tuple[CISSBioelectricValidationSpec, ...]
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


def build_ciss_bioelectric_specs(
    source_records: list[dict[str, Any]],
) -> CISSBioelectricValidationSpecBundle:
    """Build source-covered specs for the Layer 3 CISS-bioelectric block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CISSBioelectricValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            CISSBioelectricValidationSpec(
                key=key,
                validation_protocol=str(content["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(content["canonical_statement"]),
                source_equation_ids=tuple(content["source_equation_ids"]),
                source_formulae=tuple(content["source_formulae"]),
                source_mechanisms=tuple(content["source_mechanisms"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                structural_source_ledger_ids=STRUCTURAL_SOURCE_LEDGER_IDS,
                variables=tuple(content["variables"]),
                validation_targets=tuple(content["validation_targets"]),
                executable_validation_targets=tuple(content["validation_targets"]),
                null_controls=tuple(content["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 CISS-Bioelectric Feedback Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return CISSBioelectricValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CISSBioelectricValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 CISS-Bioelectric Feedback Specs",
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
                "Mechanisms:",
                *[f"- {mechanism}" for mechanism in spec.source_mechanisms],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: CISSBioelectricValidationSpecBundle,
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
    """Build the default CISS-bioelectric validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_ciss_bioelectric_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_ciss_bioelectric_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_ciss_bioelectric_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
