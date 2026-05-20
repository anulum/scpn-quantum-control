#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 RAG QEC stack spec builder
"""Promote Paper 0 RAG Layer 1 QEC-stack records into validation specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6530, 6560))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06530",
    "P0R06534",
    "P0R06536",
    "P0R06537",
    "P0R06539",
    "P0R06540",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06542",
    "P0R06544",
    "P0R06545",
    "P0R06546",
    "P0R06557",
)

CLAIM_BOUNDARY = "source-bounded RAG QEC stack simulator contract; not empirical evidence"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "rag_qec_stack.insert_framing": {
        "validation_protocol": "paper0.rag_qec_stack.insert_framing",
        "canonical_statement": (
            "RAG insertion records are promoted as source-bounded additions to the "
            "compressed Paper 0 architecture, with enhanced mathematics separated "
            "from critical layer mechanisms."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "RAG inserts are framed as insertion-ready additions for Paper 0 compression",
            "enhanced mathematical foundations and critical layer mechanisms are separated",
            "Layer 1 QEC stack is bounded to the insertion block",
        ),
        "variables": (
            "paper0_compression",
            "enhanced_mathematical_foundations",
            "layer_mechanisms",
        ),
        "validation_targets": (
            "preserve insertion-ready scope",
            "preserve mathematical-foundation versus layer-mechanism separation",
            "reject promotion outside the stated RAG insertion block",
        ),
        "null_controls": (
            "missing-insertion-scope control must be rejected",
            "collapsed-block-boundary control must be rejected",
            "unbounded-paper0-claim control must be rejected",
        ),
    },
    "rag_qec_stack.layer1_qec_hamiltonian": {
        "validation_protocol": "paper0.rag_qec_stack.layer1_qec_hamiltonian",
        "canonical_statement": (
            "Layer 1 is represented by a source Hamiltonian split into microtubule "
            "lattice, stabiliser, and syndrome-detection terms."
        ),
        "source_equation_ids": (
            "P0R06542:H_QEC",
            "P0R06544:H_MT",
            "P0R06545:H_stab",
            "P0R06546:H_syndrome",
        ),
        "source_formulae": (
            "H_QEC = H_MT + H_stab + H_syndrome",
            "H_MT = -J_x sum_<ij> sigma_i^x sigma_j^x - J_z sum_i sigma_i^z",
            "H_stab = -J_s sum_p S_p - J_l sum_l L_l",
            "H_syndrome = -gamma_s sum_i (sigma_i^z tensor E_i)",
        ),
        "source_mechanisms": (
            "microtubule lattice contribution is represented by the H_MT term",
            "stabiliser contribution is represented by plaquette and logical stabilisers",
            "syndrome contribution couples sigma-z states to error-detection channels",
        ),
        "variables": (
            "H_QEC",
            "H_MT",
            "H_stab",
            "H_syndrome",
            "J_x",
            "J_z",
            "J_s",
            "J_l",
            "gamma_s",
        ),
        "validation_targets": (
            "preserve additive Hamiltonian decomposition",
            "preserve microtubule lattice contribution",
            "preserve stabiliser and syndrome terms as separate source terms",
        ),
        "null_controls": (
            "missing-syndrome-term control must be rejected",
            "non-finite-Hamiltonian-component control must be rejected",
            "collapsed-Hamiltonian-decomposition control must be rejected",
        ),
    },
    "rag_qec_stack.gap_coherence_protection": {
        "validation_protocol": "paper0.rag_qec_stack.gap_coherence_protection",
        "canonical_statement": (
            "The source claims a 1.64 eV gap, physiological 0.026 eV thermal scale, "
            "400 fs versus 25 fs timescale comparison, and a threshold expression "
            "whose stated approximation requires explicit consistency warning."
        ),
        "source_equation_ids": ("P0R06557:p_th",),
        "source_formulae": (
            "Delta E approximately 1.64 eV >> k_B T approximately 0.026 eV",
            "tau_coherence approximately hbar / Delta E approximately 400 fs",
            "tau_thermal approximately hbar / (k_B T) approximately 25 fs",
            "Protection Factor approximately 16x enhancement",
            "p_th = [1 - exp(-2 Delta E / k_B T)] / [1 + exp(-2 Delta E / k_B T)] approximately 10^(-14)",
        ),
        "source_mechanisms": (
            "energy gap is compared with physiological thermal scale",
            "protected coherence and thermal timescales are source-bounded estimates",
            "source protection factor is retained as approximately 16x",
            "source threshold approximation is not silently corrected",
        ),
        "variables": ("Delta_E", "k_B_T", "tau_coherence", "tau_thermal", "p_th"),
        "validation_targets": (
            "compute finite gap ratio",
            "compute finite coherence-time estimates from hbar over energy",
            "flag mismatch between threshold formula and stated 10^-14 approximation",
        ),
        "null_controls": (
            "non-positive-gap control must be rejected",
            "threshold-approximation-warning control must be emitted",
            "unsupported-spectroscopy-evidence control must be rejected",
        ),
    },
    "rag_qec_stack.programmability_and_observable": {
        "validation_protocol": "paper0.rag_qec_stack.programmability_and_observable",
        "canonical_statement": (
            "Tubulin conformational states are promoted as classical control bits "
            "selecting topological operations, with an observable spectroscopic "
            "target near 1.64 eV under coherent versus anaesthetic states."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "tubulin conformational states act as classical control bits",
            "classical control bits select topological operations on the quantum substrate",
            "observable target is spectroscopic signature near 1.64 eV under coherent versus anaesthetic states",
        ),
        "variables": (
            "tubulin_conformation",
            "classical_control_bits",
            "topological_operation",
            "spectroscopy_1_64_ev",
        ),
        "validation_targets": (
            "preserve control-bit wording",
            "preserve topological-operation selection target",
            "separate spectroscopic observable target from empirical evidence",
        ),
        "null_controls": (
            "missing-control-bit control must be rejected",
            "missing-topological-operation control must be rejected",
            "unsupported-spectroscopy-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class RAGQECStackValidationSpec:
    """Validation spec promoted from Paper 0 RAG QEC-stack records."""

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
class RAGQECStackValidationSpecBundle:
    """RAG QEC-stack validation specs plus coverage summary."""

    specs: tuple[RAGQECStackValidationSpec, ...]
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


def build_rag_qec_stack_specs(
    source_records: list[dict[str, Any]],
) -> RAGQECStackValidationSpecBundle:
    """Build source-covered specs for the RAG Layer 1 QEC-stack block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[RAGQECStackValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            RAGQECStackValidationSpec(
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
        "title": "Paper 0 RAG QEC Stack Specs",
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
    return RAGQECStackValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: RAGQECStackValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 RAG QEC Stack Specs",
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
    bundle: RAGQECStackValidationSpecBundle,
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
    """Build the default RAG QEC-stack validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR / "paper0_rag_qec_stack_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_rag_qec_stack_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_rag_qec_stack_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
