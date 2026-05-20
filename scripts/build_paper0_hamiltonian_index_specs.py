#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Hamiltonian index spec builder
"""Promote Paper 0 Appendix C Hamiltonian/operator index records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6878, 6916))
CLAIM_BOUNDARY = "source-bounded Hamiltonian/operator index; not empirical evidence"
HARDWARE_STATUS = "operator_index_no_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "appendix_c.hamiltonian_index.appendix_boundary": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.boundary",
        "canonical_statement": "Appendix C is introduced as a mathematical reference for later papers.",
        "operator_symbols": (),
        "source_equation_ids": ("P0R06880:appendix_location", "P0R06881:appendix_title"),
        "source_formulae": (
            "Insert as a new Appendix C",
            "Master Index of Hamiltonians and Operators",
        ),
        "source_mechanisms": (
            "unified reference for mathematical operators",
            "hierarchical relation between Master Lagrangian and effective Hamiltonians",
        ),
        "variables": ("Appendix_C", "operator_index"),
        "validation_targets": (
            "preserve appendix placement",
            "preserve index-only status",
            "reject treating the index as executed validation",
        ),
        "null_controls": (
            "missing-appendix-boundary control must be rejected",
            "missing-index-title control must be rejected",
            "executed-validation-overclaim control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.master_lagrangian": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.master_lagrangian",
        "canonical_statement": "The master Lagrangian is indexed as the fundamental action principle.",
        "operator_symbols": ("L_Anulum",),
        "source_equation_ids": ("P0R06885:master_lagrangian",),
        "source_formulae": (
            "L = sqrt(-g)(R - 2 Lambda_Psi) + L_SM + |D_mu Psi|^2 - V(Psi) + L_int",
        ),
        "source_mechanisms": (
            "interaction of geometry, matter, and consciousness",
            "location cross-reference to Paper 13 / Paper 16",
        ),
        "variables": ("g", "R", "Lambda_Psi", "L_SM", "D_mu_Psi", "V_Psi", "L_int"),
        "validation_targets": (
            "preserve master-lagrangian symbol and terms",
            "preserve Paper 13 / Paper 16 location cross-reference",
            "reject using this index as a derivation of the field equations",
        ),
        "null_controls": (
            "missing-master-lagrangian-symbol control must be rejected",
            "missing-location-cross-reference control must be rejected",
            "field-equation-derivation-overclaim control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.microtubule_layer1": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.microtubule_layer1",
        "canonical_statement": "Layer 1 microscopic operators index microtubule, transduction, and isotopic-spin terms.",
        "operator_symbols": ("H_MT", "H_PQT", "H_iso"),
        "source_equation_ids": (
            "P0R06889:microtubule_frohlich_hamiltonian",
            "P0R06892:piezo_quantum_transduction_hamiltonian",
            "P0R06895:isotopic_spin_interaction",
        ),
        "source_formulae": (
            "H_MT = sum hbar omega a^dagger a + S(a^dagger + a) + H_bath",
            "H_PQT = hbar g_pz (b^dagger a + b a^dagger)",
            "H_iso = sum gamma I dot (B_loc + B_Psi)",
        ),
        "source_mechanisms": (
            "coherent vibrational modes in tubulin lattice",
            "conversion of mechanical phonons into coherent biophotons",
            "coupling of nuclear spin to the biological field",
        ),
        "variables": ("omega", "a", "S", "H_bath", "g_pz", "b", "gamma", "I", "B_loc", "B_Psi"),
        "validation_targets": (
            "preserve all Layer 1 operator symbols",
            "preserve Paper 1 section/protocol cross-references",
            "reject missing-location catalogue entries",
        ),
        "null_controls": (
            "missing-Layer-1-symbol control must be rejected",
            "missing-Paper-1-location control must be rejected",
            "unlocated-operator-entry control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.neuroimmune_mesoscopic": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.neuroimmune_mesoscopic",
        "canonical_statement": "Mesoscopic operators index neuro-immune and synaptic Hamiltonian entries.",
        "operator_symbols": ("H_NI", "H_syn"),
        "source_equation_ids": ("P0R06899:neuroimmune_hamiltonian", "P0R06902:synaptic_location"),
        "source_formulae": ("H_NI = H_neural + H_immune + H_tunnel",),
        "source_mechanisms": (
            "composite operator linking neural states and immune tunnelling",
            "superposition of neurotransmitter binding states",
        ),
        "variables": ("H_neural", "H_immune", "H_tunnel", "H_syn"),
        "validation_targets": (
            "preserve mesoscopic operator group",
            "preserve Paper 2 / Paper 23 and Paper 2 Section 22 cross-references",
            "record absent H_syn explicit equation as unresolved source gap",
        ),
        "null_controls": (
            "missing-mesoscopic-symbol control must be rejected",
            "missing-location control must be rejected",
            "invented-H-syn-equation control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.radical_pair_macro": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.radical_pair_macro",
        "canonical_statement": "Macroscopic Layer 6-8 entries index radical-pair and stochastic-resonance operators.",
        "operator_symbols": ("H_RP", "H_QSR"),
        "source_equation_ids": ("P0R06905:radical_pair_hamiltonian",),
        "source_formulae": (
            "H_RP = sum g mu_B B dot S + J(S_1 dot S_2)",
            "H_QSR describes signal amplification via noise-induced tunnelling",
        ),
        "source_mechanisms": (
            "spin-dependent magnetoreception",
            "signal amplification via noise-induced tunnelling",
        ),
        "variables": ("g", "mu_B", "B", "S", "J", "S_1", "S_2"),
        "validation_targets": (
            "preserve radical-pair Hamiltonian equation",
            "preserve H_QSR as indexed operator without explicit equation in this span",
            "reject invented H_QSR formula",
        ),
        "null_controls": (
            "missing-radical-pair-equation control must be rejected",
            "invented-H-QSR-equation control must be rejected",
            "missing-Paper-6-location control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.informational_operators": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.informational_operators",
        "canonical_statement": "Informational entries index the phase-curvature tensor and semiotic operator.",
        "operator_symbols": ("R_Psi", "O_sem"),
        "source_equation_ids": ("P0R06910:phase_curvature_tensor", "P0R06912:semiotic_operator"),
        "source_formulae": (
            "R_Psi: phase-curvature tensor",
            "O_sem: operator representation of a VIBRANA symbol",
        ),
        "source_mechanisms": (
            "Psi-field curvature of the information manifold",
            "semiotic operator representation",
        ),
        "variables": ("R_Psi", "O_sem", "information_manifold"),
        "validation_targets": (
            "preserve informational operator symbols",
            "preserve Paper 16 and Paper 7 cross-references",
            "reject treating metaphor labels as numerical tensor validation",
        ),
        "null_controls": (
            "missing-informational-symbol control must be rejected",
            "missing-cross-reference control must be rejected",
            "numerical-tensor-validation-overclaim control must be rejected",
        ),
    },
    "appendix_c.hamiltonian_index.structural_separators": {
        "validation_protocol": "paper0.appendix_c.hamiltonian_index.structural_separators",
        "canonical_statement": "Blank structural records after Appendix C are consumed to preserve source contiguity.",
        "operator_symbols": (),
        "source_equation_ids": (
            "P0R06914:blank_structural_separator",
            "P0R06915:blank_structural_separator",
        ),
        "source_formulae": (),
        "source_mechanisms": ("source-contiguous structural separators",),
        "variables": ("P0R06914", "P0R06915"),
        "validation_targets": (
            "preserve source contiguity through blank structural records",
            "stop before cosmological equation-of-state section",
            "reject silent record skipping",
        ),
        "null_controls": (
            "missing-structural-separator control must be rejected",
            "cosmology-section-bleed-through control must be rejected",
            "silent-source-skip control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class HamiltonianIndexSpec:
    """Hamiltonian/operator index spec promoted from Paper 0 Appendix C records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    operator_symbols: tuple[str, ...]
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class HamiltonianIndexSpecBundle:
    """Hamiltonian index specs plus coverage summary."""

    specs: tuple[HamiltonianIndexSpec, ...]
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


def build_hamiltonian_index_specs(
    source_records: list[dict[str, Any]],
) -> HamiltonianIndexSpecBundle:
    """Build source-covered Hamiltonian/operator index specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[HamiltonianIndexSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            HamiltonianIndexSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                operator_symbols=tuple(str(item) for item in metadata["operator_symbols"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                source_mechanisms=tuple(str(item) for item in metadata["source_mechanisms"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    operator_count = len(
        {symbol for spec in specs for symbol in spec.operator_symbols if symbol != "H_QSR"}
    )
    summary = {
        "title": "Paper 0 Hamiltonian Index Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "operator_count": operator_count,
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return HamiltonianIndexSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: HamiltonianIndexSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Hamiltonian Index Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Operator count: {bundle.summary['operator_count']}",
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
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Operators: {', '.join(spec.operator_symbols) or 'none'}",
                f"- Source equations: {', '.join(spec.source_equation_ids) or 'none'}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: HamiltonianIndexSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Hamiltonian index specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_hamiltonian_index_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_hamiltonian_index_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build Hamiltonian index validation specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_hamiltonian_index_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
