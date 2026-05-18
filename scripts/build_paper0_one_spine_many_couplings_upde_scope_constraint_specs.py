#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 One Spine, Many Couplings - UPDE Scope Constraint spec builder
"""Promote Paper 0 One Spine, Many Couplings - UPDE Scope Constraint records."""

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

SOURCE_LEDGER_IDS = (
    "P0R02682",
    "P0R02683",
    "P0R02684",
    "P0R02685",
    "P0R02686",
    "P0R02687",
    "P0R02688",
    "P0R02689",
    "P0R02690",
    "P0R02691",
    "P0R02692",
    "P0R02693",
    "P0R02694",
    "P0R02695",
    "P0R02696",
    "P0R02697",
    "P0R02698",
    "P0R02699",
    "P0R02700",
    "P0R02701",
    "P0R02702",
    "P0R02703",
    "P0R02704",
    "P0R02705",
    "P0R02706",
    "P0R02707",
    "P0R02708",
    "P0R02709",
    "P0R02710",
    "P0R02711",
    "P0R02712",
    "P0R02713",
    "P0R02714",
    "P0R02715",
    "P0R02716",
    "P0R02717",
    "P0R02718",
    "P0R02719",
    "P0R02720",
    "P0R02721",
    "P0R02722",
    "P0R02723",
    "P0R02724",
    "P0R02725",
    "P0R02726",
    "P0R02727",
    "P0R02728",
    "P0R02729",
    "P0R02730",
    "P0R02731",
    "P0R02732",
    "P0R02733",
    "P0R02734",
    "P0R02735",
    "P0R02736",
    "P0R02737",
    "P0R02738",
    "P0R02739",
    "P0R02740",
    "P0R02741",
    "P0R02742",
    "P0R02743",
    "P0R02744",
    "P0R02745",
)
CLAIM_BOUNDARY = "source-bounded one spine many couplings upde scope constraint source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "one_spine_many_couplings_upde_scope_constraint.one_spine_many_couplings_upde_scope_constraint": {
        "context_id": "one_spine_many_couplings_upde_scope_constraint",
        "validation_protocol": "paper0.one_spine_many_couplings_upde_scope_constraint.one_spine_many_couplings_upde_scope_constraint",
        "canonical_statement": "The source-bounded component 'One Spine, Many Couplings - UPDE Scope Constraint' preserves Paper 0 records P0R02682-P0R02745 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02682:one_spine_many_couplings_upde_scope_constraint",
            "P0R02683:one_spine_many_couplings_upde_scope_constraint",
            "P0R02684:one_spine_many_couplings_upde_scope_constraint",
            "P0R02685:one_spine_many_couplings_upde_scope_constraint",
            "P0R02686:one_spine_many_couplings_upde_scope_constraint",
            "P0R02687:one_spine_many_couplings_upde_scope_constraint",
            "P0R02688:one_spine_many_couplings_upde_scope_constraint",
            "P0R02689:one_spine_many_couplings_upde_scope_constraint",
            "P0R02690:one_spine_many_couplings_upde_scope_constraint",
            "P0R02691:one_spine_many_couplings_upde_scope_constraint",
            "P0R02692:one_spine_many_couplings_upde_scope_constraint",
            "P0R02693:one_spine_many_couplings_upde_scope_constraint",
            "P0R02694:one_spine_many_couplings_upde_scope_constraint",
            "P0R02695:one_spine_many_couplings_upde_scope_constraint",
            "P0R02696:one_spine_many_couplings_upde_scope_constraint",
            "P0R02697:one_spine_many_couplings_upde_scope_constraint",
            "P0R02698:one_spine_many_couplings_upde_scope_constraint",
            "P0R02699:one_spine_many_couplings_upde_scope_constraint",
            "P0R02700:one_spine_many_couplings_upde_scope_constraint",
            "P0R02701:one_spine_many_couplings_upde_scope_constraint",
            "P0R02702:one_spine_many_couplings_upde_scope_constraint",
            "P0R02703:one_spine_many_couplings_upde_scope_constraint",
            "P0R02704:one_spine_many_couplings_upde_scope_constraint",
            "P0R02705:one_spine_many_couplings_upde_scope_constraint",
            "P0R02706:one_spine_many_couplings_upde_scope_constraint",
            "P0R02707:one_spine_many_couplings_upde_scope_constraint",
            "P0R02708:one_spine_many_couplings_upde_scope_constraint",
            "P0R02709:one_spine_many_couplings_upde_scope_constraint",
            "P0R02710:one_spine_many_couplings_upde_scope_constraint",
            "P0R02711:one_spine_many_couplings_upde_scope_constraint",
            "P0R02712:one_spine_many_couplings_upde_scope_constraint",
            "P0R02713:one_spine_many_couplings_upde_scope_constraint",
            "P0R02714:one_spine_many_couplings_upde_scope_constraint",
            "P0R02715:one_spine_many_couplings_upde_scope_constraint",
            "P0R02716:one_spine_many_couplings_upde_scope_constraint",
            "P0R02717:one_spine_many_couplings_upde_scope_constraint",
            "P0R02718:one_spine_many_couplings_upde_scope_constraint",
            "P0R02719:one_spine_many_couplings_upde_scope_constraint",
            "P0R02720:one_spine_many_couplings_upde_scope_constraint",
            "P0R02721:one_spine_many_couplings_upde_scope_constraint",
            "P0R02722:one_spine_many_couplings_upde_scope_constraint",
            "P0R02723:one_spine_many_couplings_upde_scope_constraint",
            "P0R02724:one_spine_many_couplings_upde_scope_constraint",
            "P0R02725:one_spine_many_couplings_upde_scope_constraint",
            "P0R02726:one_spine_many_couplings_upde_scope_constraint",
            "P0R02727:one_spine_many_couplings_upde_scope_constraint",
            "P0R02728:one_spine_many_couplings_upde_scope_constraint",
            "P0R02729:one_spine_many_couplings_upde_scope_constraint",
            "P0R02730:one_spine_many_couplings_upde_scope_constraint",
            "P0R02731:one_spine_many_couplings_upde_scope_constraint",
            "P0R02732:one_spine_many_couplings_upde_scope_constraint",
            "P0R02733:one_spine_many_couplings_upde_scope_constraint",
            "P0R02734:one_spine_many_couplings_upde_scope_constraint",
            "P0R02735:one_spine_many_couplings_upde_scope_constraint",
            "P0R02736:one_spine_many_couplings_upde_scope_constraint",
            "P0R02737:one_spine_many_couplings_upde_scope_constraint",
            "P0R02738:one_spine_many_couplings_upde_scope_constraint",
            "P0R02739:one_spine_many_couplings_upde_scope_constraint",
            "P0R02740:one_spine_many_couplings_upde_scope_constraint",
            "P0R02741:one_spine_many_couplings_upde_scope_constraint",
            "P0R02742:one_spine_many_couplings_upde_scope_constraint",
            "P0R02743:one_spine_many_couplings_upde_scope_constraint",
            "P0R02744:one_spine_many_couplings_upde_scope_constraint",
            "P0R02745:one_spine_many_couplings_upde_scope_constraint",
        ),
        "source_formulae": (
            "P0R02682: One Spine, Many Couplings - UPDE Scope Constraint",
            "P0R02683: [TABLE]",
            "P0R02684: Scope note: The UPDE is universal only at the phase-reduction level (weakly coupled, noisy, limit-cycle class). Each layer instantiates its own K, noise law, and invariants. Identical phase equations do not imply identical microphysics.",
            "P0R02685: P0R02685",
            "P0R02686: Beyond Kuramoto: The Need for a Generalised Oscillator Model",
            "P0R02687: Source Material: The justification for extending the Kuramoto model to account for the complex, multi-scale, and field-coupled nature of the SCPN.",
            "P0R02688: P0R02688",
            "P0R02689: Formal Derivation of the UPDE",
            "P0R02690: Source Material: The complete mathematical presentation of the Unified Phase Dynamics Equation, meticulously defining each term: intrinsic dynamics (), intra-layer coupling (K), inter-layer coupling (), Psi-field coupling (lambda), and stochastic forces ().",
            "P0R02691: P0R02691",
            'P0R02692: The "Information-Geometric Lift": Dynamics as Gradient Flow',
            "P0R02693: Source Material: The advanced section that re-frames the UPDE, demonstrating that its dynamics represent a gradient flow on the statistical manifold defined by the Fisher Information Metric, thereby unifying the system's dynamics with its informational geometry.",
            "P0R02694: P0R02694",
            'P0R02695: The "One Spine, Many Couplings" Protocol',
            "P0R02696: Source Material: The powerful summary table and explanation showing how the universal form of the UPDE is adapted to each specific layer of the SCPN simply by changing the nature and strength of the coupling terms.",
            "P0R02697: P0R02697",
            "P0R02698: P0R02698",
            "P0R02699: P0R02699",
            "P0R02700: 3.3.1 The K_nm Inter-Layer Coupling Matrix - From Architecture to Numbers",
            "P0R02701: 3.3.1.1 Structure",
            "P0R02702: The K_nm matrix is a 16x16 real matrix defining the coupling strength between every pair of SCPN layers. It encodes 52 specific informational handshakes - directed causal pathways between layers. The diagonal is zero (no",
            "P0R02703: self-coupling). The matrix is asymmetric: bottom-up flow is stronger than top-down.",
            "P0R02704: 3.3.1.2 The UPDE with K_nm",
            "P0R02705: The master equation governing all inter-layer dynamics is:",
            "P0R02706: dtheta_n/dt = _n + Sigma_m [K_nm sin(theta_m theta_n)] + _n(t)",
            "P0R02707: where theta_n is the phase state of Layer n, _n is the natural frequency, K_nm is the coupling coefficient, and _n(t) is stochastic noise.",
            "P0R02708: 3.3.1.3 Construction Algorithm (Three-Pass)",
            "P0R02709: Pass 1 - Adjacent layers (|nm| = 1):",
            "P0R02710: K_adjacent = K_base / (1 + |ln(tau_n / tau_m)|), = 0.05",
            "P0R02711: where tau_n is the characteristic timescale of Layer n. The penalty term |ln(tau_n/tau_m)| suppresses coupling between layers with vastly different timescales. Values clipped to [0.1, 0.5]. Four calibration anchors override",
            "P0R02712: computed values for the lowest adjacent pairs.",
            "P0R02713: Pass 2 - Near-neighbour (|nm| = 2):",
            "P0R02714: K_neighbor = (K[n, mid] x K[mid, m]) / (1 + Delta/_avg)",
            "P0R02715: where mid = (n+m)/2 is the intermediate layer. The geometric mean of the two adjacent couplings, penalised by frequency mismatch. Clipped to [0.01, 0.4].",
            "P0R02716: Pass 3 - Distant (|nm| 3):",
            "P0R02717: K_distant = K_base x exp( |n m|)",
            "P0R02718: where K_base = 0.45 and = 0.3 (decay rate). Clipped to [0.001, 0.2]. Cross-hierarchy boosts override where applicable.",
            "P0R02719: 3.3.2 Calibration Anchors (Empirically Constrained)",
            "P0R02720: K[L1, L2] = 0.302 Quantum-Neural coupling",
            "P0R02721: K[L2, L3] = 0.201 Neural-Genomic coupling",
            "P0R02722: K[L3, L4] = 0.252 Genomic-Tissue coupling",
            "P0R02723: K[L4, L5] = 0.154 Tissue-Psychoemotional coupling",
            "P0R02724: These four values are the empirical ground truth of the matrix. They are not derived from the construction algorithm - they constrain it. Sources: Frhlich condensation timescales (L1->L2), activity-dependent transcription",
            "P0R02725: rates (L2->L3), bioelectric morphogenesis data (L3->L4), interoceptive coupling measurements (L4->L5).",
            "P0R02726: 3.3.3 Cross-Hierarchy Boosts",
            "P0R02727: K[L1, L16] = 0.05 Quantum-Meta feedback loop (ontological grounding)",
            "P0R02728: K[L5, L7] = 0.15 Intent-Geometry bridge (archetypal resonance)",
            "P0R02729: These non-adjacent couplings represent long-range informational pathways that bypass the layer hierarchy. They are essential for cybernetic closure (L1L16) and for the symbolic-emotional coupling that drives intentional",
            "P0R02730: action (L5L7).",
            "P0R02731: 3.3.4 Reciprocal Coupling (Asymmetry)",
            "P0R02732: For every non-zero K_nm, the reverse coupling is:",
            "P0R02733: K_mn = 0.30 x K_nm (if K_mn was not independently specified)",
            "P0R02734: This 30% asymmetry reflects the thermodynamic arrow: bottom-up information flow (sensory, biological) is stronger than top-down causal influence (intentional, teleological). The asymmetry ratio 0.30 is a free parameter",
            "P0R02735: that could be refined by experiment.",
            "P0R02736: 3.3.5 Per-Coupling Physics Functions",
            "P0R02737: The static K_nm values are mapped to dynamic state-interaction functions encoding the underlying physics mechanism for each layer pair:",
            "P0R02738: ",
            "P0R02739: Coupling Layers Physics Function",
            "P0R02740: ",
            "P0R02741: quantum_neural L1 -> L2 F = K sin(theta_m theta_n)(1 + noise_SR) - Quantum-to-Statistical Reduction with stochastic resonance",
            "P0R02742: ",
            "P0R02743: neural_genomic L2 -> L3 F = K clip(sin(theta_n), 0, 1) - Activity-dependent transcription via calcium signalling (rectified: silence does not drive)",
            "P0R02744: ",
            "P0R02745: genomic_tissue L3 -> L4 F = K sin(theta_m theta_n) - Standard Kuramoto (Turing morphogenesis)",
        ),
        "test_protocols": (
            "preserve One Spine, Many Couplings - UPDE Scope Constraint source-accounting boundary",
        ),
        "null_results": (
            "One Spine, Many Couplings - UPDE Scope Constraint is not empirical validation evidence",
        ),
        "variables": ("one_spine_many_couplings_upde_scope_constraint",),
        "validation_targets": ("preserve records P0R02682-P0R02745",),
        "null_controls": (
            "one_spine_many_couplings_upde_scope_constraint must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class OneSpineManyCouplingsUpdeScopeConstraintSpec:
    """Spec promoted from Paper 0 source records."""

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
class OneSpineManyCouplingsUpdeScopeConstraintSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[OneSpineManyCouplingsUpdeScopeConstraintSpec, ...]
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


def build_one_spine_many_couplings_upde_scope_constraint_specs(
    source_records: list[dict[str, Any]],
) -> OneSpineManyCouplingsUpdeScopeConstraintSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[OneSpineManyCouplingsUpdeScopeConstraintSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            OneSpineManyCouplingsUpdeScopeConstraintSpec(
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
        "title": "Paper 0 " + "One Spine, Many Couplings - UPDE Scope Constraint" + " Specs",
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
        "next_source_boundary": "P0R02746",
    }
    return OneSpineManyCouplingsUpdeScopeConstraintSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> OneSpineManyCouplingsUpdeScopeConstraintSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_one_spine_many_couplings_upde_scope_constraint_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: OneSpineManyCouplingsUpdeScopeConstraintSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "One Spine, Many Couplings - UPDE Scope Constraint" + " Specs",
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
    return "\n".join(lines)


def write_outputs(
    bundle: OneSpineManyCouplingsUpdeScopeConstraintSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_one_spine_many_couplings_upde_scope_constraint_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_one_spine_many_couplings_upde_scope_constraint_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
