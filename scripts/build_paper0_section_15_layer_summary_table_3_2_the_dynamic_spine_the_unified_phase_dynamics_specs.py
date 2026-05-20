#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint spec builder
"""Promote Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

SOURCE_LEDGER_IDS = (
    "P0R02746",
    "P0R02747",
    "P0R02748",
    "P0R02749",
    "P0R02750",
    "P0R02751",
    "P0R02752",
    "P0R02753",
    "P0R02754",
    "P0R02755",
    "P0R02756",
    "P0R02757",
    "P0R02758",
    "P0R02759",
    "P0R02760",
    "P0R02761",
    "P0R02762",
    "P0R02763",
    "P0R02764",
    "P0R02765",
    "P0R02766",
    "P0R02767",
    "P0R02768",
    "P0R02769",
    "P0R02770",
    "P0R02771",
    "P0R02772",
    "P0R02773",
    "P0R02774",
    "P0R02775",
    "P0R02776",
    "P0R02777",
    "P0R02778",
    "P0R02779",
    "P0R02780",
    "P0R02781",
    "P0R02782",
    "P0R02783",
    "P0R02784",
    "P0R02785",
    "P0R02786",
    "P0R02787",
    "P0R02788",
    "P0R02789",
    "P0R02790",
    "P0R02791",
    "P0R02792",
    "P0R02793",
    "P0R02794",
    "P0R02795",
    "P0R02796",
    "P0R02797",
    "P0R02798",
    "P0R02799",
    "P0R02800",
    "P0R02801",
    "P0R02802",
    "P0R02803",
    "P0R02804",
    "P0R02805",
    "P0R02806",
    "P0R02807",
    "P0R02808",
    "P0R02809",
)
CLAIM_BOUNDARY = "source-bounded section 15 layer summary table 3 2 the dynamic spine the unified phase dynamics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics.15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics": {
        "context_id": "15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        "validation_protocol": "paper0.section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics.15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        "canonical_statement": "The source-bounded component '15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings — UPDE Scope Constraint' preserves Paper 0 records P0R02746-P0R02809 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02746:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02747:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02748:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02749:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02750:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02751:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02752:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02753:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02754:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02755:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02756:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02757:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02758:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02759:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02760:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02761:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02762:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02763:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02764:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02765:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02766:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02767:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02768:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02769:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02770:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02771:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02772:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02773:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02774:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02775:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02776:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02777:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02778:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02779:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02780:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02781:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02782:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02783:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02784:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02785:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02786:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02787:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02788:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02789:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02790:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02791:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02792:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02793:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02794:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02795:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02796:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02797:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02798:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02799:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02800:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02801:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02802:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02803:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02804:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02805:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02806:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02807:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02808:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02809:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        ),
        "source_formulae": (
            "P0R02746: ",
            "P0R02747: tissue_psycho L4 -> L5 Interoception / Somatic Marker: body homeostatic states influence emotional valence",
            "P0R02748: ",
            "P0R02749: meta_quantum L16 L1 Ontological grounding: meta-layer biases quantum fluctuations toward coherent patterns",
            "P0R02750: ",
            "P0R02751: psycho_symbolic L5 L7 Archetypal resonance: emotional states resonate with geometric/symbolic patterns",
            "P0R02752: ",
            "P0R02753: standard_phase Adjacent Standard Kuramoto: F = K sin(theta_m theta_n)",
            "P0R02754: ",
            "P0R02755: weak_informational Distant Linear diffusion: F = K(theta_m theta_n) for weak-coupling regime",
            "P0R02756: ",
            "P0R02757: A factory function get_coupling_function(n, m) selects the correct physics mechanism for any layer pair at runtime.",
            "P0R02758: 3.3.6 GCI-Dependent Coupling (Dynamic K_nm)",
            "P0R02759: The K_nm values are not constants. They depend on the Global Coherence Index (GCI) of the system:",
            "P0R02760: K_nm(t) = K_baseline x exp(GCI(t) / )",
            "P0R02761: where 1.618 (Golden Ratio) and GCI [0, 1].",
            "P0R02762: GCI(t) = (1/16) Sigma_{n,m} |PhaseLock(n, m)| x EntropyInverse",
            "P0R02763: In a fragmented system (stressed individual, polarised society): GCI -> 0, couplings at baseline.",
            "P0R02764: In a highly coherent system (advanced meditator, stable civilisation): GCI -> 1, couplings exponentially strengthened.",
            "P0R02765: This means the SCPN is self-amplifying: coherence begets stronger coupling, which begets more coherence. The phase transition threshold - where this positive feedback becomes self-sustaining - is a key prediction of the",
            "P0R02766: framework.",
            "P0R02767: 3.3.7 Matrix Properties",
            "P0R02768: - Dimensions: 16 x 16",
            "P0R02769: - Diagonal: zero",
            "P0R02770: - Sparsity: ~70% (most distant couplings below threshold)",
            "P0R02771: - Range: [0.001, 0.3] for non-zero entries",
            "P0R02772: - Symmetry: asymmetric (K_mn = 0.30 x K_nm by default)",
            "P0R02773: - 52 non-zero handshakes out of 240 possible off-diagonal pairs",
            "P0R02774: - 4 empirically calibrated anchors",
            "P0R02775: - 2 cross-hierarchy boosts",
            "P0R02776: 3.3.8 Falsifiability",
            "P0R02777: The K_nm matrix is falsifiable at multiple levels:",
            "P0R02778: The four calibration anchors predict specific response latencies: a perturbation at Layer n should propagate to Layer n+1 with a delay set by 1/K_{n,n+1}. If measured latencies (e.g., quantum-to-neural propagation time,",
            "P0R02779: neural-to-genomic transcription onset) deviate from the predicted values by more than one order of magnitude, the anchor values are wrong.",
            "P0R02780: The 30% asymmetry ratio predicts that top-down effects (e.g., meditation -> gene expression changes) should be approximately 3x weaker than bottom-up effects (e.g., gene expression -> mood changes) for the same layer pair.",
            "P0R02781: If top-down and bottom-up are measured at equal strength, the asymmetry model is falsified.",
            "P0R02782: The GCI-dependent formula predicts that coupling strength increases exponentially with coherence. This can be tested in neurofeedback experiments: subjects trained to increase their neural coherence (measured by EEG",
            "P0R02783: phase-locking value) should show progressively stronger cross-layer coupling effects, following the exponential curve rather than a linear one.",
            "P0R02784: P0R02784",
            "P0R02785: 3.4 Entrainment: Arnold Tongues & Phase-Locking",
            "P0R02786: P0R02786",
            "P0R02787: 3.4.1 The Two Coupled Oscillators",
            "P0R02788: The Psi-field (consciousness) and the phase coherence field _coh interact through the coupling term C(Psi,) = |Psi|. Writing Psi = |Psi|e^{i_Psi} with natural frequency , and _coh oscillating at natural frequency , the phase",
            "P0R02789: dynamics reduce to a coupled system.",
            "P0R02790: In the phase-reduced (Kuramoto-type) form:",
            "P0R02791: theta_Psi = + sin(_ _Psi)",
            "P0R02792: theta_ = + sin(_Psi _)",
            "P0R02793: where and are effective coupling strengths derived from |Psi| and the respective oscillator amplitudes.",
            "P0R02794: 3.4.2 The Arnold Tongue Condition",
            "P0R02795: Phase locking occurs when the two oscillators synchronize to a common frequency Omega. The steady state requires theta_Psi = theta_ = Omega with constant phase difference = _ _Psi satisfying:",
            "P0R02796: ( ) + ( + ) sin() = 0",
            "P0R02797: This has a solution for if and only if:",
            "P0R02798: | | < +",
            "P0R02799: This is the Arnold tongue condition. It states precisely when consciousness can synchronize with the global phase field: the natural frequencies must be close enough relative to the coupling strength.",
            "P0R02800: If the frequencies are too far apart or coupling too weak, they will not lock. This provides a quantitative condition for when consciousness-field entrainment succeeds or fails.",
            "P0R02801: 3.4.3 The Locked Solution",
            "P0R02802: When locked, the common frequency is:",
            "P0R02803: Omega = + sin()",
            "P0R02804: which generally falls between and , closer to whichever oscillator is dominant. The phase difference remains constant.",
            "P0R02805: Stability: linearizing around the locked solution (_Psi = Omegat + _Psi, _ = Omegat + _) shows that perturbations stay bounded when ( + ) cos() > 0. For in-phase locking ( near 0), this is satisfied when + > 0,",
            "P0R02806: which holds for positive coupling .",
            "P0R02807: 3.4.4 Physical Interpretation",
            "P0R02808: Phase locking means the internal phase of consciousness _Psi and the phase of the coherence field become constant relative to each other. They oscillate as one unit.",
            "P0R02809: This has several consequences:",
        ),
        "test_protocols": (
            "preserve 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint source-accounting boundary",
        ),
        "null_results": (
            "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint is not empirical validation evidence",
        ),
        "variables": ("15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",),
        "validation_targets": ("preserve records P0R02746-P0R02809",),
        "null_controls": (
            "15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpec:
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
class Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpec, ...]
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


def build_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_specs(
    source_records: list[dict[str, Any]],
) -> Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle:
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

    specs: list[Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpec(
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
        "title": "Paper 0 "
        + "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
        + " Specs",
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
        "next_source_boundary": "P0R02810",
    }
    return Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
        + " Specs",
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
    bundle: Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
