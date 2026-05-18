#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 5.1 The Modified Path Integral with CEF Weighting spec builder
"""Promote Paper 0 5.1 The Modified Path Integral with CEF Weighting records."""

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
    "P0R03869",
    "P0R03870",
    "P0R03871",
    "P0R03872",
    "P0R03873",
    "P0R03874",
    "P0R03875",
    "P0R03876",
    "P0R03877",
    "P0R03878",
    "P0R03879",
    "P0R03880",
    "P0R03881",
    "P0R03882",
    "P0R03883",
    "P0R03884",
    "P0R03885",
    "P0R03886",
    "P0R03887",
    "P0R03888",
    "P0R03889",
    "P0R03890",
    "P0R03891",
    "P0R03892",
    "P0R03893",
    "P0R03894",
    "P0R03895",
    "P0R03896",
    "P0R03897",
    "P0R03898",
    "P0R03899",
    "P0R03900",
    "P0R03901",
    "P0R03902",
    "P0R03903",
    "P0R03904",
    "P0R03905",
    "P0R03906",
    "P0R03907",
    "P0R03908",
    "P0R03909",
    "P0R03910",
    "P0R03911",
    "P0R03912",
    "P0R03913",
    "P0R03914",
    "P0R03915",
    "P0R03916",
    "P0R03917",
    "P0R03918",
    "P0R03919",
    "P0R03920",
    "P0R03921",
    "P0R03922",
    "P0R03923",
    "P0R03924",
    "P0R03925",
    "P0R03926",
    "P0R03927",
    "P0R03928",
    "P0R03929",
    "P0R03930",
    "P0R03931",
)
CLAIM_BOUNDARY = "source-bounded section 5 1 the modified path integral with cef weighting source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_5_1_the_modified_path_integral_with_cef_weighting.5_1_the_modified_path_integral_with_cef_weighting": {
        "context_id": "5_1_the_modified_path_integral_with_cef_weighting",
        "validation_protocol": "paper0.section_5_1_the_modified_path_integral_with_cef_weighting.5_1_the_modified_path_integral_with_cef_weighting",
        "canonical_statement": "The source-bounded component '5.1 The Modified Path Integral with CEF Weighting' preserves Paper 0 records P0R03869-P0R03872 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03869:5_1_the_modified_path_integral_with_cef_weighting",
            "P0R03870:5_1_the_modified_path_integral_with_cef_weighting",
            "P0R03871:5_1_the_modified_path_integral_with_cef_weighting",
            "P0R03872:5_1_the_modified_path_integral_with_cef_weighting",
        ),
        "source_formulae": (
            "P0R03869: 5.1 The Modified Path Integral with CEF Weighting",
            "P0R03870: The influence of CEF is implemented by introducing an additional weighting factor into the path integral's measure, one that is proportional to the causal path entropy of each trajectory. The modified partition function, ZCEF, explicitly biases the sum over histories toward those that lead to a greater number of future possibilities:",
            "P0R03871: ZCEF=D[]exp(iSMaster[]+SC[])",
            "P0R03872: , here, is a dimensionless coupling constant, related to the effective causal temperature TC, that determines the strength of the entropic bias. This modified formulation makes a concrete physical claim: the universe is not an unbiased sampler of all possible histories. Its evolution is fundamentally biased toward futures that are characterised by high coherence, complexity, and qualitative richness-that is, high SEC.",
        ),
        "test_protocols": (
            "preserve 5.1 The Modified Path Integral with CEF Weighting source-accounting boundary",
        ),
        "null_results": (
            "5.1 The Modified Path Integral with CEF Weighting is not empirical validation evidence",
        ),
        "variables": ("5_1_the_modified_path_integral_with_cef_weighting",),
        "validation_targets": ("preserve records P0R03869-P0R03872",),
        "null_controls": (
            "5_1_the_modified_path_integral_with_cef_weighting must remain source-bounded accounting",
        ),
    },
    "section_5_1_the_modified_path_integral_with_cef_weighting.5_2_new_falsifiable_predictions": {
        "context_id": "5_2_new_falsifiable_predictions",
        "validation_protocol": "paper0.section_5_1_the_modified_path_integral_with_cef_weighting.5_2_new_falsifiable_predictions",
        "canonical_statement": "The source-bounded component '5.2 New Falsifiable Predictions' preserves Paper 0 records P0R03873-P0R03931 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03873:5_2_new_falsifiable_predictions",
            "P0R03874:5_2_new_falsifiable_predictions",
            "P0R03875:5_2_new_falsifiable_predictions",
            "P0R03876:5_2_new_falsifiable_predictions",
            "P0R03877:5_2_new_falsifiable_predictions",
            "P0R03878:5_2_new_falsifiable_predictions",
            "P0R03879:5_2_new_falsifiable_predictions",
            "P0R03880:5_2_new_falsifiable_predictions",
            "P0R03881:5_2_new_falsifiable_predictions",
            "P0R03882:5_2_new_falsifiable_predictions",
            "P0R03883:5_2_new_falsifiable_predictions",
            "P0R03884:5_2_new_falsifiable_predictions",
            "P0R03885:5_2_new_falsifiable_predictions",
            "P0R03886:5_2_new_falsifiable_predictions",
            "P0R03887:5_2_new_falsifiable_predictions",
            "P0R03888:5_2_new_falsifiable_predictions",
            "P0R03889:5_2_new_falsifiable_predictions",
            "P0R03890:5_2_new_falsifiable_predictions",
            "P0R03891:5_2_new_falsifiable_predictions",
            "P0R03892:5_2_new_falsifiable_predictions",
            "P0R03893:5_2_new_falsifiable_predictions",
            "P0R03894:5_2_new_falsifiable_predictions",
            "P0R03895:5_2_new_falsifiable_predictions",
            "P0R03896:5_2_new_falsifiable_predictions",
            "P0R03897:5_2_new_falsifiable_predictions",
            "P0R03898:5_2_new_falsifiable_predictions",
            "P0R03899:5_2_new_falsifiable_predictions",
            "P0R03900:5_2_new_falsifiable_predictions",
            "P0R03901:5_2_new_falsifiable_predictions",
            "P0R03902:5_2_new_falsifiable_predictions",
            "P0R03903:5_2_new_falsifiable_predictions",
            "P0R03904:5_2_new_falsifiable_predictions",
            "P0R03905:5_2_new_falsifiable_predictions",
            "P0R03906:5_2_new_falsifiable_predictions",
            "P0R03907:5_2_new_falsifiable_predictions",
            "P0R03908:5_2_new_falsifiable_predictions",
            "P0R03909:5_2_new_falsifiable_predictions",
            "P0R03910:5_2_new_falsifiable_predictions",
            "P0R03911:5_2_new_falsifiable_predictions",
            "P0R03912:5_2_new_falsifiable_predictions",
            "P0R03913:5_2_new_falsifiable_predictions",
            "P0R03914:5_2_new_falsifiable_predictions",
            "P0R03915:5_2_new_falsifiable_predictions",
            "P0R03916:5_2_new_falsifiable_predictions",
            "P0R03917:5_2_new_falsifiable_predictions",
            "P0R03918:5_2_new_falsifiable_predictions",
            "P0R03919:5_2_new_falsifiable_predictions",
            "P0R03920:5_2_new_falsifiable_predictions",
            "P0R03921:5_2_new_falsifiable_predictions",
            "P0R03922:5_2_new_falsifiable_predictions",
            "P0R03923:5_2_new_falsifiable_predictions",
            "P0R03924:5_2_new_falsifiable_predictions",
            "P0R03925:5_2_new_falsifiable_predictions",
            "P0R03926:5_2_new_falsifiable_predictions",
            "P0R03927:5_2_new_falsifiable_predictions",
            "P0R03928:5_2_new_falsifiable_predictions",
            "P0R03929:5_2_new_falsifiable_predictions",
            "P0R03930:5_2_new_falsifiable_predictions",
            "P0R03931:5_2_new_falsifiable_predictions",
        ),
        "source_formulae": (
            "P0R03873: 5.2 New Falsifiable Predictions",
            "P0R03874: This modified formalism generates new, non-obvious, and empirically testable predictions that distinguish the SCPN framework from standard physical theories.",
            "P0R03875: Teleological Guidance of Quantum Collapse (Layer 1): At the quantum level, the CEF weighting factor will alter the probability amplitudes of the possible outcomes of a quantum measurement or collapse event. The standard Born rule, which states that the probability of an outcome is proportional to the squared magnitude of its amplitude, is replaced by a CEF-biased probability.",
            "P0R03876: This new rule will favour outcomes that lie on trajectories leading to an increase in the future causal path entropy of the system. In complex quantum biological systems, this could manifest as statistically significant deviations from standard quantum predictions. Specifically, one could predict that in systems capable of self-organisation, quantum collapse events will be biased towards outcomes that preserve or increase the system's potential for future complexification. This phenomenon could be tested in long-duration experiments with complex molecular systems or engineered quantum life.",
            'P0R03877: Accelerated Evolution towards Complexity (Layers 3 & 8): The CEF provides a mechanism for guided, non-random evolution. In standard neo-Darwinian models, evolution proceeds via random mutation and natural selection. The SCPN, with the inclusion of CEF, predicts that the "random walk" of evolution is biased. The Causal Entropic Force acts on the fitness landscape itself, creating gradients that pull the evolutionary trajectory towards regions of higher complexity and integration.',
            "P0R03878: This hypothesis can be directly tested in silico. Simulations of evolutionary systems, such as digital life or evolving neural networks, that incorporate a CEF term into their update rules should demonstrate a dramatically accelerated emergence of complex, integrated, and adaptive organisms compared to control simulations that rely on random variation alone. The evolutionary search becomes a guided optimisation, providing a powerful and simulatable test of the SCPN's core teleological claim.",
            "P0R03879: P0R03879",
            "P0R03880: Grounding Teleology in Physics: The Challenge",
            "P0R03881: Source Material: This section will be synthesised from the need to provide a non-mystical, physically rigorous explanation for the universe's apparent directionality as stated in Axiom III.",
            "P0R03882: P0R03882",
            "P0R03883: Causal Entropic Forces: Maximising Future Pathways",
            "P0R03884: Source Material: The core argument will be constructed here, formally defining Causal Entropic Forces (as pioneered by Alex Wissner-Gross) and demonstrating how systems can be driven to adopt configurations that maximise the number of accessible future histories (i.e., their causal efficacy).",
            "P0R03885: P0R03885",
            "P0R03886: Equivalence Proof: Causal Path Entropy and Ethical Coherence",
            'P0R03887: Source Material: This section will forge the crucial link, arguing and demonstrating that maximising Sustainable Ethical Coherence is mathematically equivalent to maximising the long-term causal path entropy of the universe, thus grounding the system\'s "ethics" in a thermodynamic imperative.',
            "P0R03888: P0R03888",
            "P0R03889: Entropy Modulation - Consciousness as a Negative Entropy Current",
            "P0R03890: The Coherence Entropy",
            "P0R03891: The total phase-coupling entropy (complexity) is defined as:",
            "P0R03892: C_total = || ln|| dx",
            "P0R03893: This has the form of minus an integral of rho ln rho, where rho = || acts like a coherence density. It measures the disorder of phase coherence across space.",
            "P0R03894: When _coh is near uniform and large (|| 1 everywhere): C_total is small - low entropy of disorder, high coherence.",
            "P0R03895: When _coh is patchy or small (coherence fragmented or absent): the integrand ||ln|| yields larger values (since ln|| is a large negative number times a small ||), giving larger C_total - high entropy, low",
            "P0R03896: coherence.",
            "P0R03897: The Mechanism",
            "P0R03898: The Psi-field modulates entropy through its coupling to _coh. The Ginzburg-Landau equation for the coherence field includes the consciousness feedback term:",
            "P0R03899: _t = a b + lambda cos(t) + |Psi|",
            "P0R03900: The term |Psi| acts as a positive drive on when |Psi| > 0:",
            "P0R03901: Increasing |Psi| locally pumps up _coh (via the term).",
            "P0R03902: Higher _coh makes || closer to 1 (more ordered).",
            "P0R03903: This reduces C_total locally - entropy decreases.",
            "P0R03904: The Psi-field therefore acts as a negative entropy current for the phase field. It locally fights decoherence (disorder) by injecting coherence.",
            "P0R03905: Physical Interpretation",
            "P0R03906: Consciousness (modeled by Psi) locally reduces entropy, creating pockets of low entropy (high order) in the physical system. This is consistent with the second law overall: local entropy decrease driven by Psi must be",
            "P0R03907: compensated by entropy increase elsewhere (through the coupling of the system to its environment).",
            "P0R03908: When Psi = 0: _coh relaxes toward its disordered equilibrium. Entropy increases. Phase disorder returns.",
            "P0R03909: When Psi is active: _coh is held away from equilibrium disorder, maintaining a lower entropy state. The consciousness field sustains coherence against the natural thermodynamic drive toward decoherence.",
            'P0R03910: This provides a thermodynamic grounding for the framework: the Psi-field is not a mysterious force but a coherence pump. Its physical role is to maintain low-entropy phase states that would otherwise decay. The "cost" of',
            "P0R03911: this coherence maintenance is energy dissipation elsewhere in the system, consistent with standard thermodynamics.",
            "P0R03912: Connection to Decoherence Suppression",
            "P0R03913: If _coh is tied to quantum phase coherence, then Psi increasing _coh means Psi slows decoherence. A high _coh means more quantum coherence is preserved across the system.",
            "P0R03914: This connects directly to the consciousness-modulated decoherence rate :",
            "P0R03915: _psi(t) = e^{lambda_psi psi(r,t)}",
            "P0R03916: The mechanism is now clear: larger Psi -> larger _coh -> lower effective entropy -> reduced decoherence rate _psi. The exponential form in the decoherence rate arises naturally from the entropy functional's logarithmic",
            "P0R03917: structure.",
            "P0R03918: Connection to the Arrow of Time",
            "P0R03919: Phase coherence and entropy are inversely related. If the Psi-field modulates entropy, it could modulate the local arrow of time.",
            "P0R03920: Phase locking lifts the system against entropy increase: increasing phase coherence corresponds to moving against the thermodynamic arrow. In this picture, consciousness does not reverse time but locally slows the entropy",
            "P0R03921: production rate, effectively stretching the timescale over which coherent structures persist.",
            "P0R03922: This could be tested: systems with higher measured consciousness correlates (EEG coherence, integrated information) should exhibit measurably lower local entropy production rates compared to matched controls. The",
            "P0R03923: predicted effect size is set by lambda_{psi,EM} .",
            "P0R03924: P0R03924",
            "P0R03925: The Nieh-Yan Torsion Preservation Law Conformal Invariance of Topological Torsion",
            "P0R03926: P0R03926",
            "P0R03927: To ensure survival across the MMC reset, the Conformal Invariant Torsion ($\\mathcal{T}_{SEC}$) is identified with the Nieh-Yan topological density, which is invariant under conformal rescaling $g \\to \\Omega^2 g$:",
            "P0R03928: j_sec_invariant = integrate(t_tensor ^ t_tensor - r_curvature_form ^ vielbein ^ vielbein, m_manifold)",
            "P0R03929: Legend of Equation Components:",
            "P0R03930: j_sec_invariant: The dimensionless conserved Ethical Functional across aeons. | t_tensor: The torsion 2-form. | r_curvature_form: The Riemann curvature 2-form. | vielbein: The frame field (tetrad) representing the local scale. | ^: The exterior (wedge) product.",
            "P0R03931: P0R03931",
        ),
        "test_protocols": ("preserve 5.2 New Falsifiable Predictions source-accounting boundary",),
        "null_results": ("5.2 New Falsifiable Predictions is not empirical validation evidence",),
        "variables": ("5_2_new_falsifiable_predictions",),
        "validation_targets": ("preserve records P0R03873-P0R03931",),
        "null_controls": (
            "5_2_new_falsifiable_predictions must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section51TheModifiedPathIntegralWithCefWeightingSpec:
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
class Section51TheModifiedPathIntegralWithCefWeightingSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section51TheModifiedPathIntegralWithCefWeightingSpec, ...]
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


def build_section_5_1_the_modified_path_integral_with_cef_weighting_specs(
    source_records: list[dict[str, Any]],
) -> Section51TheModifiedPathIntegralWithCefWeightingSpecBundle:
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

    specs: list[Section51TheModifiedPathIntegralWithCefWeightingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section51TheModifiedPathIntegralWithCefWeightingSpec(
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
        "title": "Paper 0 " + "5.1 The Modified Path Integral with CEF Weighting" + " Specs",
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
        "next_source_boundary": "P0R03932",
    }
    return Section51TheModifiedPathIntegralWithCefWeightingSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section51TheModifiedPathIntegralWithCefWeightingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_5_1_the_modified_path_integral_with_cef_weighting_specs(
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


def render_report(bundle: Section51TheModifiedPathIntegralWithCefWeightingSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "5.1 The Modified Path Integral with CEF Weighting" + " Specs",
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
    bundle: Section51TheModifiedPathIntegralWithCefWeightingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_5_1_the_modified_path_integral_with_cef_weighting_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_5_1_the_modified_path_integral_with_cef_weighting_validation_specs_{date_tag}.md"
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
