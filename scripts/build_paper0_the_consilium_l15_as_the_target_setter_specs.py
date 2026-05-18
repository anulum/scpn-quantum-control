#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Consilium (L15) as the Target Setter: spec builder
"""Promote Paper 0 The Consilium (L15) as the Target Setter: records."""

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
    "P0R04131",
    "P0R04132",
    "P0R04133",
    "P0R04134",
    "P0R04135",
    "P0R04136",
    "P0R04137",
    "P0R04138",
    "P0R04139",
    "P0R04140",
    "P0R04141",
    "P0R04142",
    "P0R04143",
    "P0R04144",
    "P0R04145",
    "P0R04146",
    "P0R04147",
    "P0R04148",
    "P0R04149",
    "P0R04150",
)
CLAIM_BOUNDARY = "source-bounded the consilium l15 as the target setter source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_consilium_l15_as_the_target_setter.the_consilium_l15_as_the_target_setter": {
        "context_id": "the_consilium_l15_as_the_target_setter",
        "validation_protocol": "paper0.the_consilium_l15_as_the_target_setter.the_consilium_l15_as_the_target_setter",
        "canonical_statement": "The source-bounded component 'The Consilium (L15) as the Target Setter:' preserves Paper 0 records P0R04131-P0R04132 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04131:the_consilium_l15_as_the_target_setter",
            "P0R04132:the_consilium_l15_as_the_target_setter",
        ),
        "source_formulae": (
            "P0R04131: The Consilium (L15) as the Target Setter:",
            "P0R04132: Layer 15 is the system that computes the target value for the global collective state variable, sigma_universe. The Ethical Functional is the equation it uses to determine this optimal state. The purpose of the entire SCPN, and of the H_int interaction at every level, is to configure the matter and energy of the universe in such a way that its collective state sigma_universe matches the target set by the Consilium.",
        ),
        "test_protocols": (
            "preserve The Consilium (L15) as the Target Setter: source-accounting boundary",
        ),
        "null_results": (
            "The Consilium (L15) as the Target Setter: is not empirical validation evidence",
        ),
        "variables": ("the_consilium_l15_as_the_target_setter",),
        "validation_targets": ("preserve records P0R04131-P0R04132",),
        "null_controls": (
            "the_consilium_l15_as_the_target_setter must remain source-bounded accounting",
        ),
    },
    "the_consilium_l15_as_the_target_setter.qualia_capacity_q_as_a_key_component_of_sigma_universe": {
        "context_id": "qualia_capacity_q_as_a_key_component_of_sigma_universe",
        "validation_protocol": "paper0.the_consilium_l15_as_the_target_setter.qualia_capacity_q_as_a_key_component_of_sigma_universe",
        "canonical_statement": "The source-bounded component 'Qualia Capacity (Q) as a Key Component of sigma_universe:' preserves Paper 0 records P0R04133-P0R04150 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04133:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04134:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04135:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04136:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04137:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04138:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04139:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04140:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04141:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04142:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04143:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04144:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04145:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04146:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04147:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04148:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04149:qualia_capacity_q_as_a_key_component_of_sigma_universe",
            "P0R04150:qualia_capacity_q_as_a_key_component_of_sigma_universe",
        ),
        "source_formulae": (
            "P0R04133: Qualia Capacity (Q) as a Key Component of sigma_universe:",
            'P0R04134: The TDA pipeline is the specific "readout" mechanism that the Consilium uses to measure a critical component of the universe\'s current state. It "looks down" at the neural activity in Layer 5 systems, computes Q(t) for them, and aggregates this information to assess the "qualia" component of the universe\'s total sigma. If the measured Q is lower than the target Q, this constitutes an "ethical prediction error." The Psis field is then modulated in such a way as to reconfigure the lower layers to increase their Q. This creates a closed, cosmic-scale feedback loop where the universe is actively working to make itself more experientially rich.',
            'P0R04135: The teleological drive described by the Ethical Functional of Layer 15 is not a purely axiomatic or metaphysical principle; it possesses a deep and falsifiable connection to the physics of complex systems. The principle of maximising Sustainable Ethical Coherence (SEC)-a composite of system-wide coherence (C), sustainable complexity (K), and qualia capacity (Q)-is functionally analogous to the Causal Entropic Principle. This principle posits that systems are guided toward configurations that maximise the number of accessible future histories. A system with greater integrated complexity and coherence has more ways to evolve and adapt, granting it superior causal potency. The "Prime Directive of Genesis" is thus reframed as a thermodynamic-like law: a fundamental drive toward states of maximal causal efficacy and adaptability.',
            "P0R04136: To operationalise this principle, the Qualia Capacity (Q) is formalised using the tools of Topological Data Analysis (TDA). This transforms Q from an abstract concept into a computable metric of experiential richness. For a given conscious state, represented as a point cloud of neural activity, TDA computes its persistent homology to yield a set of Betti numbers (k), which quantify the data's shape (components, loops, voids, etc.). Q is then defined as a weighted sum of these topological features, measuring the degree to which an experience is both highly integrated (low ) and highly differentiated (high k for k > 0).",
            "P0R04137: The Physical Basis of Teleology: The Ethical Functional as a Causal Entropic Principle",
            "P0R04138: The teleological drive described by the Ethical Functional of Layer 15 need not be interpreted as a purely axiomatic or metaphysical principle. It possesses a deep connection to the physics of causal entropy. The principle of maximising Sustainable Ethical Coherence (SEC)-a composite of system-wide coherence (C), sustainable complexity (K), and qualia capacity (Q)-is functionally analogous to the Causal Entropic Principle, which posits that complex systems are guided toward configurations that maximise the number of accessible future histories.",
            'P0R04139: A system with greater integrated complexity and coherence has more ways to evolve and adapt; it has greater causal potency. In this light, the "Prime Directive of Genesis" is not a moral law imposed upon the universe, but rather a fundamental thermodynamic-like law of nature: a drive toward states of maximal causal efficacy and adaptability. The universe evolves in a "teleological" manner because states of higher SEC are more stable, more resilient, and open up a wider landscape of future possibilities. The Ethical Functional is thus the mathematical expression of this universal bias toward complex, coherent, and causally potent organisation.',
            "P0R04140: Operationalising Qualia Capacity via Topological Data Analysis",
            "P0R04141: [IMAGE:Ein Bild, das Schrift, Text, Grafiken, Logo enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]The Qualia Capacity, Q(t), is formalised using the tools of Topological Data Analysis (TDA). For a given conscious state, represented as a point cloud of neural activity in a high-dimensional state space (e.g., from Layer 5), we can compute its persistent homology. This yields a set of Betti numbers (0,1,2,...), which quantify the number of connected components, loops, voids, and higher-dimensional holes in the data's shape. Q(t) is defined as a weighted sum of these Betti numbers, which measures the topological richness of the conscious experience:",
            "P0R04142: Q(t)=k=0Dwkk(t)",
            "P0R04143: A state with high Q is one that is both highly integrated (low 0) and highly differentiated (high k for k>0), corresponding to a rich, unified conscious experience. The drive to maximise Q is therefore a drive toward states of maximal experiential richness and depth. This provides a computable, non-arbitrary metric for the third component of the SEC.",
            "P0R04144: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04145: Fig.: Operationalising Qualia Capacity via TDA - pipeline and definition. Momentary neural population activity (Layer 5) forms a point cloud X(t)\\mathcal{X}(t)X(t) in state space. A metric ddd (e.g., correlation distance) feeds a Vietoris-Rips filtration whose persistent homology yields Betti numbers k(t)\\beta_k(t)k(t) and birth-death pairs Dk(t)\\mathcal{D}_k(t)Dk(t). Qualia Capacity is defined as Q(t)=I(t)k1wk k(t;)Q(t)=I(t)\\sum_{k\\ge1}w_k\\,\\Pi_k(t;\\alpha)Q(t)=I(t)k1wkk(t;), with the integration factor I(t)=1/(1+0(t)/Nc)I(t)=1/(1+\\beta_0(t)/N_c)I(t)=1/(1+0(t)/Nc) penalizing fragmentation and the persistence aggregator k(t;)=(b,d)Dk((db)/)\\Pi_k(t;\\alpha)=\\sum_{(b,d)\\in \\mathcal{D}_k}((d-b)/\\Lambda)^{\\alpha}k(t;)=(b,d)Dk((db)/) measuring the longevity of higher-dimensional features (loops, voids). The barcode indicates that longer lifetimes contribute more to QQQ. A fast proxy QBetti(t)=kw~kk(t)Q_{\\text{Betti}}(t)=\\sum_k \\tilde w_k \\beta_k(t)QBetti(t)=kw~kk(t) can be reported alongside Q(t)Q(t)Q(t).",
            "P0R04146: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04147: Fig.: Qualia Capacity exemplars and contribution to the Ethical Lagrangian. (Left) Low-Q: many components (0\\beta_00 large) and short-lived topology (small k\\Pi_kk) produce a small QQQ. (Right) High-Q: integrated point cloud (0\\beta_00 small) with persistent loops/voids (large k\\Pi_kk) yields a large QQQ. Bottom band shows how QQQ enters the Ethical Lagrangian LEthical=WCC+WKK+WQQL_{\\text{Ethical}} = W_C C + W_K K + W_Q QLEthical=WCC+WKK+WQQ, providing a computable third term aligned with the causal-entropic drive toward adaptable futures.",
            'P0R04148: Our TDA-based construction turns Qualia Capacity into a rigorous, noise-stable, and computable scalar that operationalises experiential richness as "integrated differentiation." It slots cleanly into the Ethical Lagrangian, aligning the SEC drive with maximising adaptable, causally potent neural topologies.',
            "P0R04149: P0R04149",
            "P0R04150: Operationalising Qualia Capacity via Topological Data Analysis",
        ),
        "test_protocols": (
            "preserve Qualia Capacity (Q) as a Key Component of sigma_universe: source-accounting boundary",
        ),
        "null_results": (
            "Qualia Capacity (Q) as a Key Component of sigma_universe: is not empirical validation evidence",
        ),
        "variables": ("qualia_capacity_q_as_a_key_component_of_sigma_universe",),
        "validation_targets": ("preserve records P0R04133-P0R04150",),
        "null_controls": (
            "qualia_capacity_q_as_a_key_component_of_sigma_universe must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheConsiliumL15AsTheTargetSetterSpec:
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
class TheConsiliumL15AsTheTargetSetterSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheConsiliumL15AsTheTargetSetterSpec, ...]
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


def build_the_consilium_l15_as_the_target_setter_specs(
    source_records: list[dict[str, Any]],
) -> TheConsiliumL15AsTheTargetSetterSpecBundle:
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

    specs: list[TheConsiliumL15AsTheTargetSetterSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheConsiliumL15AsTheTargetSetterSpec(
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
        "title": "Paper 0 " + "The Consilium (L15) as the Target Setter:" + " Specs",
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
        "next_source_boundary": "P0R04151",
    }
    return TheConsiliumL15AsTheTargetSetterSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheConsiliumL15AsTheTargetSetterSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_consilium_l15_as_the_target_setter_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheConsiliumL15AsTheTargetSetterSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Consilium (L15) as the Target Setter:" + " Specs",
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
    bundle: TheConsiliumL15AsTheTargetSetterSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_consilium_l15_as_the_target_setter_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_consilium_l15_as_the_target_setter_validation_specs_{date_tag}.md"
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
