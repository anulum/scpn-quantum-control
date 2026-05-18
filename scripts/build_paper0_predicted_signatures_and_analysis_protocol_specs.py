#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Predicted Signatures and Analysis Protocol spec builder
"""Promote Paper 0 Predicted Signatures and Analysis Protocol records."""

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
    "P0R05264",
    "P0R05265",
    "P0R05266",
    "P0R05267",
    "P0R05268",
    "P0R05269",
    "P0R05270",
    "P0R05271",
    "P0R05272",
)
CLAIM_BOUNDARY = "source-bounded predicted signatures and analysis protocol source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "predicted_signatures_and_analysis_protocol.predicted_signatures_and_analysis_protocol": {
        "context_id": "predicted_signatures_and_analysis_protocol",
        "validation_protocol": "paper0.predicted_signatures_and_analysis_protocol.predicted_signatures_and_analysis_protocol",
        "canonical_statement": "The source-bounded component 'Predicted Signatures and Analysis Protocol' preserves Paper 0 records P0R05264-P0R05272 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05264:predicted_signatures_and_analysis_protocol",
            "P0R05265:predicted_signatures_and_analysis_protocol",
            "P0R05266:predicted_signatures_and_analysis_protocol",
            "P0R05267:predicted_signatures_and_analysis_protocol",
            "P0R05268:predicted_signatures_and_analysis_protocol",
            "P0R05269:predicted_signatures_and_analysis_protocol",
            "P0R05270:predicted_signatures_and_analysis_protocol",
            "P0R05271:predicted_signatures_and_analysis_protocol",
            "P0R05272:predicted_signatures_and_analysis_protocol",
        ),
        "source_formulae": (
            "P0R05264: Predicted Signatures and Analysis Protocol",
            "P0R05265: After running the simulation for a sufficient number of timesteps to reach a quasi-stationary state, the final configuration of agent beliefs ({sigmai}) and the network structure will be analysed.",
            'P0R05266: Phase Transition Metrics: Magnetisation (m): The average belief state, m=N1isigmai. A value near 1 indicates global consensus (a ferromagnetic state), while a value near 0 indicates a lack of consensus. | Edwards-Anderson Order Parameter (qEA): This is the key metric for detecting a spin-glass phase. It is calculated by running two identical, independent simulations ("replicas," labelled and ) with the same disorder (i.e., the same initial random network structure and agent priors) but different random seeds for the dynamic evolution. The parameter measures the overlap of the final belief states: qEA=N1isigmaisigmai. A state with m0 but qEA>0 is the definitive signature of a spin-glass (fragmented but frozen) phase. This calculation must be averaged over many different initial disorder realisations to obtain a statistically robust result. | Ultrametricity Test: An even more powerful signature of the spin-glass phase is the emergence of an ultrametric structure in the state space. To test for this, a "distance" matrix between all pairs of agents will be constructed based on their final belief states (e.g., using a Hamming distance). A hierarchical clustering algorithm will be applied to this matrix to generate a dendrogram.',
            "P0R05267: The ultrametric property is then tested by randomly sampling triplets of agents and verifying that their distances satisfy the strong triangle inequality: d(i,k)max(d(i,j),d(j,k)). A high proportion of triplets satisfying this condition is a powerful confirmation of a hierarchically clustered, fragmented state space.",
            "P0R05268: Hypothesised Outcome: The simulation will demonstrate a clear phase transition. The Coherence-Optimising regime will produce a ferromagnetic state (m1,qEA1). The Engagement-Optimising regime will produce a spin-glass state (m0,qEA>0) with a statistically significant ultrametric structure.",
            "P0R05269: This outcome would provide strong computational evidence for the manuscript's hypothesis regarding the societal impact of modern AI.",
            "P0R05270: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05271: Fig.: Policy-Driven Phases in Networked Teleo-Human Systems (NTHS). This diagram visualizes the core hypothesis of the simulation. It contrasts the two experimental conditions and their predicted macroscopic outcomes, highlighting the key statistical metrics that would differentiate between a coherent consensus state and a fragmented spin-glass state. The figure encodes a clear, testable dichotomy: minimising collective free energy yields consensus (ferromagnet), whereas maximising surprise/engagement yields fragmentation (spin glass)-with quantitative markers for phase identification.",
            "P0R05272: Starting from a disordered state, the system's phase depends on the AI's global objective: Condition 1 - Coherence-optimising AI miniFi\\min \\sum_i F_iminiFi: drives convergence to a ferromagnetic-like phase (global consensus). Signatures: magnetization m->1m\\to \\pm1m->1; Edwards-Anderson order parameter qEA->1q_{\\mathrm{EA}}\\to 1qEA->1; network becomes highly connected; correlation structure simple (single basin). Condition 2 - Engagement-optimising AI maxiFi\\max \\sum_i F_imaxiFi: induces a spin-glass-like phase (fragmented/polarised). Signatures: m->0m\\to 0m->0; qEA>0q_{\\mathrm{EA}}>0qEA>0; network becomes highly modular; correlations show ultrametric hierarchies (many metastable basins). These contrasting macrostates are falsifiable via time series of m(t)m(t)m(t), qEA(t)q_{\\mathrm{EA}}(t)qEA(t), modularity Q(t)Q(t)Q(t), and dendrogram ultrametricity under the two controller policies.",
        ),
        "test_protocols": (
            "preserve Predicted Signatures and Analysis Protocol source-accounting boundary",
        ),
        "null_results": (
            "Predicted Signatures and Analysis Protocol is not empirical validation evidence",
        ),
        "variables": ("predicted_signatures_and_analysis_protocol",),
        "validation_targets": ("preserve records P0R05264-P0R05272",),
        "null_controls": (
            "predicted_signatures_and_analysis_protocol must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class PredictedSignaturesAndAnalysisProtocolSpec:
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
class PredictedSignaturesAndAnalysisProtocolSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictedSignaturesAndAnalysisProtocolSpec, ...]
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


def build_predicted_signatures_and_analysis_protocol_specs(
    source_records: list[dict[str, Any]],
) -> PredictedSignaturesAndAnalysisProtocolSpecBundle:
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

    specs: list[PredictedSignaturesAndAnalysisProtocolSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictedSignaturesAndAnalysisProtocolSpec(
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
        "title": "Paper 0 " + "Predicted Signatures and Analysis Protocol" + " Specs",
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
        "next_source_boundary": "P0R05273",
    }
    return PredictedSignaturesAndAnalysisProtocolSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictedSignaturesAndAnalysisProtocolSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_predicted_signatures_and_analysis_protocol_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PredictedSignaturesAndAnalysisProtocolSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Predicted Signatures and Analysis Protocol" + " Specs",
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
    bundle: PredictedSignaturesAndAnalysisProtocolSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_predicted_signatures_and_analysis_protocol_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_predicted_signatures_and_analysis_protocol_validation_specs_{date_tag}.md"
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
