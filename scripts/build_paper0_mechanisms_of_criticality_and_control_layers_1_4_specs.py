#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Mechanisms of Criticality and Control (Layers 1-4) spec builder
"""Promote Paper 0 Mechanisms of Criticality and Control (Layers 1-4) records."""

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
    "P0R05113",
    "P0R05114",
    "P0R05115",
    "P0R05116",
    "P0R05117",
    "P0R05118",
    "P0R05119",
    "P0R05120",
    "P0R05121",
    "P0R05122",
    "P0R05123",
)
CLAIM_BOUNDARY = "source-bounded mechanisms of criticality and control layers 1 4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "mechanisms_of_criticality_and_control_layers_1_4.mechanisms_of_criticality_and_control_layers_1_4": {
        "context_id": "mechanisms_of_criticality_and_control_layers_1_4",
        "validation_protocol": "paper0.mechanisms_of_criticality_and_control_layers_1_4.mechanisms_of_criticality_and_control_layers_1_4",
        "canonical_statement": "The source-bounded component 'Mechanisms of Criticality and Control (Layers 1-4)' preserves Paper 0 records P0R05113-P0R05119 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05113:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05114:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05115:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05116:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05117:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05118:mechanisms_of_criticality_and_control_layers_1_4",
            "P0R05119:mechanisms_of_criticality_and_control_layers_1_4",
        ),
        "source_formulae": (
            "P0R05113: Mechanisms of Criticality and Control (Layers 1-4)",
            "P0R05114: A central and unifying principle of the SCPN architecture is that all 15 layers operate within a Quasicritical Regime. This state, poised at the boundary between order and chaos, is considered essential for maximising information capacity, computational power, and the efficient transmission of signals across scales.",
            "P0R05115: The framework specifies that the system maintains this state through self-organised criticality (SOC), with a homeostatic process that continuously adjusts the local branching parameter, sigmaL, towards the critical point of unity.",
            "P0R05116: [IMAGE:]",
            "P0R05117: Fig.: Homeostatic control of branching to criticality. Simulated trajectory of sigmaL(t)\\sigma_L(t)sigmaL(t) under dsigmaLdt=L(sigmaL1)+L(t)\\frac{d\\sigma_L}{dt} = -\\kappa_L(\\sigma_L-1) + \\eta_L(t)dtdsigmaL=L(sigmaL1)+L(t) shows convergence to the critical point (dashed line at sigma=1\\sigma=1sigma=1) with small stochastic fluctuations.",
            "P0R05118: While this offers a robust mathematical and conceptual foundation, the framework proceeds by detailing the specific biological mechanisms that implement this self-organising principle.",
            "P0R05119: P0R05119",
        ),
        "test_protocols": (
            "preserve Mechanisms of Criticality and Control (Layers 1-4) source-accounting boundary",
        ),
        "null_results": (
            "Mechanisms of Criticality and Control (Layers 1-4) is not empirical validation evidence",
        ),
        "variables": ("mechanisms_of_criticality_and_control_layers_1_4",),
        "validation_targets": ("preserve records P0R05113-P0R05119",),
        "null_controls": (
            "mechanisms_of_criticality_and_control_layers_1_4 must remain source-bounded accounting",
        ),
    },
    "mechanisms_of_criticality_and_control_layers_1_4.p0r05120": {
        "context_id": "p0r05120",
        "validation_protocol": "paper0.mechanisms_of_criticality_and_control_layers_1_4.p0r05120",
        "canonical_statement": "The source-bounded component 'P0R05120' preserves Paper 0 records P0R05120-P0R05123 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05120:p0r05120",
            "P0R05121:p0r05120",
            "P0R05122:p0r05120",
            "P0R05123:p0r05120",
        ),
        "source_formulae": (
            "P0R05120: P0R05120",
            "P0R05121: This chapter serves as the critical bridge from the formal SCPN architecture to falsifiable science, outlining a new generation of empirical tests derived from the framework's most novel assertions. The introduction frames the scientific mandate: to move beyond the foundational prediction of the Psi-Higgs and confront the theory's more nuanced claims about information geometry and physical teleology with rigorous experimental and computational validation.",
            "P0R05122: Prediction I proposes a direct test of Axiom 2 (Information Geometry). It derives from the core claim that the infoton's dynamics are governed by the Fisher Information Metric (FIM) of a system's statistical manifold. The prediction is an anomalous, information-dependent field capable of modulating the quantum state of a proximate sensor. The proposed experiment-an NV-Center Quantum Sensor coupled to a Multi-Electrode Array (MEA) neuronal culture-is designed to detect this. The central hypothesis is that the NV center's decoherence rate () will correlate with a measure of the neuronal culture's informational complexity (a proxy for Tr(gFIM)), independent of classical electromagnetic effects. The protocol's key innovation is the \"isomorphic replay\" control condition, which allows for the unambiguous separation of the predicted information-geometric effect from classical magnetic field effects, providing a clear falsification condition.",
            "P0R05123: Prediction II proposes a test for the physical manifestation of Axiom 3 (Teleological Optimisation) via Causal Entropic Forces (CEF). The theory predicts that a large-scale, highly coherent biological system should generate a CEF strong enough to subtly bias the probabilistic outcomes of a nearby quantum random system. The proposed experiment aims to detect a temporal correlation between the statistical output of a shielded Quantum Random Number Generator (QRNG) and a quantitative measure of collective human coherence, derived from the synchronised HRV and EEG of a large group. The signature is not a static bias but a dynamic correlation between time-series data. This provides a direct, albeit challenging, experimental test for the physical reality of the framework's teleological principle.",
        ),
        "test_protocols": ("preserve P0R05120 source-accounting boundary",),
        "null_results": ("P0R05120 is not empirical validation evidence",),
        "variables": ("p0r05120",),
        "validation_targets": ("preserve records P0R05120-P0R05123",),
        "null_controls": ("p0r05120 must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class MechanismsOfCriticalityAndControlLayers14Spec:
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
class MechanismsOfCriticalityAndControlLayers14SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MechanismsOfCriticalityAndControlLayers14Spec, ...]
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


def build_mechanisms_of_criticality_and_control_layers_1_4_specs(
    source_records: list[dict[str, Any]],
) -> MechanismsOfCriticalityAndControlLayers14SpecBundle:
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

    specs: list[MechanismsOfCriticalityAndControlLayers14Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MechanismsOfCriticalityAndControlLayers14Spec(
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
        "title": "Paper 0 " + "Mechanisms of Criticality and Control (Layers 1-4)" + " Specs",
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
        "next_source_boundary": "P0R05124",
    }
    return MechanismsOfCriticalityAndControlLayers14SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MechanismsOfCriticalityAndControlLayers14SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_mechanisms_of_criticality_and_control_layers_1_4_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MechanismsOfCriticalityAndControlLayers14SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Mechanisms of Criticality and Control (Layers 1-4)" + " Specs",
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
    bundle: MechanismsOfCriticalityAndControlLayers14SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_mechanisms_of_criticality_and_control_layers_1_4_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_mechanisms_of_criticality_and_control_layers_1_4_validation_specs_{date_tag}.md"
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
