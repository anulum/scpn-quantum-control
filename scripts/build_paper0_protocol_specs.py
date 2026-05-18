#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Protocol spec builder
"""Promote Paper 0 Protocol records."""

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
    "P0R05191",
    "P0R05192",
    "P0R05193",
    "P0R05194",
    "P0R05195",
    "P0R05196",
    "P0R05197",
    "P0R05198",
    "P0R05199",
    "P0R05200",
    "P0R05201",
)
CLAIM_BOUNDARY = "source-bounded protocol source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "protocol.protocol": {
        "context_id": "protocol",
        "validation_protocol": "paper0.protocol.protocol",
        "canonical_statement": "The source-bounded component 'Protocol' preserves Paper 0 records P0R05191-P0R05196 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05191:protocol",
            "P0R05192:protocol",
            "P0R05193:protocol",
            "P0R05194:protocol",
            "P0R05195:protocol",
            "P0R05196:protocol",
        ),
        "source_formulae": (
            "P0R05191: Protocol",
            "P0R05192: Baseline Characterization: Characterize the NV center ensemble's intrinsic coherence times (T2) using standard Ramsey interferometry pulse sequences while the neuronal culture is in a quiescent state (induced by tetrodotoxin, TTX). This establishes the baseline decoherence rate, baseline=1/T2, in the absence of complex network activity. | Activity Modulation: Induce different states of network activity (e.g., synchronous bursting via the GABA$_A$ antagonist bicuculline) and continuously record both the neuronal spike trains from the MEA and the NV center's coherence time (T2) via repeated Ramsey measurements.",
            "P0R05193: 2a. Isomorphic Activity Control Condition: To rigorously exclude the possibility that observed NV-center decoherence is due to classical EM fields rather than the hypothesized informational coupling, a crucial control condition must be implemented.",
            'P0R05194: After recording a period of complex, critical activity (e.g., spontaneous avalanches), this exact spike train pattern will be "replayed" to the culture using the MEA\'s stimulation capabilities, while the culture itself is pharmacologically silenced (e.g., with TTX or CNQX/AP5).',
            'P0R05195: This "replay" condition generates a nearly identical classical magnetic field signature at the NV-center location but possesses minimal intrinsic informational complexity (the FIM of a deterministic playback is near zero). The critical test is the comparison of the NV-center decoherence rate during spontaneous complex activity (_spontaneous) versus the decoherence rate during the isomorphic replay (_replay). The SCPN predicts a statistically significant difference, Delta = _spontaneous - _replay > 0, demonstrating an excess decoherence correlated with intrinsic complexity independent of the classical field strength.',
            "P0R05196: Data Analysis: From the MEA spike train data, calculate the time-resolved magnetic field strength at the NV center's location using the Biot-Savart law to control for classical electromagnetic effects. | From the same spike train data, compute a time-resolved proxy for the informational complexity of the network activity. A direct, real-time calculation of the FIM is computationally prohibitive. A practical and robust proxy, such as the Lempel-Ziv complexity of the binarised network activity or the power-law exponent (tau) of the neuronal avalanche size distribution, will be used. A lower exponent tau indicates activity closer to criticality and thus higher complexity. | Perform a multiple regression analysis on the time-series data. The dependent variable is the anomalous decoherence rate, (t)=(1/T2(t))baseline. The independent variables are the calculated classical magnetic field strength and the informational complexity metric. The central test is for a statistically significant regression coefficient for the complexity metric, independent of the magnetic field.",
        ),
        "test_protocols": ("preserve Protocol source-accounting boundary",),
        "null_results": ("Protocol is not empirical validation evidence",),
        "variables": ("protocol",),
        "validation_targets": ("preserve records P0R05191-P0R05196",),
        "null_controls": ("protocol must remain source-bounded accounting",),
    },
    "protocol.falsification_condition": {
        "context_id": "falsification_condition",
        "validation_protocol": "paper0.protocol.falsification_condition",
        "canonical_statement": "The source-bounded component 'Falsification Condition' preserves Paper 0 records P0R05197-P0R05201 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05197:falsification_condition",
            "P0R05198:falsification_condition",
            "P0R05199:falsification_condition",
            "P0R05200:falsification_condition",
            "P0R05201:falsification_condition",
        ),
        "source_formulae": (
            "P0R05197: Falsification Condition",
            "P0R05198: The theory is falsified if no statistically significant correlation is found between the NV center's anomalous decoherence and the informational complexity of the neuronal culture, or if any observed correlation can be fully explained by classical electromagnetic effects that were not adequately controlled for.",
            "P0R05199: Experimental Protocol Flowchart",
            'P0R05200: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]Fig.: Replay-Control Test for Information-Geometric Coupling. This flowchart details the sequence of the experimental procedure, highlighting the critical "Isomorphic Replay" control condition that isolates the informational component of the interaction from classical electromagnetic effects. This protocol cleanly separates classical EM effects from information-geometric effects: if replay fails to reproduce spontaneous decoherence-and complexity significantly explains residuals-then the data favour the proposed Psi-FIM coupling.',
            'P0R05201: 1) Baseline (TTX): Silence the culture and measure NV coherence $\\mathbf{T2}\\mathbf{*}\\mathbf{T}_{\\mathbf{2}}^{\\mathbf{*}}\\mathbf{T2}\\mathbf{*}\\mathbf{}$ via Ramsey to establish $\\Gamma baseline\\Gamma_{\\text{baseline}}\\Gamma baseline$. 2) Spontaneous (Bicuculline): $Record\\ Sspontaneous(t)S\\_\\{"\\{ spontaneous\\}\\}(t)Sspontaneous(t\\mathbf{)}\\ and\\ simultaneous\\ NV\\ coherence\\ \\Gamma spontaneous(t)\\Gamma\\_\\{"\\{ spontaneous\\}\\}(t)\\Gamma spontaneous(t$). 3) Isomorphic Replay Control (TTX): Re-silence, replay Sspontaneous(t)S_{\\text{spontaneous}}(t)Sspontaneous(t) via MEA stimulation, and measure $\\Gamma replay(t)\\Gamma_{\\text{replay}}(t)\\Gamma replay(t).$ 4) Analysis: Classical EM control: Compute the B-field at the NV from $Sspontaneous(t)S_{\\text{spontaneous}}(t)Sspontaneous(t)(Biot - Savart)$. Informational metric: Compute a proxy for Tr(gFIM)\\mathrm{Tr}(g_{\\mathrm{FIM}})Tr(gFIM) (e.g., Lempel-Ziv, avalanche exponent tau\\tautau). Hypothesis tests: Core $test - \\Delta\\Gamma = \\Gamma spontaneous - \\Gamma replay > 0\\Delta\\Gamma = \\Gamma_{\\text{spontaneous}} - \\Gamma_{\\text{replay}} > 0\\Delta\\Gamma = \\Gamma spontaneous - \\Gamma replay > 0.Regression - model\\delta\\Gamma(t) = \\Gamma spontaneous(t) - \\Gamma baseline(t)\\delta\\Gamma(t) = \\Gamma_{\\text{spontaneous}}(t) - \\Gamma_{\\text{baseline}}(t)\\delta\\Gamma(t) = \\Gamma spontaneous(t) - \\Gamma baseline(t)$ with regressors B-field(t) and Complexity(t); seek a significant positive coefficient on the complexity term. Falsification: If $\\Delta\\Gamma\\Delta\\Gamma\\Delta\\Gamma$ is not >0>0>0 or the complexity coefficient is not significant, the coupling hypothesis is disconfirmed.',
        ),
        "test_protocols": ("preserve Falsification Condition source-accounting boundary",),
        "null_results": ("Falsification Condition is not empirical validation evidence",),
        "variables": ("falsification_condition",),
        "validation_targets": ("preserve records P0R05197-P0R05201",),
        "null_controls": ("falsification_condition must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
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
class ProtocolSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ProtocolSpec, ...]
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


def build_protocol_specs(source_records: list[dict[str, Any]]) -> ProtocolSpecBundle:
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

    specs: list[ProtocolSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ProtocolSpec(
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
        "title": "Paper 0 " + "Protocol" + " Specs",
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
        "next_source_boundary": "P0R05202",
    }
    return ProtocolSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> ProtocolSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_protocol_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ProtocolSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Protocol" + " Specs",
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
    bundle: ProtocolSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_protocol_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_protocol_validation_specs_{date_tag}.md"
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
