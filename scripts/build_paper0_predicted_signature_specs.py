#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Predicted Signature spec builder
"""Promote Paper 0 Predicted Signature records."""

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
    "P0R05171",
    "P0R05172",
    "P0R05173",
    "P0R05174",
    "P0R05175",
    "P0R05176",
    "P0R05177",
    "P0R05178",
    "P0R05179",
    "P0R05180",
    "P0R05181",
)
CLAIM_BOUNDARY = (
    "source-bounded predicted signature source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "predicted_signature.predicted_signature": {
        "context_id": "predicted_signature",
        "validation_protocol": "paper0.predicted_signature.predicted_signature",
        "canonical_statement": "The source-bounded component 'Predicted Signature' preserves Paper 0 records P0R05171-P0R05181 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05171:predicted_signature",
            "P0R05172:predicted_signature",
            "P0R05173:predicted_signature",
            "P0R05174:predicted_signature",
            "P0R05175:predicted_signature",
            "P0R05176:predicted_signature",
            "P0R05177:predicted_signature",
            "P0R05178:predicted_signature",
            "P0R05179:predicted_signature",
            "P0R05180:predicted_signature",
            "P0R05181:predicted_signature",
        ),
        "source_formulae": (
            "P0R05171: Predicted Signature",
            "P0R05172: The existence of an anomalous, information-dependent field that subtly modulates the quantum state evolution of nearby sensitive detectors is predicted. The specific signature is a correlation between a measure of the source system's informational complexity and the decoherence rate (or Larmor frequency) of a quantum sensor, after controlling for all classical electromagnetic and environmental effects.",
            "P0R05173: The magnitude of this anomalous decoherence, , is hypothesised to be proportional to the trace of the FIM, Tr(gFIM), a scalar measure of the total informational volume of the source system's statistical manifold.",
            "P0R05174: $\\delta\\Gamma \\propto Tr\\left( gFIM(\\theta) \\right) - Tr(gbaselineFIM)$",
            "P0R05175: P0R05175",
            "P0R05176: Theoretical Mechanism Diagram",
            "P0R05177: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Display enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05178: Fig.: Psi-Field on the Statistical Manifold (FIM->Decoherence). This diagram illustrates the core physical claim: the Psi-field's interaction is mediated by the informational geometry (Fisher Information Metric) of a source system, which in turn perturbs a nearby quantum sensor. This figure operationalises the information-geometric route: if the Psi-field couples to FIM geometry, decoherence shifts in a proximate quantum probe provide a direct, quantitative experimental handle on consciousness-linked modulation.",
            "P0R05179: A neuronal culture defines a statistical manifold of firing-pattern distributions endowed with the Fisher information metric gFIMg_{\\text{FIM}}gFIM. In this proposal, the Psi-field couples not to spacetime directly but to this information geometry. The coupling perturbs a nearby quantum sensor (e.g., NV center), producing an anomalous decoherence \\delta\\Gamma. The testable hypothesis is a scaling law",
            "P0R05180: Tr (gFIM),\\delta\\Gamma \\;\\propto\\; \\mathrm{Tr}\\!\\big(g_{\\text{FIM}}\\big),Tr(gFIM),",
            "P0R05181: so manipulations that increase information-geometric curvature (e.g., structured bursting, controlled synchrony) should increase decoherence in the sensor, providing a quantitative bridge from neural statistics to quantum readouts.",
        ),
        "test_protocols": ("preserve Predicted Signature source-accounting boundary",),
        "null_results": ("Predicted Signature is not empirical validation evidence",),
        "variables": ("predicted_signature",),
        "validation_targets": ("preserve records P0R05171-P0R05181",),
        "null_controls": ("predicted_signature must remain source-bounded accounting",),
    }
}


@dataclass(frozen=True, slots=True)
class PredictedSignatureSpec:
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
class PredictedSignatureSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictedSignatureSpec, ...]
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


def build_predicted_signature_specs(
    source_records: list[dict[str, Any]],
) -> PredictedSignatureSpecBundle:
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

    specs: list[PredictedSignatureSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictedSignatureSpec(
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
        "title": "Paper 0 " + "Predicted Signature" + " Specs",
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
        "next_source_boundary": "P0R05182",
    }
    return PredictedSignatureSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> PredictedSignatureSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_predicted_signature_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PredictedSignatureSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Predicted Signature" + " Specs",
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
    bundle: PredictedSignatureSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_predicted_signature_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_predicted_signature_validation_specs_{date_tag}.md"
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
