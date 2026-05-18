#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting spec builder
"""Promote Paper 0 Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting records."""

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
    "P0R06179",
    "P0R06180",
    "P0R06181",
    "P0R06182",
    "P0R06183",
    "P0R06184",
    "P0R06185",
    "P0R06186",
    "P0R06187",
    "P0R06188",
    "P0R06189",
    "P0R06190",
    "P0R06191",
    "P0R06192",
    "P0R06193",
    "P0R06194",
    "P0R06195",
    "P0R06196",
)
CLAIM_BOUNDARY = "source-bounded resolving the amplitude friction the stuart landau upgrade for precision source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision.resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision": {
        "context_id": "resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
        "validation_protocol": "paper0.resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision.resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
        "canonical_statement": "The source-bounded component 'Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting' preserves Paper 0 records P0R06179-P0R06196 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06179:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06180:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06181:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06182:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06183:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06184:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06185:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06186:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06187:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06188:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06189:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06190:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06191:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06192:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06193:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06194:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06195:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
            "P0R06196:resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",
        ),
        "source_formulae": (
            "P0R06179: Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting",
            "P0R06180: P0R06180",
            "P0R06181: While the Kuramoto formulation of the UPDE elegantly captures the minimization of prediction error via phase-locking ($\\sin(\\theta_j - \\theta_i) \\to 0$), it introduces a severe limitation when strictly mapped to the Free Energy Principle (FEP). The FEP relies not just on the mean of a prediction (encoded by phase, $\\theta$), but fundamentally on its precision, or inverse variance (encoded by amplitude, $R$).",
            "P0R06182: In biological neural networks, precision weighting is physically instantiated by the modulatory gain of synaptic populations-effectively, the amplitude of the oscillatory signal. Because the standard Kuramoto model assumes all oscillators possess a constant, uniform amplitude, it mathematically erases the system's ability to represent confidence. Consequently, mapping the Salience Network's function to a purely phase-coupled UPDE is mathematically incomplete.",
            "P0R06183: To rigorously resolve this and complete the HPC mapping, the UPDE must be formally upgraded from a phase-only Kuramoto model to a network of Stuart-Landau Oscillators. The state of each neural ensemble $j$ is now defined by a complex variable $Z_j = R_j e^{i\\theta_j}$, incorporating both amplitude ($R_j$) and phase ($\\theta_j$).",
            "P0R06184: The upgraded complex dynamics are governed by:",
            "P0R06185: $$\\dot{Z}_j = Z_j (\\rho_j + i\\omega_j - |Z_j|^2) + \\sum_{k} K_{jk} Z_k + \\eta_j(t)$$",
            "P0R06186: Where $\\rho_j$ determines the bifurcation parameter (the distance from the critical point). By separating this complex equation into its polar coordinates, the true, precision-weighted Phase Dynamics Equation emerges:",
            "P0R06187: $$\\dot{\\theta}_j = \\omega_j + \\sum_{k} K_{jk} \\frac{R_k}{R_j} \\sin(\\theta_k - \\theta_j) + \\eta_j^\\theta(t)$$",
            "P0R06188: $$\\dot{R}_j = R_j(\\rho_j - R_j^2) + \\sum_{k} K_{jk} R_k \\cos(\\theta_k - \\theta_j) + \\eta_j^R(t)$$",
            "P0R06189: The Active Inference Mapping:",
            "P0R06190: This Stuart-Landau formulation provides the exact mathematical isomorphism to Active Inference:",
            "P0R06191: The Mean (Belief): $\\theta_j$ represents the expected value of the generative model's prediction. | The Precision (Confidence): $R_j$ represents the precision matrix ($\\Pi_j$) of that belief.",
            "P0R06192: The crucial term in the upgraded phase equation is the amplitude ratio: $\\frac{R_k}{R_j}$. This is the literal mathematical definition of precision-weighted prediction error.",
            "P0R06193: If an incoming sensory signal $k$ has very high precision/amplitude ($R_k \\gg R_j$), the phase update $\\dot{\\theta}_j$ is dominated by the prediction error $\\sin(\\theta_k - \\theta_j)$, forcing the higher-level generative model to rapidly update its beliefs. | Conversely, if the prior belief is highly precise ($R_j \\gg R_k$), the error signal is effectively muted, and the system ignores the sensory noise.",
            "P0R06194: The Role of the Salience Network:",
            "P0R06195: This formalism physically grounds Layer 5 dynamics. The Salience Network does not alter the phase of the network directly; it acts on the radial equation ($\\dot{R}_j$) by modulating the local bifurcation parameter $\\rho_j$ via neuromodulatory gain (e.g., dopamine or acetylcholine). By dynamically pumping the amplitude $R_k$ of specific perceptual channels, the Salience Network forces those specific prediction errors to dominate the global UPDE, dictating the trajectory of Free Energy minimization.",
            "P0R06196: P0R06196",
        ),
        "test_protocols": (
            "preserve Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting is not empirical validation evidence",
        ),
        "variables": ("resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision",),
        "validation_targets": ("preserve records P0R06179-P0R06196",),
        "null_controls": (
            "resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpec:
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
class ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpec, ...]
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


def build_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_specs(
    source_records: list[dict[str, Any]],
) -> ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle:
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

    specs: list[ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpec(
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
        + "Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting"
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
        "next_source_boundary": "P0R06197",
    }
    return ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_specs(
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
    bundle: ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Resolving the Amplitude Friction: The Stuart-Landau Upgrade for Precision Weighting"
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
    bundle: ResolvingTheAmplitudeFrictionTheStuartLandauUpgradeForPrecisionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_resolving_the_amplitude_friction_the_stuart_landau_upgrade_for_precision_validation_specs_{date_tag}.md"
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
