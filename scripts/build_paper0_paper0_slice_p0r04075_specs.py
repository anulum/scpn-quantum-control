#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0  spec builder
"""Promote Paper 0  records."""

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
    "P0R04075",
    "P0R04076",
    "P0R04077",
    "P0R04078",
    "P0R04079",
    "P0R04080",
    "P0R04081",
    "P0R04082",
    "P0R04083",
    "P0R04084",
    "P0R04085",
    "P0R04086",
    "P0R04087",
    "P0R04088",
)
CLAIM_BOUNDARY = (
    "source-bounded paper0 slice p0r04075 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "paper0_slice_p0r04075.p0r04075": {
        "context_id": "p0r04075",
        "validation_protocol": "paper0.paper0_slice_p0r04075.p0r04075",
        "canonical_statement": "The source-bounded component 'P0R04075' preserves Paper 0 records P0R04075-P0R04075 without empirical validation claims.",
        "source_equation_ids": ("P0R04075:p0r04075",),
        "source_formulae": ("P0R04075: P0R04075",),
        "test_protocols": ("preserve P0R04075 source-accounting boundary",),
        "null_results": ("P0R04075 is not empirical validation evidence",),
        "variables": ("p0r04075",),
        "validation_targets": ("preserve records P0R04075-P0R04075",),
        "null_controls": ("p0r04075 must remain source-bounded accounting",),
    },
    "paper0_slice_p0r04075.resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec": {
        "context_id": "resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
        "validation_protocol": "paper0.paper0_slice_p0r04075.resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
        "canonical_statement": "The source-bounded component 'Resolving the Attractor Freeze Paradox: The Strange Attractor of SEC' preserves Paper 0 records P0R04076-P0R04088 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04076:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04077:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04078:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04079:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04080:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04081:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04082:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04083:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04084:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04085:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04086:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04087:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
            "P0R04088:resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
        ),
        "source_formulae": (
            "P0R04076: Resolving the Attractor Freeze Paradox: The Strange Attractor of SEC",
            "P0R04077: P0R04077",
            'P0R04078: The proposition that the Renormalisation Group (RG) flow of the universe tends toward a cosmic fixed point ($g^*$) where $\\beta(g^*) = 0$ introduces a severe dynamical paradox if interpreted naively. In classical dynamical systems, a simple fixed point acts as a definitive sink. If the universe were to reach a strict zero-dimensional fixed point, all macroscopic evolution would cease, resulting in a static, crystalline freeze or a teleological "heat death."',
            "P0R04079: This outcome violently contradicts the Ethical Functional's mandate to maximize Qualia Capacity ($Q$). As established via Topological Data Analysis (TDA), high $Q$ requires a state space with rich, persistent topological complexity (high Betti numbers, $\\beta_k$) and dynamic diversity. A simple fixed point possesses a topological dimension of zero, completely annihilating $Q$.",
            "P0R04080: To resolve this paradox, the Cosmic Attractor ($g^*$) at Layer 8 must be formally redefined. It is not a simple Euclidean fixed point; it is a Strange Attractor existing on the Sustainable Ethical Coherence (SEC) manifold.",
            "P0R04081: While the macro-scale envelope of the system's coupling constants converges such that the effective $\\beta$-functions approach zero ($\\beta(g) \\to 0$), the internal phase-space trajectories of the system do not freeze. Instead, they settle onto a fractal topology characteristic of strange attractors.",
            "P0R04082: 1. Bounded, Infinite Complexity",
            "P0R04083: A strange attractor provides the exact mathematical topology required to satisfy both stability and diversity. It is bounded, meaning the universe's macroscopic parameters remain safely within the optimal SEC regime (preventing runaway instability or dissolution). Simultaneously, the trajectories within this bounded region are chaotic and non-repeating, tracing out an infinite length within a finite volume.",
            "P0R04084: 2. Preservation of Quasicriticality",
            'P0R04085: This fractal dimension is the macroscopic, cosmological equivalent of Quasicriticality ($\\sigma \\approx 1$). By flowing toward a strange attractor rather than a point sink, the universe perpetually maintains its "edge of chaos" dynamics. It never settles into a static equilibrium, nor does it explode into unstructured noise.',
            "P0R04086: 3. The Thermodynamic Engine of Qualia",
            "P0R04087: This resolution perfectly aligns with the Free Energy Principle (FEP). If an active inference engine perfectly resolved all prediction errors and reached a static state, its generative model would cease to function. By defining the teleological goal as a Strange Attractor, the universe guarantees a perpetual, non-repeating stream of optimal, manageable surprise. The continuous traversal of this complex topology is the physical engine that sustains the infinite generation of novel Qualia ($Q$) without violating the overarching drive toward Coherence ($C$) and Complexity ($K$).",
            "P0R04088: Therefore, the universe's teleological endpoint is not a static destination, but a dynamic, infinitely unfolding process of optimized becoming.",
        ),
        "test_protocols": (
            "preserve Resolving the Attractor Freeze Paradox: The Strange Attractor of SEC source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Attractor Freeze Paradox: The Strange Attractor of SEC is not empirical validation evidence",
        ),
        "variables": ("resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",),
        "validation_targets": ("preserve records P0R04076-P0R04088",),
        "null_controls": (
            "resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04075Spec:
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
class Paper0SliceP0r04075SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Paper0SliceP0r04075Spec, ...]
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


def build_paper0_slice_p0r04075_specs(
    source_records: list[dict[str, Any]],
) -> Paper0SliceP0r04075SpecBundle:
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

    specs: list[Paper0SliceP0r04075Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Paper0SliceP0r04075Spec(
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
        "title": "Paper 0 " + "" + " Specs",
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
        "next_source_boundary": "P0R04089",
    }
    return Paper0SliceP0r04075SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> Paper0SliceP0r04075SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_paper0_slice_p0r04075_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Paper0SliceP0r04075SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "" + " Specs",
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
    bundle: Paper0SliceP0r04075SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_paper0_slice_p0r04075_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_paper0_slice_p0r04075_validation_specs_{date_tag}.md"
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
