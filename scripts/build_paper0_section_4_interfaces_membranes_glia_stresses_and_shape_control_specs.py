#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4. Interfaces, membranes, glia: stresses and shape control spec builder
"""Promote Paper 0 4. Interfaces, membranes, glia: stresses and shape control records."""

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
    "P0R04754",
    "P0R04755",
    "P0R04756",
    "P0R04757",
    "P0R04758",
    "P0R04759",
    "P0R04760",
    "P0R04761",
    "P0R04762",
    "P0R04763",
    "P0R04764",
    "P0R04765",
    "P0R04766",
    "P0R04767",
    "P0R04768",
)
CLAIM_BOUNDARY = "source-bounded section 4 interfaces membranes glia stresses and shape control source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_interfaces_membranes_glia_stresses_and_shape_control.4_interfaces_membranes_glia_stresses_and_shape_control": {
        "context_id": "4_interfaces_membranes_glia_stresses_and_shape_control",
        "validation_protocol": "paper0.section_4_interfaces_membranes_glia_stresses_and_shape_control.4_interfaces_membranes_glia_stresses_and_shape_control",
        "canonical_statement": "The source-bounded component '4. Interfaces, membranes, glia: stresses and shape control' preserves Paper 0 records P0R04754-P0R04768 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04754:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04755:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04756:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04757:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04758:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04759:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04760:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04761:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04762:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04763:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04764:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04765:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04766:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04767:4_interfaces_membranes_glia_stresses_and_shape_control",
            "P0R04768:4_interfaces_membranes_glia_stresses_and_shape_control",
        ),
        "source_formulae": (
            "P0R04754: 4. Interfaces, membranes, glia: stresses and shape control",
            "P0R04755: Purpose. Provide a first-principles handle on when electromagnetic fields can modulate membrane/interface mechanics and thereby bias morphology, vesicle traffic, or perivascular transport. The treatment is conservative: Maxwell traction plus capillarity, with no exotic assumptions.",
            "P0R04756: Field traction and capillarity. In linear media the Maxwell stress tensor is T_em = (EE E I) + mu(BB B I).",
            "P0R04757: The normal field traction at an interface with unit normal n (pointing medium 1 -> 2) is P_em nT_emn = (E_n E_t) + mu (B_n B_t).",
            'P0R04758: The static curvature balance is the Young-Laplace form with an EM correction, Deltap + DeltaP_em = ,where is twice the mean curvature and Delta denotes the jump 2 1 across . These relations supply the direct "fields -> curvature/pressure" bridge.',
            "P0R04759: Canonical cases (use whichever matches the preparation). Case I: purely tangential fields (E_n = B_n = 0). Then P_em = ( E_t + mu B_t) and the jump is DeltaP_em = [Delta() E_t + Delta(mu) B_t].",
            "P0R04760: Tangential fields therefore reduce the mechanical pressure needed to sustain curvature when > .",
            "P0R04761: Case II: purely normal fields (E_t = B_t = 0). With D_n, B_n continuous across , DeltaP_em = ( E_n,2 E_n,1) + (mu mu) B_n.",
            "P0R04762: Apparent vs intrinsic surface tension. Fields alter the normal-stress balance and can be re-expressed as Deltap = _eff with _eff DeltaP_em/ for a given geometry; the intrinsic is unchanged unless electrochemical work is performed (Lippmann relation d = sigma dV at true electrode interfaces). Biological membranes are typically non-faradaic in our use-cases, so treat DeltaP_em as a curvature-side correction, not a change of itself.",
            "P0R04763: Ripple and vesicle windows. Linearising the normal-stress boundary condition gives a capillary-wave dispersion with an EM correction: (/rho_eff) k + (DeltaP_em/rho_eff) k. The k-term is the field-induced normal stress and sets a frequency-length window where small deformations are amplified or suppressed. This lets you predict whether a given field geometry biases budding, fusion, or tether formation in a specified size range.",
            "P0R04764: Perivascular and glial contexts. In inhomogeneous media the bulk body-force density is f_em = T_em, which augments Navier-Stokes (rho(_t v + vv) = p + v + f_em). Use this to gauge whether field gradients can measurably steer perivascular flows or astrocytic end-foot shears; interfacial balances still reduce to Deltap + DeltaP_em = at boundaries.",
            "P0R04765: Worked scale check (template). For water-air with E_t 1x10 V m and Delta 79 , DeltaP_em 3.5 Pa, which is ~2.4% of the Laplace pressure of a 1 mm droplet (2/R 144 Pa). Replace numbers with membrane-relevant -contrast and curvature to estimate effect sizes in your system.",
            "P0R04766: Operational recipe (how to use this in practice).",
            "P0R04767: Field decomposition at . From your PNP(+R)/electrotonic model or measurement, estimate E_n, E_t (and B if relevant) on each side of the membrane or interface. Use clean-interface boundary data [E_t] = 0, [B_t] = 0, [D_n] = sigma_s, [B_n] = 0. | Compute DeltaP_em. Choose Case I or II (or the mixed formula from P_em above) and evaluate DeltaP_em with your , mu. | Compare to curvature pressure. Form the ratio r |DeltaP_em|/( ||). If r 0.01 in your geometry, expect detectable bias in shape dynamics with modern imaging; if r 0.01, morphology is likely field-insensitive at that scale. (Use your system's , ; do not assume the droplet numbers.) | Choose the observable. For membranes/vesicles: curvature change, budding rate, tether force; for perivascular spaces: shape-mode amplitudes or flow-rate shifts. For electrodes or lipid monolayers, Lippmann electrocapillarity provides an independent cross-check. | Validate with shape metrology. Pendant-drop/bubble methods: (i) fit from Young-Laplace without field; (ii) apply field, compute DeltaP_em from the traction formulae; (iii) refit using Deltap + DeltaP_em = to separate geometry from field traction. For membranes, use flicker spectroscopy or micropipette aspiration analogues.",
            "P0R04768: Tie-in to the affective field A = F. The affective field modulates coupling gains; this section defines the mechanical leg of that pathway. When A perturbs ionic distributions or transmembrane potentials, it moves E_n/E_t and thus DeltaP_em, which in turn alters curvature-dependent processes (vesicle release probability, end-foot geometry, perivascular aperture). The control-theoretic gains you introduced for sigma1 can be extended here by adding DeltaP_em-sensitive terms to the gain scheduling.",
        ),
        "test_protocols": (
            "preserve 4. Interfaces, membranes, glia: stresses and shape control source-accounting boundary",
        ),
        "null_results": (
            "4. Interfaces, membranes, glia: stresses and shape control is not empirical validation evidence",
        ),
        "variables": ("4_interfaces_membranes_glia_stresses_and_shape_control",),
        "validation_targets": ("preserve records P0R04754-P0R04768",),
        "null_controls": (
            "4_interfaces_membranes_glia_stresses_and_shape_control must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section4InterfacesMembranesGliaStressesAndShapeControlSpec:
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
class Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section4InterfacesMembranesGliaStressesAndShapeControlSpec, ...]
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


def build_section_4_interfaces_membranes_glia_stresses_and_shape_control_specs(
    source_records: list[dict[str, Any]],
) -> Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle:
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

    specs: list[Section4InterfacesMembranesGliaStressesAndShapeControlSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section4InterfacesMembranesGliaStressesAndShapeControlSpec(
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
        + "4. Interfaces, membranes, glia: stresses and shape control"
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
        "next_source_boundary": "P0R04769",
    }
    return Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_interfaces_membranes_glia_stresses_and_shape_control_specs(
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


def render_report(bundle: Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4. Interfaces, membranes, glia: stresses and shape control" + " Specs",
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
    bundle: Section4InterfacesMembranesGliaStressesAndShapeControlSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_interfaces_membranes_glia_stresses_and_shape_control_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_interfaces_membranes_glia_stresses_and_shape_control_validation_specs_{date_tag}.md"
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
