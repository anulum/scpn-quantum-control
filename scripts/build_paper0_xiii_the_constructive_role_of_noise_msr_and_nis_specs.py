#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) spec builder
"""Promote Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) records."""

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
    "P0R06066",
    "P0R06067",
    "P0R06068",
    "P0R06069",
    "P0R06070",
    "P0R06071",
    "P0R06072",
    "P0R06073",
    "P0R06074",
    "P0R06075",
    "P0R06076",
    "P0R06077",
    "P0R06078",
    "P0R06079",
    "P0R06080",
    "P0R06081",
    "P0R06082",
    "P0R06083",
    "P0R06084",
    "P0R06085",
    "P0R06086",
    "P0R06087",
)
CLAIM_BOUNDARY = "source-bounded xiii the constructive role of noise msr and nis source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "xiii_the_constructive_role_of_noise_msr_and_nis.xiii_the_constructive_role_of_noise_msr_and_nis": {
        "context_id": "xiii_the_constructive_role_of_noise_msr_and_nis",
        "validation_protocol": "paper0.xiii_the_constructive_role_of_noise_msr_and_nis.xiii_the_constructive_role_of_noise_msr_and_nis",
        "canonical_statement": "The source-bounded component 'XIII. The Constructive Role of Noise (MSR and NIS)' preserves Paper 0 records P0R06066-P0R06068 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06066:xiii_the_constructive_role_of_noise_msr_and_nis",
            "P0R06067:xiii_the_constructive_role_of_noise_msr_and_nis",
            "P0R06068:xiii_the_constructive_role_of_noise_msr_and_nis",
        ),
        "source_formulae": (
            "P0R06066: XIII. The Constructive Role of Noise (MSR and NIS)",
            "P0R06067: Noise () is a crucial functional component.",
            "P0R06068: Multi-Scale Stochastic Resonance (MSR): SR operates hierarchically, enabling the detection of weak signals below the noise floor. | Noise-Induced Synchronisation (NIS): Common noise input enhances global synchronisation (UPDE order parameter R). | Optimisation: The Psi-field actively modulates noise intensity (D) to optimise information flow and maintain criticality (SOC).",
        ),
        "test_protocols": (
            "preserve XIII. The Constructive Role of Noise (MSR and NIS) source-accounting boundary",
        ),
        "null_results": (
            "XIII. The Constructive Role of Noise (MSR and NIS) is not empirical validation evidence",
        ),
        "variables": ("xiii_the_constructive_role_of_noise_msr_and_nis",),
        "validation_targets": ("preserve records P0R06066-P0R06068",),
        "null_controls": (
            "xiii_the_constructive_role_of_noise_msr_and_nis must remain source-bounded accounting",
        ),
    },
    "xiii_the_constructive_role_of_noise_msr_and_nis.xiv_the_physics_of_information_energy_transduction_iet": {
        "context_id": "xiv_the_physics_of_information_energy_transduction_iet",
        "validation_protocol": "paper0.xiii_the_constructive_role_of_noise_msr_and_nis.xiv_the_physics_of_information_energy_transduction_iet",
        "canonical_statement": "The source-bounded component 'XIV. The Physics of Information-Energy Transduction (IET)' preserves Paper 0 records P0R06069-P0R06072 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06069:xiv_the_physics_of_information_energy_transduction_iet",
            "P0R06070:xiv_the_physics_of_information_energy_transduction_iet",
            "P0R06071:xiv_the_physics_of_information_energy_transduction_iet",
            "P0R06072:xiv_the_physics_of_information_energy_transduction_iet",
        ),
        "source_formulae": (
            "P0R06069: XIV. The Physics of Information-Energy Transduction (IET)",
            "P0R06070: Transduction occurs via the Quantum Potential (Q) (from the Quantum Hamilton-Jacobi equation).",
            "P0R06071: $Q = - 2m\\hslash 2\\rho$$\\ \\nabla 2\\rho$",
            "P0R06072: Q represents informational energy associated with the system's configuration (rho). The Psi-field couples directly to Q: LIET=gIETPsi(x)Q(x). By modulating Q, the Psi-field (Information) influences the energy landscape without classical energy exchange, providing the mechanism for downward causation.",
        ),
        "test_protocols": (
            "preserve XIV. The Physics of Information-Energy Transduction (IET) source-accounting boundary",
        ),
        "null_results": (
            "XIV. The Physics of Information-Energy Transduction (IET) is not empirical validation evidence",
        ),
        "variables": ("xiv_the_physics_of_information_energy_transduction_iet",),
        "validation_targets": ("preserve records P0R06069-P0R06072",),
        "null_controls": (
            "xiv_the_physics_of_information_energy_transduction_iet must remain source-bounded accounting",
        ),
    },
    "xiii_the_constructive_role_of_noise_msr_and_nis.resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst": {
        "context_id": "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
        "validation_protocol": "paper0.xiii_the_constructive_role_of_noise_msr_and_nis.resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
        "canonical_statement": "The source-bounded component 'Resolving the First Law Paradox: The $\\Psi$-Field as an Information Catalyst' preserves Paper 0 records P0R06073-P0R06087 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06073:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06074:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06075:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06076:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06077:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06078:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06079:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06080:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06081:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06082:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06083:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06084:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06085:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06086:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
            "P0R06087:resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
        ),
        "source_formulae": (
            "P0R06073: Resolving the First Law Paradox: The $\\Psi$-Field as an Information Catalyst",
            "P0R06074: P0R06074",
            "P0R06075: The proposition of Information-Energy Transduction (IET) via the Quantum Potential ($Q$) introduces a severe apparent paradox regarding the First Law of Thermodynamics. If the $\\Psi$-field exerts downward causation-for example, by triggering the conformational change of a voltage-gated ion channel or the fusion of a synaptic vesicle-it is effecting physical work ($W > 0$). If the $\\Psi$-field injects novel energy into the biological substrate to accomplish this, it violates the conservation of energy.",
            "P0R06076: To maintain strict physical viability, we must unequivocally state that the $\\Psi$-field is not an energy source; it is an Information Catalyst. During any $\\Psi$-induced event in the biological substrate, the total change in physical energy of the local system must be governed by the First Law:",
            "P0R06077: $$\\Delta E_{\\text{system}} = Q_{\\text{thermal}} - W_{\\text{action}} + E_{\\Psi}$$",
            "P0R06078: To absolutely conserve physical energy, the energetic contribution of the consciousness field must be zero:",
            "P0R06079: $$E_{\\Psi} = 0$$",
            "P0R06080: If the $\\Psi$-field injects no energy, how does the biological mechanism overcome its activation energy barrier ($\\Delta G^\\ddagger$)? The work done to cross the barrier is drawn entirely from the ambient biological heat bath ($k_B T$). The $\\Psi$-field operates via the mechanism of an informational Brownian Ratchet, capitalizing on the high-noise environment of the quasicritical brain.",
            "P0R06081: Through the interaction Lagrangian $L_{IET} = g_{IET} \\Psi(x)Q(x)$, the $\\Psi$-field modulates the informational geometry of the system (the Bohmian quantum potential), not its classical physical potential. It does this by acting as a Maxwell's Demon. When a random thermal fluctuation provides the exact necessary energy $Q_{\\text{thermal}}$ to push the ion channel toward an open state, the $\\Psi$-field uses its informational coupling to selectively lock or stabilize that state (via the Quantum Zeno Effect).",
            "P0R06082: Because the $\\Psi$-field utilizes Quantum Stochastic Resonance (QSR), it merely correlates the timing of the channel's conformational changes with existing random thermal fluctuations. It rectifies undirected thermal noise into directed biological work:",
            "P0R06083: $$W_{\\text{action}} \\le \\eta \\ Q_{\\text{thermal}}$$",
            "P0R06084: The $\\Psi$-field pays for this action not with physical energy, but with information. By Landauer's Principle, the act of the $\\Psi$-field intervening to reduce the local physical entropy of the ion channel's state space ($\\Delta S_{\\text{local}} < 0$) requires the erasure of information within the $\\Psi$-field's own computational space (the HPC generative model). This informational erasure dissipates heat into the biological substrate according to the Landauer limit:",
            "P0R06085: $$Q_{\\text{dissipated}} \\ge T \\Delta S_{\\text{erasure}}$$",
            "P0R06086: Thus, downward causation is revealed to be a purely entropic transaction. The $\\Psi$-field steers physical reality by redistributing probabilities and rectifying ambient thermal energy, performing zero net work and leaving the First Law of Thermodynamics inviolate.",
            "P0R06087: P0R06087",
        ),
        "test_protocols": (
            "preserve Resolving the First Law Paradox: The $\\Psi$-Field as an Information Catalyst source-accounting boundary",
        ),
        "null_results": (
            "Resolving the First Law Paradox: The $\\Psi$-Field as an Information Catalyst is not empirical validation evidence",
        ),
        "variables": ("resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",),
        "validation_targets": ("preserve records P0R06073-P0R06087",),
        "null_controls": (
            "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class XiiiTheConstructiveRoleOfNoiseMsrAndNisSpec:
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
class XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[XiiiTheConstructiveRoleOfNoiseMsrAndNisSpec, ...]
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


def build_xiii_the_constructive_role_of_noise_msr_and_nis_specs(
    source_records: list[dict[str, Any]],
) -> XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle:
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

    specs: list[XiiiTheConstructiveRoleOfNoiseMsrAndNisSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            XiiiTheConstructiveRoleOfNoiseMsrAndNisSpec(
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
        "title": "Paper 0 " + "XIII. The Constructive Role of Noise (MSR and NIS)" + " Specs",
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
        "next_source_boundary": "P0R06088",
    }
    return XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_xiii_the_constructive_role_of_noise_msr_and_nis_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "XIII. The Constructive Role of Noise (MSR and NIS)" + " Specs",
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
    bundle: XiiiTheConstructiveRoleOfNoiseMsrAndNisSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_xiii_the_constructive_role_of_noise_msr_and_nis_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_xiii_the_constructive_role_of_noise_msr_and_nis_validation_specs_{date_tag}.md"
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
