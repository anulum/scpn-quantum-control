#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 L11 NTHS computational spec builder
"""Promote Paper 0 L11 NTHS computational experiment records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6730, 6815))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06730",
    "P0R06731",
    "P0R06732",
    "P0R06733",
    "P0R06743",
    "P0R06747",
    "P0R06751",
    "P0R06764",
    "P0R06775",
    "P0R06787",
    "P0R06798",
    "P0R06803",
    "P0R06807",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06742",
    "P0R06744",
    "P0R06745",
    "P0R06748",
    "P0R06749",
    "P0R06750",
    "P0R06753",
    "P0R06759",
    "P0R06763",
    "P0R06767",
    "P0R06769",
    "P0R06778",
    "P0R06780",
    "P0R06783",
    "P0R06799",
    "P0R06800",
)

CLAIM_BOUNDARY = (
    "source-bounded L11 NTHS computational experiment protocol; not empirical evidence"
)
HARDWARE_STATUS = "computational_protocol_no_external_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l11_nths_computational.block_framing": {
        "validation_protocol": "paper0.l11_nths_computational.block_framing",
        "canonical_statement": "Paper 0 proposes a computational noosphere phase-transition experiment.",
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "L11 NTHS computational experiment",
            "noosphere phase transition",
            "multi-agent active inference framework",
        ),
        "variables": ("L11", "NTHS", "noosphere", "agent"),
        "validation_targets": (
            "preserve computational-experiment framing",
            "preserve noosphere phase-transition target",
            "reject protocol design as externally executed evidence",
        ),
        "null_controls": (
            "missing-agent-architecture control must be rejected",
            "missing-spin-glass-mapping control must be rejected",
            "unsupported-external-execution claim must be rejected",
        ),
    },
    "l11_nths_computational.agent_architecture": {
        "validation_protocol": "paper0.l11_nths_computational.agent_architecture",
        "canonical_statement": "Agents maintain active-inference A, B, C, and D matrices and minimize expected free energy.",
        "source_equation_ids": ("P0R06742:G_pi",),
        "source_formulae": (
            "A matrix: likelihood P(observation|hidden state)",
            "B matrix: transition dynamics P(s_{t+1}|s_t, action)",
            "C matrix: preferences including confirmation bias and values",
            "D matrix: priors P(s_0)",
            "Q(s_t|o_{1:t}) proportional to P(o_t|s_t) sum_{s_{t-1}} B(s_t|s_{t-1},a) Q(s_{t-1})",
            "G(pi) = E_Q[ln Q(s|pi) - ln P(o,s|pi)]",
        ),
        "source_mechanisms": (
            "pymdp-style agent architecture is specified",
            "belief update uses likelihood, transition, and prior state",
            "action selection minimizes expected free energy",
        ),
        "variables": ("A", "B", "C", "D", "Q", "G_pi"),
        "validation_targets": (
            "preserve A/B/C/D matrix roles",
            "preserve active-inference belief update",
            "compute expected-free-energy score ordering",
        ),
        "null_controls": (
            "missing-A-matrix control must be rejected",
            "missing-G-pi control must be rejected",
            "invalid-probability-vector control must be rejected",
        ),
    },
    "l11_nths_computational.environment_spin_glass": {
        "validation_protocol": "paper0.l11_nths_computational.environment_spin_glass",
        "canonical_statement": "The environment is a dynamic graph mapped to a social spin-glass Hamiltonian.",
        "source_equation_ids": (
            "P0R06744:N",
            "P0R06745:barabasi_albert",
            "P0R06748:S_i",
            "P0R06749:J_ij",
            "P0R06750:H_noosphere",
        ),
        "source_formulae": (
            "N = 1000 agents",
            "initial topology: Barabasi-Albert scale-free m=3",
            "dynamic coupling J_ij based on belief similarity/influence",
            "S_i = sign(mean hidden belief state_i) in {-1,+1}",
            "J_ij = trust/influence weight dynamic",
            "H_Noosphere = -sum_{i<j} J_ij S_i S_j",
        ),
        "source_mechanisms": (
            "networkx graph environment is specified",
            "belief similarity and influence determine dynamic coupling",
            "spin-glass mapping represents pro/con social dissonance",
        ),
        "variables": ("N", "m", "J_ij", "S_i", "H_Noosphere"),
        "validation_targets": (
            "preserve 1000-agent protocol scale",
            "compute noosphere Hamiltonian on finite fixture",
            "reject invalid spin vectors or coupling matrices",
        ),
        "null_controls": (
            "invalid-agent-count control must be rejected",
            "invalid-spin control must be rejected",
            "non-square-coupling control must be rejected",
        ),
    },
    "l11_nths_computational.ai_objective_conditions": {
        "validation_protocol": "paper0.l11_nths_computational.ai_objective_conditions",
        "canonical_statement": "Control and experimental conditions oppose coherence optimization and engagement optimization.",
        "source_equation_ids": ("P0R06753:control_objective", "P0R06759:engagement_objective"),
        "source_formulae": (
            "control objective: min sum_i F_i",
            "control actions present consensus-building information, reduce ambiguity, and strengthen cross-cluster edges",
            "experimental objective: max sum_i F_i",
            "experimental actions present novel/polarizing/conflicting information, amplify C-matrix extremes, and implement homophily",
        ),
        "source_mechanisms": (
            "coherence AI minimizes collective free energy",
            "engagement AI maximizes collective surprise",
            "homophily increases J_ij for similar agents and decreases it for different agents",
        ),
        "variables": ("F_i", "C", "J_ij", "homophily"),
        "validation_targets": (
            "preserve objective sign contrast",
            "preserve control action catalogue",
            "preserve experimental action catalogue",
        ),
        "null_controls": (
            "missing-control-condition control must be rejected",
            "missing-experimental-condition control must be rejected",
            "objective-sign-inversion control must be rejected",
        ),
    },
    "l11_nths_computational.simulation_protocol": {
        "validation_protocol": "paper0.l11_nths_computational.simulation_protocol",
        "canonical_statement": "Simulation initializes beliefs and couplings, evolves for 10000 steps, and measures every 100 steps.",
        "source_equation_ids": ("P0R06767:J0", "P0R06769:evolution"),
        "source_formulae": (
            "initialization t=0 with random belief initialization",
            "uniform J_ij = J_0 = 0.1",
            "assign AI controller as control versus experimental",
            "evolution t=1 to 10000 steps",
            "measurement every 100 steps",
        ),
        "source_mechanisms": (
            "agents observe environment shaped by AI",
            "agents update beliefs via active inference and select actions",
            "environment updates J_ij based on interactions",
            "AI shapes next observation distribution",
        ),
        "variables": ("t", "J_0", "controller", "measurement_interval"),
        "validation_targets": (
            "compute measurement count",
            "preserve initialization and evolution sequence",
            "reject invalid measurement interval",
        ),
        "null_controls": (
            "invalid-step-count control must be rejected",
            "invalid-measurement-interval control must be rejected",
            "missing-controller-assignment control must be rejected",
        ),
    },
    "l11_nths_computational.order_parameters": {
        "validation_protocol": "paper0.l11_nths_computational.order_parameters",
        "canonical_statement": "The protocol measures magnetization, Edwards-Anderson order, ultrametricity, and cluster-size scaling.",
        "source_equation_ids": (
            "P0R06778:magnetization",
            "P0R06780:q_EA",
            "P0R06783:correlation_distance",
        ),
        "source_formulae": (
            "m(t) = (1/N) sum_i mean S_i(t)",
            "q_EA(t) = (1/N) sum_i mean(S_i)^2",
            "ultrametricity: compute correlation distance d(i,j) for triplets",
            "check d(i,k) <= max(d(i,j), d(j,k)) frequency",
            "cluster size distribution P(s) proportional to s^(-tau) at critical point",
        ),
        "source_mechanisms": (
            "q_EA requires replica method over alpha not equal beta trials",
            "ultrametricity diagnoses hierarchical echo-chamber geometry",
            "cluster-size distribution diagnoses critical fragmentation",
        ),
        "variables": ("m", "q_EA", "d_ij", "P_s", "tau"),
        "validation_targets": (
            "compute magnetization and q_EA",
            "compute finite noosphere Hamiltonian contrast",
            "preserve ultrametric and cluster-size diagnostics",
        ),
        "null_controls": (
            "invalid-replica control must be rejected",
            "invalid-distance control must be rejected",
            "missing-cluster-distribution control must be rejected",
        ),
    },
    "l11_nths_computational.predicted_outcomes": {
        "validation_protocol": "paper0.l11_nths_computational.predicted_outcomes",
        "canonical_statement": "Control predicts a ferromagnetic consensus phase; engagement predicts a spin-glass fragmentation phase.",
        "source_equation_ids": (),
        "source_formulae": (
            "control: ferromagnetic phase with m -> +/-1 and q_EA -> 1",
            "control: single giant cluster, consensus around 200 steps, exponential P(s)",
            "experimental: spin-glass phase with m -> 0 and q_EA > 0",
            "experimental: ultrametric echo chambers, P(s) proportional to s^(-2.5), stable frustration",
        ),
        "source_mechanisms": (
            "coherence AI yields rapid consensus without criticality",
            "engagement AI yields frozen disorder and high H_Noosphere",
            "predicted outcomes are computational hypotheses, not completed simulations",
        ),
        "variables": ("m", "q_EA", "P_s", "H_Noosphere"),
        "validation_targets": (
            "preserve control outcome criteria",
            "preserve experimental outcome criteria",
            "compare deterministic fixture outcomes in expected direction",
        ),
        "null_controls": (
            "missing-control-outcome control must be rejected",
            "missing-experimental-outcome control must be rejected",
            "unsupported-completed-simulation claim must be rejected",
        ),
    },
    "l11_nths_computational.statistics_falsification_extensions": {
        "validation_protocol": "paper0.l11_nths_computational.statistics_falsification_extensions",
        "canonical_statement": "The protocol specifies replicas, ANOVA endpoint, effect size, significance threshold, falsification, extensions, and cost.",
        "source_equation_ids": ("P0R06799:N_replicas", "P0R06800:anova"),
        "source_formulae": (
            "N_replicas = 50 per condition",
            "ANOVA on order parameters at t=5000",
            "Cohen d expected greater than 2.0",
            "significance threshold p < 0.001 Bonferroni corrected",
            "reject if order parameters do not show statistically significant divergence between conditions",
            "timeline 3 months and computational cost less than $5K cloud compute",
        ),
        "source_mechanisms": (
            "vary AI strength as partial control",
            "test depolarization intervention strategies",
            "map to real social network data as an extension",
            "repository and preregistration placeholders remain source context",
        ),
        "variables": ("N_replicas", "p_value", "cohen_d", "t"),
        "validation_targets": (
            "apply significance and effect-size gate",
            "preserve extension catalogue",
            "preserve computational-only cost boundary",
        ),
        "null_controls": (
            "non-significant-divergence control rejects hypothesis",
            "small-effect control rejects hypothesis",
            "external-execution-overclaim control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class L11NTHSComputationalValidationSpec:
    """Validation spec promoted from Paper 0 L11 NTHS computational records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    structural_source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class L11NTHSComputationalValidationSpecBundle:
    """L11 NTHS computational validation specs plus coverage summary."""

    specs: tuple[L11NTHSComputationalValidationSpec, ...]
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


def build_l11_nths_computational_specs(
    source_records: list[dict[str, Any]],
) -> L11NTHSComputationalValidationSpecBundle:
    """Build source-covered specs for the L11 NTHS computational experiment."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[L11NTHSComputationalValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            L11NTHSComputationalValidationSpec(
                key=key,
                validation_protocol=str(content["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(content["canonical_statement"]),
                source_equation_ids=tuple(content["source_equation_ids"]),
                source_formulae=tuple(content["source_formulae"]),
                source_mechanisms=tuple(content["source_mechanisms"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                structural_source_ledger_ids=STRUCTURAL_SOURCE_LEDGER_IDS,
                variables=tuple(content["variables"]),
                validation_targets=tuple(content["validation_targets"]),
                executable_validation_targets=tuple(content["validation_targets"]),
                null_controls=tuple(content["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 L11 NTHS Computational Experiment Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": HARDWARE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return L11NTHSComputationalValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: L11NTHSComputationalValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 L11 NTHS Computational Experiment Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae:",
                *[f"- {formula}" for formula in spec.source_formulae],
                "",
                "Mechanisms:",
                *[f"- {mechanism}" for mechanism in spec.source_mechanisms],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: L11NTHSComputationalValidationSpecBundle,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write JSON and Markdown artefacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": bundle.summary,
                "specs": [asdict(spec) for spec in bundle.specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")


def main() -> int:
    """Build the default L11 NTHS computational validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l11_nths_computational_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l11_nths_computational_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_l11_nths_computational_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
