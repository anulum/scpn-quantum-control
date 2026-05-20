# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware job dossiers
"""Standard documentation schema for submission-ready hardware jobs.

Every candidate hardware job must have a dossier before submission. The dossier
records the scientific question, falsification boundary, expected observables,
budget, platform fit, risks, decision tree, paper impact, follow-up avenue, and
reproducibility package. This keeps QPU usage justified and prevents vague or
post-hoc hardware runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareJobDossier:
    """Submission-readiness dossier for one hardware job."""

    job_id: str
    title: str
    purpose: str
    hypothesis: str
    falsification_condition: str
    expected_observables: tuple[str, ...]
    circuit_summary: Mapping[str, Any]
    qpu_budget: Mapping[str, Any]
    platform_fit: Mapping[str, str]
    risks_and_confounds: tuple[str, ...]
    decision_tree: Mapping[str, str]
    paper_impact: str
    follow_up_avenue: str
    possibilities_opened: tuple[str, ...]
    claim_boundary: str
    reproducibility_package: Mapping[str, str]
    prerequisites: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_text(self.job_id, "job_id")
        _require_text(self.title, "title")
        _require_text(self.purpose, "purpose")
        _require_text(self.hypothesis, "hypothesis")
        _require_text(self.falsification_condition, "falsification_condition")
        _require_non_empty_tuple(self.expected_observables, "expected_observables")
        _require_non_empty_mapping(self.circuit_summary, "circuit_summary")
        _require_non_empty_mapping(self.qpu_budget, "qpu_budget")
        _require_non_empty_mapping(self.platform_fit, "platform_fit")
        _require_non_empty_tuple(self.risks_and_confounds, "risks_and_confounds")
        _require_non_empty_mapping(self.decision_tree, "decision_tree")
        _require_text(self.paper_impact, "paper_impact")
        _require_text(self.follow_up_avenue, "follow_up_avenue")
        _require_non_empty_tuple(self.possibilities_opened, "possibilities_opened")
        _require_text(self.claim_boundary, "claim_boundary")
        _require_non_empty_mapping(self.reproducibility_package, "reproducibility_package")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the dossier for preregistration manifests."""
        return {
            "job_id": self.job_id,
            "title": self.title,
            "purpose": self.purpose,
            "hypothesis": self.hypothesis,
            "falsification_condition": self.falsification_condition,
            "expected_observables": list(self.expected_observables),
            "circuit_summary": dict(self.circuit_summary),
            "qpu_budget": dict(self.qpu_budget),
            "platform_fit": dict(self.platform_fit),
            "risks_and_confounds": list(self.risks_and_confounds),
            "decision_tree": dict(self.decision_tree),
            "paper_impact": self.paper_impact,
            "follow_up_avenue": self.follow_up_avenue,
            "possibilities_opened": list(self.possibilities_opened),
            "claim_boundary": self.claim_boundary,
            "reproducibility_package": dict(self.reproducibility_package),
            "prerequisites": list(self.prerequisites),
        }

    def to_markdown(self) -> str:
        """Render the dossier as a human-reviewable Markdown section."""
        lines = [
            f"# {self.title}",
            "",
            f"Job ID: `{self.job_id}`",
            "",
            "## Purpose",
            self.purpose,
            "",
            "## Hypothesis",
            self.hypothesis,
            "",
            "## Falsification Condition",
            self.falsification_condition,
            "",
            "## Expected Observables",
            *_bullet_lines(self.expected_observables),
            "",
            "## Circuit / Package Summary",
            *_mapping_lines(self.circuit_summary),
            "",
            "## QPU Budget",
            *_mapping_lines(self.qpu_budget),
            "",
            "## Platform Fit",
            *_mapping_lines(self.platform_fit),
            "",
            "## Risks and Confounds",
            *_bullet_lines(self.risks_and_confounds),
            "",
            "## Decision Tree",
            *_mapping_lines(self.decision_tree),
            "",
            "## Paper Impact",
            self.paper_impact,
            "",
            "## Follow-up Avenue",
            self.follow_up_avenue,
            "",
            "## Possibilities Opened",
            *_bullet_lines(self.possibilities_opened),
            "",
            "## Claim Boundary",
            self.claim_boundary,
            "",
            "## Reproducibility Package",
            *_mapping_lines(self.reproducibility_package),
        ]
        if self.prerequisites:
            lines.extend(["", "## Prerequisites", *_bullet_lines(self.prerequisites)])
        return "\n".join(lines) + "\n"


def build_s1_feedback_job_dossier(
    *,
    circuit_summary: Mapping[str, Any],
    qpu_budget: Mapping[str, Any],
    ready_platforms: Sequence[str],
    manual_review_platforms: Sequence[str],
) -> HardwareJobDossier:
    """Build the standard dossier for the S1 monitored-feedback job."""
    return HardwareJobDossier(
        job_id="s1_dynamic_feedback_readiness",
        title="S1 monitored Kuramoto-XY feedback dynamic-circuit run",
        purpose=(
            "Test whether a monitored cross-shot feedback policy can steer the "
            "Kuramoto-XY synchronisation observable on hardware under a bounded "
            "dynamic-circuit payload."
        ),
        hypothesis=(
            "If the feedback loop survives hardware noise and provider-side "
            "dynamic-circuit execution, the observed live order parameter should "
            "move toward the preregistered target more often than a matched "
            "open-loop control at the same circuit family, shots, and layout."
        ),
        falsification_condition=(
            "The feedback arm fails if it does not improve the target-order-"
            "parameter error relative to the matched open-loop control, or if "
            "transpilation/readout/latency overhead dominates so strongly that "
            "the feedback action is statistically indistinguishable from noise."
        ),
        expected_observables=(
            "live finite-shot Kuramoto order parameter R_live",
            "statevector/simulator reference R where available",
            "feedback action sequence",
            "applied and next coupling-scale sequence",
            "readout-count distribution per round",
            "target-error reduction against matched open-loop control",
        ),
        circuit_summary=circuit_summary,
        qpu_budget=qpu_budget,
        platform_fit={
            "ready": ", ".join(ready_platforms) if ready_platforms else "none",
            "manual_review": ", ".join(manual_review_platforms)
            if manual_review_platforms
            else "none",
        },
        risks_and_confounds=(
            "Python-level feedback is only cross-shot/cross-circuit; intra-shot feedback must be provider-side.",
            "Dynamic-circuit transpilation may add depth and conditional-control overhead.",
            "Readout asymmetry and reset errors can mimic feedback success or failure.",
            "Backend calibration drift can change the open-loop and feedback arms differently.",
            "Analogue/native XY platforms require a separate formulation rather than this dynamic-circuit payload.",
        ),
        decision_tree={
            "positive": (
                "Promote to a replicated feedback-vs-open-loop hardware study and "
                "prepare a paper section on hardware-observed synchronisation steering."
            ),
            "null": (
                "Report as a bounded negative/no-effect result and inspect readout, "
                "layout, and conditional-control overhead before any larger run."
            ),
            "negative": (
                "Treat dynamic feedback as harmful for this circuit family and pivot "
                "toward analogue/native XY or offline adaptive-design variants."
            ),
            "contradictory": (
                "Run only preregistered diagnostics: layout replay, readout-mitigation "
                "cross-check, and matched simulator/noise-model comparison."
            ),
        },
        paper_impact=(
            "Would update the S1 feedback documentation and, if hardware data are "
            "accepted, seed a dedicated hybrid-feedback paper or a follow-up section "
            "in the software-methods series."
        ),
        follow_up_avenue=(
            "A positive result opens replicated backend studies and adaptive feedback "
            "policy learning; a negative result motivates native analogue feedback or "
            "provider-side lower-latency dynamic-circuit variants."
        ),
        possibilities_opened=(
            "feedback-stabilised synchronisation witnesses",
            "hardware-calibrated adaptive Kuramoto control",
            "comparison of dynamic-circuit and analogue-native feedback platforms",
            "budget-justified QPU requests for feedback control rather than exploratory runs",
        ),
        claim_boundary=(
            "This job cannot prove sub-microsecond real-time control unless feedback is "
            "implemented provider-side, cannot establish quantum advantage, and cannot "
            "generalise beyond the tested backend, layout, circuit family, and calibration window."
        ),
        reproducibility_package={
            "preregistration_manifest": "data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json",
            "raw_counts_path": "data/s1_feedback_loop/raw_counts/",
            "analysis_script": "scripts/analyse_s1_feedback_hardware.py",
            "latency_benchmark": "data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json",
            "claim_boundary_doc": "docs/campaigns/hybrid_feedback_loop_s1_2026-05-06.md",
        },
        prerequisites=(
            "export preregistration manifest before any submission",
            "confirm backend supports required dynamic-circuit operations",
            "record live transpiled depth and operation counts",
            "obtain explicit QPU-budget approval",
        ),
    )


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_non_empty_tuple(value: tuple[str, ...], name: str) -> None:
    if not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty text entries")


def _require_non_empty_mapping(value: Mapping[str, Any], name: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _bullet_lines(values: Sequence[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _mapping_lines(values: Mapping[str, Any]) -> list[str]:
    return [f"- `{key}`: {value}" for key, value in values.items()]
