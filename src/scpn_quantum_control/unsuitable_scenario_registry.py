# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — unsuitable-scenario + anti-silent-wrong registry
"""Versioned unsuitable-scenario and anti-silent-wrong-gradient registry.

This module is the product surface for unsuitable-scenario (negative-space governance). It
publishes scenarios that must **fail closed** rather than silently produce wrong
gradients, and provides pure probe/lookup APIs so planners, docs, and tests share
one catalogue.

The surface is pure and deterministic. It does not execute gradients, hide bugs,
or invent green support for unknown scenario identifiers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

ScenarioKind = Literal["unsuitable_scenario", "anti_silent_wrong"]
"""High-level classification for a registry entry."""

RefuseOutcome = Literal[
    "raise_value_error",
    "permanent_boundary",
    "fail_closed_plan",
    "unsupported_transform",
]
"""Expected refuse outcome class when the scenario is probed."""

UNSUITABLE_SCENARIO_REGISTRY_SCHEMA: Final[str] = "unsuitable_scenario_registry.v1"
"""JSON schema identifier for serialised registry payloads."""

UNSUITABLE_SCENARIO_CLAIM_BOUNDARY: Final[str] = (
    "unsuitable-scenario and anti-silent-wrong registry only; entries document "
    "explicit refuse paths and competitor failure modes, never invent gradient "
    "success, hardware execution, or silent-tape recovery claims"
)
"""Shared claim boundary attached to every catalogue row and probe result."""


@dataclass(frozen=True, slots=True)
class UnsuitableScenarioRecord:
    """One versioned unsuitable or anti-silent-wrong catalogue entry.

    Attributes
    ----------
    scenario_id
        Stable taxonomy key (for example ``unsuitable:complex.objective``).
    kind
        Scenario family.
    trigger
        Short description of the condition that must refuse.
    expected_outcome
        Refuse outcome class.
    expected_error
        Error or boundary token callers should observe.
    reason
        Non-empty human-readable refusal reason.
    evidence
        Evidence labels or deep-link pointers (not performance claims).
    related_route_ids
        Optional governed route IDs linked to this scenario.
    test_id
        Stable test identifier expected to exercise the refuse path.
    citation
        Optional literature or competitor citation label.
    claim_boundary
        Non-promotional claim boundary string.

    """

    scenario_id: str
    kind: ScenarioKind
    trigger: str
    expected_outcome: RefuseOutcome
    expected_error: str
    reason: str
    evidence: tuple[str, ...]
    related_route_ids: tuple[str, ...] = ()
    test_id: str = ""
    citation: str = ""
    claim_boundary: str = UNSUITABLE_SCENARIO_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue-entry invariants."""
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if self.kind not in {"unsuitable_scenario", "anti_silent_wrong"}:
            raise ValueError(f"unknown scenario kind: {self.kind!r}")
        if self.expected_outcome not in {
            "raise_value_error",
            "permanent_boundary",
            "fail_closed_plan",
            "unsupported_transform",
        }:
            raise ValueError(f"unknown expected_outcome: {self.expected_outcome!r}")
        if not self.trigger or not self.trigger.strip():
            raise ValueError("trigger must be non-empty")
        if not self.expected_error or not self.expected_error.strip():
            raise ValueError("expected_error must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if any(not item or not item.strip() for item in self.evidence):
            raise ValueError("evidence labels must be non-empty strings")
        if any(not item or not item.strip() for item in self.related_route_ids):
            raise ValueError("related_route_ids must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this scenario record."""
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "trigger": self.trigger,
            "expected_outcome": self.expected_outcome,
            "expected_error": self.expected_error,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "related_route_ids": list(self.related_route_ids),
            "test_id": self.test_id,
            "citation": self.citation,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ScenarioProbeResult:
    """Deterministic probe result for one scenario identifier.

    Attributes
    ----------
    scenario_id
        Requested scenario identifier.
    refused
        Always ``True`` for known catalogue entries and unknown/blank fail-closed
        probes — this registry never invents green support.
    selected
        Resolved catalogue row, or a synthetic unknown-boundary row.
    message
        Operator-facing refuse message.
    notes
        Additional deterministic notes.

    """

    scenario_id: str
    refused: bool
    selected: UnsuitableScenarioRecord
    message: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate probe-result invariants."""
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if not self.refused:
            raise ValueError(
                "ScenarioProbeResult.refused must be True; green probes are forbidden"
            )
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe result."""
        return {
            "scenario_id": self.scenario_id,
            "refused": self.refused,
            "selected": self.selected.to_dict(),
            "message": self.message,
            "notes": list(self.notes),
            "claim_boundary": UNSUITABLE_SCENARIO_CLAIM_BOUNDARY,
        }


def _scenario(
    scenario_id: str,
    kind: ScenarioKind,
    trigger: str,
    expected_outcome: RefuseOutcome,
    expected_error: str,
    reason: str,
    *,
    evidence: Sequence[str],
    related_route_ids: Sequence[str] = (),
    test_id: str = "",
    citation: str = "",
) -> UnsuitableScenarioRecord:
    """Build one validated catalogue row."""
    return UnsuitableScenarioRecord(
        scenario_id=scenario_id,
        kind=kind,
        trigger=trigger,
        expected_outcome=expected_outcome,
        expected_error=expected_error,
        reason=reason,
        evidence=tuple(evidence),
        related_route_ids=tuple(related_route_ids),
        test_id=test_id,
        citation=citation,
    )


# Canonical catalogue: every entry is an explicit refuse path (no blanks, no green).
_CANONICAL_SCENARIOS: Final[tuple[UnsuitableScenarioRecord, ...]] = (
    _scenario(
        "unsuitable:complex.objective_without_wirtinger",
        "unsuitable_scenario",
        "Complex-valued objective requested without an explicit Wirtinger contract.",
        "unsupported_transform",
        "ValueError:complex_wirtinger_required",
        (
            "Complex reverse-mode without a Wirtinger contract is refused; silent "
            "real-gradient substitution is forbidden."
        ),
        evidence=("transform_algebra_unsupported_boundary", "unsuitable_scenario_seed"),
        related_route_ids=("transform:unsupported.complex_objective",),
        test_id="test_probe_complex_objective_refuses",
    ),
    _scenario(
        "unsuitable:hardware.gradient_without_ticket",
        "unsuitable_scenario",
        "Live hardware gradient submission without owner-ticket evidence.",
        "permanent_boundary",
        "permanent_boundary:provider.hardware.gradient_live",
        (
            "Live hardware gradients remain owner-ticket gated and never plan as "
            "local supported routes."
        ),
        evidence=("no_submit_default", "provider_hardware_safety_audit"),
        related_route_ids=("provider:hardware.gradient_live",),
        test_id="test_probe_hardware_without_ticket_refuses",
    ),
    _scenario(
        "unsuitable:rust.dynamic_axes_replay",
        "unsuitable_scenario",
        "Dynamic axes / dynamic indexing requested on Rust Program AD static replay.",
        "permanent_boundary",
        "permanent_boundary:rust.program_ad.dynamic_axes",
        (
            "Dynamic axes and indexing remain typed fail-closed; static registry "
            "replay must not silently approximate them."
        ),
        evidence=("dynamic_boundary_fail_closed_audit", "rust_program_ad_parity"),
        related_route_ids=("rust:program_ad.dynamic_axes",),
        test_id="test_probe_rust_dynamic_axes_refuses",
    ),
    _scenario(
        "unsuitable:torch.fullgraph_compile_unregistered",
        "unsuitable_scenario",
        "Registered fullgraph torch.compile lowering for Phase-QNode requested.",
        "fail_closed_plan",
        "implementation_path:adapter.torch.fullgraph_compile",
        (
            "Fullgraph torch.compile lowering is not registered; non-fullgraph "
            "local routes remain the supported path and must not silently upgrade."
        ),
        evidence=("torch_maturity_boundary",),
        related_route_ids=("adapter:torch.fullgraph_compile",),
        test_id="test_probe_torch_fullgraph_refuses",
    ),
    _scenario(
        "unsuitable:pennylane.hardware_plugin_gradient",
        "unsuitable_scenario",
        "Hardware-plugin gradient execution through PennyLane devices.",
        "permanent_boundary",
        "permanent_boundary:adapter.pennylane.hardware_plugin_gradient",
        (
            "Hardware-plugin gradients require owner-ticketed evidence chains and "
            "never silently plan as local supported routes."
        ),
        evidence=("provider_hardware_safety_audit", "pennylane_boundary"),
        related_route_ids=("adapter:pennylane.hardware_plugin_gradient",),
        test_id="test_probe_pennylane_hardware_refuses",
    ),
    _scenario(
        "unsuitable:rl.research_without_preregistration",
        "unsuitable_scenario",
        "RL-adjacent witness or pulse research requested without a preregistration ID.",
        "fail_closed_plan",
        "RLResearchGovernanceError:preregistration_id_missing",
        (
            "RL-adjacent search is disabled by default and requires fixed seeds, "
            "bounded evaluations, deterministic zero-noise evaluation, and a "
            "preregistration ID before local witness discovery can run."
        ),
        evidence=("rl_research_governance", "no_production_control_default"),
        related_route_ids=("research:rl.witness_discovery", "research:rl.pulse_optimisation"),
        test_id="test_probe_rl_without_preregistration_refuses",
    ),
    _scenario(
        "anti_silent:differentiation_interface.compiled_tape",
        "anti_silent_wrong",
        (
            "Value-dependent control flow under DifferentiationInterface.jl ReverseDiff "
            "compiled tapes (competitor silent-wrong class)."
        ),
        "permanent_boundary",
        "permanent_boundary:competitor.di_jl.silent_wrong_grads",
        (
            "Documented competitor failure mode: compiled reverse-mode tapes may "
            "yield silently wrong gradients under value-dependent control flow; "
            "SCPN refuses silent degradation for the same class."
        ),
        evidence=(
            "plan_sota_addendum_8_1",
            "citation:DifferentiationInterface.jl",
            "route_matrix_competitor_fixture",
        ),
        related_route_ids=("competitor:differentiation_interface.silent_wrong_grads",),
        test_id="test_probe_di_jl_anti_silent_fixture",
        citation="DifferentiationInterface.jl ReverseDiff compiled-tape notes",
    ),
    _scenario(
        "anti_silent:catalyst.qjit_vmap_quantum",
        "anti_silent_wrong",
        "Catalyst qjit + jax.vmap over quantum instructions without batching rules.",
        "permanent_boundary",
        "permanent_boundary:compiler.catalyst.qjit_vmap",
        (
            "Catalyst documents missing batching rules for quantum instructions; "
            "vmap-inside-qjit is a permanent competitor boundary, not a silent gap "
            "to fill with invent-green support."
        ),
        evidence=(
            "catalyst_sharp_bits",
            "competitive_baseline",
            "route_matrix_competitor_fixture",
        ),
        related_route_ids=("compiler:catalyst.qjit_vmap",),
        test_id="test_probe_catalyst_vmap_anti_silent_fixture",
        citation="Catalyst sharp bits / batching rules for quantum instructions",
    ),
    _scenario(
        "anti_silent:catalyst.no_broadcast_adaptive_shots",
        "anti_silent_wrong",
        "Adaptive finite-shot trainability dry-run requiring broadcast/vmap on Catalyst.",
        "permanent_boundary",
        "permanent_boundary:competitor.catalyst.no_broadcast_adaptive_shots",
        (
            "Catalyst comparison rows document no-broadcast/no-vmap limitations for "
            "adaptive finite-shot trainability; silent success is refused."
        ),
        evidence=("finite_shot_trainability_boundary", "catalyst_comparison"),
        related_route_ids=("competitor:catalyst.no_broadcast_adaptive_shots",),
        test_id="test_probe_catalyst_adaptive_shots_fixture",
        citation="Catalyst adaptive finite-shot trainability boundary",
    ),
)


def _catalogue_map() -> dict[str, UnsuitableScenarioRecord]:
    """Return the scenario_id → record map for the canonical catalogue."""
    mapping = {row.scenario_id: row for row in _CANONICAL_SCENARIOS}
    if len(mapping) != len(_CANONICAL_SCENARIOS):
        raise RuntimeError("duplicate scenario_id in unsuitable scenario catalogue")
    return mapping


_SCENARIO_BY_ID: Final[Mapping[str, UnsuitableScenarioRecord]] = _catalogue_map()


def list_unsuitable_scenario_ids() -> tuple[str, ...]:
    """Return all canonical scenario identifiers in stable catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered scenario identifiers.

    """
    return tuple(row.scenario_id for row in _CANONICAL_SCENARIOS)


def get_unsuitable_scenario(scenario_id: str) -> UnsuitableScenarioRecord:
    """Return one catalogue row or raise for unknown identifiers.

    Parameters
    ----------
    scenario_id
        Taxonomy key to look up.

    Returns
    -------
    UnsuitableScenarioRecord
        The matching catalogue row.

    Raises
    ------
    ValueError
        If ``scenario_id`` is empty/blank or not present in the catalogue.

    """
    if not scenario_id or not str(scenario_id).strip():
        raise ValueError("scenario_id must be a non-empty string")
    key = str(scenario_id).strip()
    try:
        return _SCENARIO_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown unsuitable scenario_id {key!r}; refuse silent invent-green "
            f"support (known_count={len(_SCENARIO_BY_ID)})"
        ) from exc


def iter_unsuitable_scenarios(
    *,
    kind: ScenarioKind | None = None,
    expected_outcome: RefuseOutcome | None = None,
) -> tuple[UnsuitableScenarioRecord, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    expected_outcome
        Optional refuse-outcome filter.

    Returns
    -------
    tuple[UnsuitableScenarioRecord, ...]
        Matching rows.

    """
    rows: Iterable[UnsuitableScenarioRecord] = _CANONICAL_SCENARIOS
    if kind is not None:
        rows = (row for row in rows if row.kind == kind)
    if expected_outcome is not None:
        rows = (row for row in rows if row.expected_outcome == expected_outcome)
    return tuple(rows)


def build_unsuitable_scenario_registry() -> dict[str, object]:
    """Build the full serialisable unsuitable-scenario registry payload.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue cell (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_SCENARIOS]
    unsuitable = sum(1 for row in _CANONICAL_SCENARIOS if row.kind == "unsuitable_scenario")
    anti_silent = sum(1 for row in _CANONICAL_SCENARIOS if row.kind == "anti_silent_wrong")
    return {
        "schema": UNSUITABLE_SCENARIO_REGISTRY_SCHEMA,
        "claim_boundary": UNSUITABLE_SCENARIO_CLAIM_BOUNDARY,
        "scenario_count": len(rows),
        "unsuitable_scenario_count": unsuitable,
        "anti_silent_wrong_count": anti_silent,
        "blank_entry_count": 0,
        "scenarios": rows,
    }


def _unknown_scenario_record(scenario_id: str) -> UnsuitableScenarioRecord:
    """Synthesise a permanent_boundary record for unknown scenario identifiers."""
    return UnsuitableScenarioRecord(
        scenario_id=f"unknown:{scenario_id}",
        kind="unsuitable_scenario",
        trigger="Unknown scenario identifier requested.",
        expected_outcome="raise_value_error",
        expected_error="ValueError:unknown_scenario_id",
        reason=(
            f"scenario_id {scenario_id!r} is not in the unsuitable catalogue; "
            "blank or invent-green cells are forbidden"
        ),
        evidence=("unsuitable_scenario_registry.unknown_scenario",),
        related_route_ids=(),
        test_id="test_probe_unknown_fail_closed",
    )


def probe_unsuitable_scenario(
    scenario_id: str,
    *,
    unknown_policy: Literal["raise", "boundary"] = "raise",
) -> ScenarioProbeResult:
    """Probe one scenario and return a fail-closed refuse result.

    Known scenarios always refuse with the catalogue reason. Unknown IDs either
    raise (default) or synthesise a permanent-boundary refuse row.

    Parameters
    ----------
    scenario_id
        Taxonomy key to probe.
    unknown_policy
        ``raise`` (default) rejects unknown IDs with :class:`ValueError`.
        ``boundary`` returns a synthetic refuse result for operator inspection.

    Returns
    -------
    ScenarioProbeResult
        Always-refused probe result for known (and boundary-policy unknown) IDs.

    Raises
    ------
    ValueError
        If ``scenario_id`` is blank, or unknown under ``unknown_policy='raise'``,
        or if ``unknown_policy`` is not recognised.

    """
    if not scenario_id or not str(scenario_id).strip():
        raise ValueError("scenario_id must be a non-empty string")
    key = str(scenario_id).strip()
    record = _SCENARIO_BY_ID.get(key)
    notes: list[str] = []
    if record is None:
        if unknown_policy == "raise":
            raise ValueError(
                f"unknown unsuitable scenario_id {key!r}; refuse silent invent-green support"
            )
        if unknown_policy != "boundary":
            raise ValueError(
                f"unknown_policy must be 'raise' or 'boundary' (got {unknown_policy!r})"
            )
        selected = _unknown_scenario_record(key)
        notes.append("unknown_policy=boundary synthesised refuse row")
        return ScenarioProbeResult(
            scenario_id=key,
            refused=True,
            selected=selected,
            message=selected.reason,
            notes=tuple(notes),
        )

    if record.kind == "anti_silent_wrong":
        notes.append("anti-silent-wrong fixture: competitor or silent-tape class")
    if record.related_route_ids:
        notes.append("related_route_ids=" + ",".join(record.related_route_ids))

    message = f"refused:{record.expected_outcome}:{record.expected_error} — {record.reason}"
    return ScenarioProbeResult(
        scenario_id=key,
        refused=True,
        selected=record,
        message=message,
        notes=tuple(notes),
    )


def assert_unsuitable_registry_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry payload contains zero blank or green entries.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_unsuitable_scenario_registry`. When
        omitted, a fresh registry is built.

    Returns
    -------
    dict[str, object]
        The validated payload.

    Raises
    ------
    ValueError
        If blank entries, missing reasons, or count drift are detected.

    """
    registry = dict(payload) if payload is not None else build_unsuitable_scenario_registry()
    scenarios = registry.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("unsuitable scenario registry must contain a non-empty scenarios list")
    blank = 0
    for index, row in enumerate(scenarios):
        if not isinstance(row, Mapping):
            raise ValueError(f"scenario row {index} must be a mapping")
        scenario_id = row.get("scenario_id")
        kind = row.get("kind")
        reason = row.get("reason")
        expected_error = row.get("expected_error")
        if not scenario_id:
            blank += 1
            continue
        if kind not in {"unsuitable_scenario", "anti_silent_wrong"}:
            blank += 1
            continue
        if not reason or not expected_error:
            raise ValueError(f"scenario {scenario_id!r} missing reason or expected_error")
    if blank:
        raise ValueError(
            f"unsuitable scenario registry has {blank} blank or invalid entries; refuse green"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    scenario_count = registry.get("scenario_count", -1)
    if not isinstance(scenario_count, int) or scenario_count != len(scenarios):
        raise ValueError("scenario_count does not match scenarios list length")
    return registry


__all__ = [
    "UNSUITABLE_SCENARIO_CLAIM_BOUNDARY",
    "UNSUITABLE_SCENARIO_REGISTRY_SCHEMA",
    "RefuseOutcome",
    "ScenarioKind",
    "ScenarioProbeResult",
    "UnsuitableScenarioRecord",
    "assert_unsuitable_registry_integrity",
    "build_unsuitable_scenario_registry",
    "get_unsuitable_scenario",
    "iter_unsuitable_scenarios",
    "list_unsuitable_scenario_ids",
    "probe_unsuitable_scenario",
]
