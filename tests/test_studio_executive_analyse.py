# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive analyse handler tests
"""Tests for the synchronisation-witness ``analyse`` handler."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionRegistry,
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_analyse import (  # noqa: E402
    ANALYSE_VERB,
    AnalyseActionHandler,
    _as_float,
    _as_positive_int,
    _normalise_analyse,
    _safe_slug,
)

_CLOUD: dict[str, Any] = {
    "phases": [0.0, 0.05, -0.04, 0.02],
    "thresholds": [0.0, 0.5, 1.0, 2.0, 3.0],
    "reference_scale": 0.5,
}


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(AnalyseActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters = dict(_CLOUD)
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=ANALYSE_VERB, action_id="analyse-4node", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_analyse_synchronised_cloud_witnesses_one_component() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["n_nodes"] == 4
    assert outputs["analysis_schema"] == "studio.sync-analysis.v1"
    assert outputs["order_parameter"] > 0.99
    assert outputs["persistent_component_count"] == 1
    assert outputs["witness_passed"] is True
    assert outputs["betti0_curve"][0] == 4
    assert record.script is not None


def test_analyse_two_cluster_cloud_counts_two_components() -> None:
    phases = [0.0, 0.01, math.pi, math.pi + 0.01]
    record = run_action(_request(phases=phases, expected_components=2), registry=_registry())
    outputs = record.result.outputs
    assert outputs["persistent_component_count"] == 2
    assert outputs["order_parameter"] < 0.1
    assert outputs["witness_passed"] is True


def test_analyse_loop_cloud_has_dominant_h1_lifetime() -> None:
    phases = list(np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False))
    record = run_action(
        _request(phases=phases, thresholds=[0.0, 0.5, 1.0, 2.0, 3.2], reference_scale=1.0),
        registry=_registry(),
    )
    outputs = record.result.outputs
    assert outputs["order_parameter"] < 1e-9
    assert outputs["dominant_h1_persistence"] > 0.0


def test_analyse_plan_defaults_backend_read_only() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "numpy"
    assert plan.requires_approval is False
    assert len(plan.steps) == 6


def test_analyse_accepts_declared_rust_backend() -> None:
    plan = preview_action(_request(backend="rust"), registry=_registry())
    assert plan.backend == "rust"


def test_analyse_rejects_undeclared_backend() -> None:
    handler = AnalyseActionHandler()
    contract = resolve_verb_contract(ANALYSE_VERB)
    with pytest.raises(ValueError, match="is not declared for the analyse verb"):
        handler.plan(_request(backend="abacus"), contract)


def test_generated_analyse_script_embeds_summary_and_compiles() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert repr(record.result.outputs["order_parameter"]) in source
    assert "phase_cloud_synchronisation_witness" in source


# --------------------------------------------------------------------------- #
# _as_float / _as_positive_int / _safe_slug
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "1", None, float("inf"), float("nan")])
def test_as_float_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float("v", bad)


def test_as_float_accepts_numbers() -> None:
    assert _as_float("v", 2) == 2.0


@pytest.mark.parametrize("bad", [True, "two", 0, 999])
def test_as_positive_int_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_positive_int("v", bad, maximum=32)


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("analyse-4node.1") == "analyse_4node_1"
    assert _safe_slug("!!!") == "action"


# --------------------------------------------------------------------------- #
# _normalise_analyse validation branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"phases": "cloud"},
        {"phases": [0.0]},
        {"phases": [0.0] * 33},
        {"phases": [0.0, "x"]},
        {"thresholds": "grid"},
        {"thresholds": [0.5]},
        {"thresholds": [0.0, 1.0, 0.5]},
        {"thresholds": [0.0, 0.0, 1.0]},
        {"thresholds": [-1.0, 0.0, 1.0], "reference_scale": 0.0},
        {"reference_scale": 5.0},
        {"reference_scale": -1.0},
        {"reference_scale": "mid"},
        {"expected_components": 0},
        {"expected_components": True},
        {"expected_components": 99},
    ],
)
def test_normalise_analyse_rejects_invalid(overrides: dict[str, Any]) -> None:
    parameters = dict(_CLOUD)
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_analyse(parameters)


def test_normalise_analyse_accepts_bounded_cloud() -> None:
    analyse_spec = _normalise_analyse(_CLOUD)
    assert len(analyse_spec["phases"]) == 4
    assert analyse_spec["expected_components"] == 1
    assert analyse_spec["reference_scale"] == 0.5
