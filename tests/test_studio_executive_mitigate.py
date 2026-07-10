# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive mitigate handler tests
"""Tests for the zero-noise-extrapolation ``mitigate`` handler."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionRegistry,
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_mitigate import (  # noqa: E402
    MITIGATE_VERB,
    MitigateActionHandler,
    _as_float,
    _as_float_sequence,
    _normalise_mitigate,
    _safe_slug,
)

_NOISE_SCALES = [1.0, 2.0, 3.0]
_EXPECTATIONS = [0.91, 0.83, 0.76]
_ERRORS = [0.01, 0.012, 0.015]


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(MitigateActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters: dict[str, Any] = {
        "noise_scales": _NOISE_SCALES,
        "expectation_values": _EXPECTATIONS,
        "standard_errors": _ERRORS,
    }
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=MITIGATE_VERB, action_id="zne-sweep", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_mitigate_weighted_extrapolation_matches_linear_fit() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["mitigation_schema"] == "studio.mitigation.v1"
    assert outputs["method"] == "wls-delta"
    assert outputs["order"] == 1
    assert outputs["n_points"] == 3
    # The zero-noise estimate must exceed every measured (noise-amplified)
    # value for a monotonically decaying sweep.
    assert outputs["zero_noise_estimate"] > max(_EXPECTATIONS)
    assert outputs["standard_error"] > 0.0
    assert outputs["interval_low"] <= outputs["zero_noise_estimate"] <= outputs["interval_high"]
    assert outputs["interval_width"] == pytest.approx(
        outputs["interval_high"] - outputs["interval_low"]
    )
    assert outputs["coverage"] == 0.95
    assert record.script is not None


def test_mitigate_exact_linear_sweep_recovers_intercept() -> None:
    # y = 1.0 - 0.1 x exactly: the order-1 fit must recover 1.0 to machine
    # precision regardless of weighting.
    record = run_action(
        _request(
            noise_scales=[1.0, 2.0, 4.0],
            expectation_values=[0.9, 0.8, 0.6],
            standard_errors=[0.01, 0.01, 0.01],
        ),
        registry=_registry(),
    )
    assert record.result.status == "succeeded", record.result.error
    assert record.result.outputs["zero_noise_estimate"] == pytest.approx(1.0, abs=1e-12)


def test_mitigate_ordinary_least_squares_without_errors() -> None:
    record = run_action(
        _request(
            noise_scales=[1.0, 1.5, 2.0, 3.0],
            expectation_values=[0.9, 0.85, 0.8, 0.71],
            standard_errors=None,
        ),
        registry=_registry(),
    )
    assert record.result.status == "succeeded", record.result.error
    assert record.result.outputs["method"] == "ols-delta"


def test_mitigate_quadratic_order() -> None:
    record = run_action(
        _request(
            noise_scales=[1.0, 2.0, 3.0, 4.0],
            expectation_values=[0.9, 0.75, 0.55, 0.3],
            standard_errors=[0.01, 0.01, 0.01, 0.01],
            order=2,
        ),
        registry=_registry(),
    )
    assert record.result.status == "succeeded", record.result.error
    assert record.result.outputs["order"] == 2


def test_mitigate_fails_closed_on_underdetermined_ols() -> None:
    record = run_action(
        _request(
            noise_scales=[1.0, 2.0],
            expectation_values=[0.9, 0.8],
            standard_errors=None,
        ),
        registry=_registry(),
    )
    assert record.result.status == "failed"
    assert record.result.error is not None
    assert "order + 2 points" in record.result.error
    assert record.script is None


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_mitigate_plan_defaults_numpy_backend() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "numpy"
    assert plan.requires_approval is False
    assert len(plan.steps) == 5
    assert "weighted least squares" in plan.steps[2]
    assert plan.parameters["order"] == 1
    assert plan.parameters["coverage"] == 0.95


def test_mitigate_plan_names_ols_weighting() -> None:
    plan = preview_action(
        _request(
            noise_scales=[1.0, 2.0, 3.0, 4.0],
            expectation_values=[0.9, 0.8, 0.7, 0.6],
            standard_errors=None,
        ),
        registry=_registry(),
    )
    assert "ordinary least squares" in plan.steps[2]


def test_mitigate_rejects_undeclared_backend() -> None:
    handler = MitigateActionHandler()
    contract = resolve_verb_contract(MITIGATE_VERB)
    with pytest.raises(ValueError, match="is not declared for the mitigate verb"):
        handler.plan(_request(backend="abacus"), contract)


# --------------------------------------------------------------------------- #
# generated script
# --------------------------------------------------------------------------- #
def test_generated_mitigate_script_embeds_estimate_and_compiles() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert record.script.filename == "mitigate_zne_sweep.py"
    assert "zne_extrapolate_with_uncertainty" in source
    assert (
        f"EXPECTED_ZERO_NOISE_ESTIMATE = {record.result.outputs['zero_noise_estimate']!r}"
        in source
    )
    assert f"EXPECTED_METHOD = {record.result.outputs['method']!r}" in source
    assert record.script.digest.startswith("sha256:")


def test_generated_ols_script_embeds_none_errors() -> None:
    record = run_action(
        _request(
            noise_scales=[1.0, 1.5, 2.0, 3.0],
            expectation_values=[0.9, 0.85, 0.8, 0.71],
            standard_errors=None,
        ),
        registry=_registry(),
    )
    assert record.script is not None
    assert "STANDARD_ERRORS = None" in record.script.source


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "one", None, float("nan"), float("inf")])
def test_as_float_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float("v", bad)


@pytest.mark.parametrize("bad", ["text", 1.0, [1.0, "x"], [True]])
def test_as_float_sequence_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float_sequence("v", bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": 1},
        {"noise_scales": [1.0]},
        {"noise_scales": [1.0 + 0.1 * i for i in range(33)]},
        {"noise_scales": [0.5, 2.0, 3.0]},
        {"noise_scales": [1.0, 1.0, 3.0]},
        {"expectation_values": [0.9, 0.8]},
        {"standard_errors": [0.01, 0.012]},
        {"standard_errors": [0.01, 0.012, 0.0]},
        {"order": 3},
        {"order": True},
        {"coverage": 0.0},
        {"coverage": 1.0},
        {"coverage": "wide"},
    ],
)
def test_normalise_mitigate_rejects_invalid(overrides: dict[str, Any]) -> None:
    parameters: dict[str, Any] = {
        "noise_scales": _NOISE_SCALES,
        "expectation_values": _EXPECTATIONS,
        "standard_errors": _ERRORS,
    }
    parameters.update(overrides)
    # A 33-point noise_scales override also invalidates the matching-length
    # checks; every case must raise either way.
    with pytest.raises(ValueError):
        _normalise_mitigate(parameters)


def test_normalise_mitigate_defaults() -> None:
    spec = _normalise_mitigate(
        {"noise_scales": _NOISE_SCALES, "expectation_values": _EXPECTATIONS}
    )
    assert spec["standard_errors"] is None
    assert spec["order"] == 1
    assert spec["coverage"] == 0.95


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("zne-sweep.v1") == "zne_sweep_v1"
    assert _safe_slug("!!!") == "action"
