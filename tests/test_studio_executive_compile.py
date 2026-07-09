# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive compile handler tests
"""Tests for the read-only XY compile ``compile`` handler."""

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
from scpn_quantum_control.studio.executive_compile import (  # noqa: E402
    COMPILE_VERB,
    CompileActionHandler,
    _as_float,
    _normalise_compile,
    _safe_slug,
)

_NETWORK: dict[str, Any] = {
    "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
    "omega": [-0.1, 0.05, 0.05],
    "time": 0.1,
    "trotter_steps": 1,
    "trotter_order": 1,
}


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(CompileActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=COMPILE_VERB, action_id="compile-3node", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_compile_builds_a_verified_bit_exact_unit() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded"
    outputs = record.result.outputs
    assert outputs["verified"] is True
    assert outputs["n_nodes"] == 3
    assert outputs["recompute_schema"] == "studio.xy-compile-recompute.v1"
    assert outputs["verifiability_mode"] == "recompute"
    assert outputs["exactness_class"] == "bit-exact"
    assert outputs["input_sha256"].startswith("sha256:")
    assert record.script is not None


def test_compile_plan_defaults_backend_read_only() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 4


def test_compile_accepts_declared_rust_backend() -> None:
    plan = preview_action(_request(backend="rust"), registry=_registry())
    assert plan.backend == "rust"


def test_compile_rejects_undeclared_backend() -> None:
    handler = CompileActionHandler()
    contract = resolve_verb_contract(COMPILE_VERB)
    with pytest.raises(ValueError, match="is not declared for the compile verb"):
        handler.plan(_request(backend="abacus"), contract)


def test_generated_compile_script_embeds_digest_and_compiles() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert record.result.outputs["input_sha256"] in source
    assert "build_xy_compile_recompute_unit" in source
    assert "verify_xy_compile_recompute_unit" in source


def test_compile_trotter_order_two_is_accepted() -> None:
    record = run_action(_request(trotter_order=2, trotter_steps=2), registry=_registry())
    assert record.result.outputs["trotter_order"] == 2
    assert record.result.outputs["verified"] is True


# --------------------------------------------------------------------------- #
# _as_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "1", None, float("inf"), float("nan")])
def test_as_float_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float("v", bad)


def test_as_float_accepts_numbers() -> None:
    assert _as_float("v", 2) == 2.0


# --------------------------------------------------------------------------- #
# _safe_slug
# --------------------------------------------------------------------------- #
def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("compile-3node.1") == "compile_3node_1"
    assert _safe_slug("!!!") == "action"


# --------------------------------------------------------------------------- #
# _normalise_compile validation branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"K_nm": "matrix"},
        {"K_nm": [1.0, [0.0, 1.0]]},
        {"K_nm": [[0.0]]},
        {"K_nm": [[0.0] * 17 for _ in range(17)], "omega": [0.0] * 17},
        {"K_nm": [[0.0, 1.0], [1.0, 0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[1.0, 0.0], [0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[0.0, 1.0], [2.0, 0.0]], "omega": [0.0, 0.0]},
        {"omega": "not-a-list"},
        {"omega": [0.1, 0.2]},
        {"time": 0.0},
        {"time": -1.0},
        {"trotter_steps": 0},
        {"trotter_steps": 999},
        {"trotter_steps": True},
        {"trotter_steps": "two"},
        {"trotter_order": 3},
        {"trotter_order": True},
    ],
)
def test_normalise_compile_rejects_invalid(overrides: dict[str, Any]) -> None:
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_compile(parameters)


def test_normalise_compile_accepts_bounded_network() -> None:
    compile_spec = _normalise_compile(_NETWORK)
    assert len(compile_spec["K_nm"]) == 3
    assert compile_spec["trotter_order"] == 1
