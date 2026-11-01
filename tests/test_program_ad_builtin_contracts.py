# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — builtin contract initialization through public imports
"""Exercise fresh public imports and preserve user-managed primitive transforms."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult

_ROUTES = (
    "scpn_quantum_control.whole_program_ad_api",
    "scpn_quantum_control.differentiable",
    "scpn_quantum_control",
)
_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_C = np.array([0.5, -1.0])


def _objective(p: Any) -> Any:
    """Compose the polynomial, linear solve and scalar objective."""
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    return np.sum(x * x) + 3.0 * np.sum(x)


def _exercise_route(route: str) -> dict[str, object]:
    """Check the complete public program against an independent analytic oracle."""
    api = importlib.import_module(route)
    evaluate = cast(Callable[..., WholeProgramADResult], api.whole_program_value_and_grad)
    hashes = []
    for point in ([0.7, -0.4], [-1.3, 0.25], [2.0, 1.5]):
        p = np.array(point)
        b = p * p + 2.0 * p + _C
        x = np.array([(2.0 * b[0] - b[1]) / 5.0, (-b[0] + 3.0 * b[1]) / 5.0])
        outer = 2.0 * x + 3.0
        gradient = (2.0 * p + 2.0) * np.array(
            [(2.0 * outer[0] - outer[1]) / 5.0, (-outer[0] + 3.0 * outer[1]) / 5.0]
        )
        result = evaluate(_objective, p)
        assert result.value == pytest.approx(float(x @ x + 3.0 * x.sum()), rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(result.gradient, gradient, rtol=1e-12, atol=1e-12)
        assert result.method == "whole_program_ad" and result.program_ir is not None
        serialization = result.program_ir.serialization
        assert json.loads(serialization)["effects"]
        assert len(result.parameter_names) == len(point)
        hashes.append(hashlib.sha256(serialization.encode()).hexdigest())
    assert len(set(hashes)) == 1
    return {
        "hashes": hashes,
        "facade_loaded": "scpn_quantum_control.differentiable" in sys.modules,
    }


@pytest.mark.parametrize("route", _ROUTES)
def test_cold_public_import_executes_complete_program(route: str) -> None:
    """Each fresh interpreter executes without an earlier facade import.

    Parameters
    ----------
    route
        Supported public import route selected for this interpreter.

    """
    script = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "import test_program_ad_builtin_contracts as checks; "
        "report=checks._exercise_route(sys.argv[2]); "
        "checks._exercise_reload(sys.argv[2]); print(json.dumps(report))"
    )
    process = subprocess.run(
        [sys.executable, "-B", "-c", script, str(Path(__file__).parent), route],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["facade_loaded"] == (route != _ROUTES[0])
    assert len(report["hashes"]) == 3


def test_loaded_public_routes_capture_identical_programs() -> None:
    """Public aliases retain identical executable semantics and portable IR."""
    captures = [_exercise_route(route)["hashes"] for route in _ROUTES]
    assert captures[0] == captures[1] == captures[2]
    direct = importlib.import_module(_ROUTES[0])
    facade = importlib.import_module(_ROUTES[1])
    assert direct.whole_program_value_and_grad is facade.whole_program_value_and_grad


def _exercise_reload(owner: str) -> None:
    """Reload a public owner while retaining custom transforms and other families."""
    module = importlib.import_module(owner)
    from scpn_quantum_control.program_ad_registry import DEFAULT_CUSTOM_DERIVATIVE_REGISTRY

    registry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY
    snapshot = registry.transform_snapshot()
    original = next(
        transform
        for identity, transform in snapshot.items()
        if identity.key == "scpn.program_ad.elementwise:multiply@1"
    )
    custom = replace(
        original, derivative_rule=replace(original.derivative_rule, name="user_multiply")
    )
    registry.register_transform(custom, overwrite=True)
    try:
        before_reload = registry.transform_snapshot()
        importlib.reload(module)
        after_reload = registry.transform_snapshot()
        assert before_reload.keys() == after_reload.keys()
        assert all(after_reload[key] is value for key, value in before_reload.items())
        assert registry.require(custom.identity) is custom.derivative_rule
        _exercise_route(_ROUTES[0])
    finally:
        registry.register_transform(original, overwrite=True)


def test_facade_reload_preserves_user_transform() -> None:
    """Keep user transforms across facade reload without replacing API class objects."""
    _exercise_reload(_ROUTES[1])
