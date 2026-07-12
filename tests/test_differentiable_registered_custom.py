# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable registered custom tests
# scpn-quantum-control -- registered custom derivative wrapper tests
"""Tests for extracted registry-backed custom derivative wrappers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable_registered_custom import (
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
)
from scpn_quantum_control.program_ad_registry import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy closeness across registered custom derivative payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def _registry_with_rule() -> tuple[CustomDerivativeRegistry, PrimitiveIdentity]:
    """Return a private registry with one exact quadratic-vector rule."""

    identity = PrimitiveIdentity("scpn.test.registered_custom", "quadratic_vector")
    registry = CustomDerivativeRegistry()
    registry.register(
        identity,
        CustomDerivativeRule(
            name="registered_quadratic_vector",
            value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
            jvp_rule=lambda values, tangent: np.array(
                [
                    tangent[0] * values[1] + values[0] * tangent[1],
                    2.0 * values[0] * tangent[0],
                ],
                dtype=np.float64,
            ),
            vjp_rule=lambda values, cotangent: np.array(
                [
                    cotangent[0] * values[1] + 2.0 * cotangent[1] * values[0],
                    cotangent[0] * values[0],
                ],
                dtype=np.float64,
            ),
            parameter_names=("theta", "phi"),
            trainable=(True, False),
        ),
    )
    return registry, identity


def test_facade_and_package_root_reuse_extracted_registered_wrappers() -> None:
    """Facade and package-root exports should point at the extracted wrappers."""

    assert differentiable.registered_custom_jvp is registered_custom_jvp
    assert differentiable.registered_custom_vjp is registered_custom_vjp
    assert differentiable.registered_custom_jacobian is registered_custom_jacobian
    assert scpn.registered_custom_jvp is registered_custom_jvp
    assert scpn.registered_custom_vjp is registered_custom_vjp
    assert scpn.registered_custom_jacobian is registered_custom_jacobian


def test_registered_custom_wrappers_resolve_private_registry_rules() -> None:
    """Registered wrappers should route through registry-resolved exact rules."""

    registry, identity = _registry_with_rule()

    jvp = registered_custom_jvp(
        identity,
        [2.0, 3.0],
        [0.5, 10.0],
        registry=registry,
    )
    vjp = registered_custom_vjp(
        identity,
        [2.0, 3.0],
        [11.0, 13.0],
        registry=registry,
    )
    jacobian = registered_custom_jacobian(identity, [2.0, 3.0], registry=registry)

    _assert_allclose(jvp, [1.5, 2.0])
    _assert_allclose(vjp.vjp, [85.0, 0.0])
    _assert_allclose(jacobian.value, [6.0, 4.0])
    _assert_allclose(jacobian.jacobian, [[3.0, 0.0], [4.0, 0.0]])
    assert vjp.parameter_names == ("theta", "phi")
    assert jacobian.trainable == (True, False)


def test_registered_custom_wrappers_fail_closed_for_missing_identity() -> None:
    """Registered wrappers should preserve registry missing-rule diagnostics."""

    missing = PrimitiveIdentity("scpn.test.registered_custom", "missing")
    registry = CustomDerivativeRegistry()

    with pytest.raises(ValueError, match="no custom derivative rule registered"):
        registered_custom_jvp(missing, [1.0], [1.0], registry=registry)
    with pytest.raises(ValueError, match="no custom derivative rule registered"):
        registered_custom_vjp(missing, [1.0], [1.0], registry=registry)
    with pytest.raises(ValueError, match="no custom derivative rule registered"):
        registered_custom_jacobian(missing, [1.0], registry=registry)
