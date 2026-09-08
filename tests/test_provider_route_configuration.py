# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Route Configuration Tests
"""Resolve public configuration references against actual adapter declarations."""

import ast
from pathlib import Path

import pytest

from scpn_quantum_control.hardware.aggregators import built_in_aggregator_provider_routes
from scpn_quantum_control.hardware.provider_route_configuration import (
    provider_route_credential_refs,
)


def test_builtin_credential_references_resolve_to_actual_constructor_parameters() -> None:
    """Every route's configuration reference names a real adapter input."""
    root = Path(__file__).resolve().parents[1]
    modules = {route.adapter_module for route in built_in_aggregator_provider_routes()}
    for module in modules:
        refs = provider_route_credential_refs(module)
        assert refs, module
        tree = ast.parse((root / "src" / (module.replace(".", "/") + ".py")).read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for ref in refs:
            module_name, symbol = ref.split(":")
            class_name, method, parameter = symbol.split(".")
            assert module_name == module and method == "__init__"
            constructor = next(
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == method
            )
            assert parameter in {
                arg.arg for arg in (*constructor.args.args, *constructor.args.kwonlyargs)
            }, ref


@pytest.mark.parametrize("module", ["custom.provider", "scpn_quantum_control.hardware.unknown"])
def test_unknown_configuration_is_not_reported_as_credential_free(module: str) -> None:
    """Missing registration stays unknown rather than an empty credential claim."""
    assert provider_route_credential_refs(module) is None


def test_direct_and_broker_authentication_boundaries_stay_distinct() -> None:
    """A brokered IQM route points to broker configuration, not the direct client."""
    prefix = "scpn_quantum_control.hardware."
    assert provider_route_credential_refs(prefix + "hal_iqm") == (
        prefix + "hal_iqm:IQMHALAdapter.__init__.backend",
    )
    assert provider_route_credential_refs(prefix + "hal_qbraid") == (
        prefix + "hal_qbraid:QbraidRuntimeHALAdapter.__init__.device",
        prefix + "hal_qbraid:QbraidRuntimeHALAdapter.__init__.provider",
        prefix + "hal_qbraid:QbraidRuntimeHALAdapter.__init__.provider_factory",
    )
