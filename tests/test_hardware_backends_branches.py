# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the backend registry and HAL descriptors
"""Branch and guard tests for the backend registry and HAL descriptor helpers.

Covers the descriptor type guard, the auto-discovery wrapper, the unknown HAL
profile lookup, the BackendProfile type guards, the local-adapter and
cost-estimate descriptor branches, and the built-in backend import-availability
fallbacks.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from scpn_quantum_control.hardware import backends as be
from scpn_quantum_control.hardware.hal import BackendCapabilities, BackendProfile


def _capabilities(*, supports_cost_estimate: bool = False) -> BackendCapabilities:
    return BackendCapabilities(
        supports_shots=True,
        supports_counts=True,
        supports_statevector=False,
        supports_mid_circuit_measurement=False,
        supports_analog=False,
        supports_pulse=False,
        supports_cost_estimate=supports_cost_estimate,
    )


def _local_profile(*, supports_cost_estimate: bool = False) -> BackendProfile:
    return BackendProfile(
        backend_id="local_qiskit_aer",
        provider="test_provider",
        broker="direct",
        modality="gate_model",
        sdk_package="test-sdk",
        ir_formats=("openqasm3",),
        capabilities=_capabilities(supports_cost_estimate=supports_cost_estimate),
    )


class _BadDescriptorBackend:
    name = "bad_descriptor"

    def is_available(self) -> bool:
        return True

    def descriptor(self) -> Any:
        return "not a descriptor"


def test_describe_backend_rejects_wrong_descriptor_type() -> None:
    """A descriptor() returning the wrong type is rejected."""
    be.register_backend("bad_descriptor", _BadDescriptorBackend)
    try:
        with pytest.raises(be.BackendRegistrationError, match="expected QuantumBackendDescriptor"):
            be.describe_backend("bad_descriptor")
    finally:
        be.unregister_backend("bad_descriptor")


def test_list_quantum_backends_auto_discovers() -> None:
    """The descriptor listing auto-discovers and returns sorted descriptors."""
    descriptors = be.list_quantum_backends()
    names = [descriptor.name for descriptor in descriptors]
    assert names == sorted(names)


def test_describe_hal_backend_profile_unknown_id() -> None:
    """An unknown HAL profile id raises."""
    with pytest.raises(KeyError, match="unknown HAL backend profile"):
        be.describe_hal_backend_profile("does_not_exist")


@pytest.mark.parametrize(
    "helper",
    [
        be._hal_profile_descriptor,
        be._hal_profile_execution_mode,
        be._hal_profile_can_simulate,
        be._hal_profile_capability_tokens,
    ],
)
def test_hal_profile_helpers_reject_non_profile(helper: Any) -> None:
    """Each HAL profile helper rejects a non-profile argument."""
    with pytest.raises(TypeError, match="profile must be a BackendProfile"):
        helper(object())


def test_execution_mode_local_adapter_for_plain_gate_model() -> None:
    """A non-cloud gate-model profile resolves to the local-adapter execution mode."""
    assert be._hal_profile_execution_mode(_local_profile()) == "local_adapter"


def test_capability_tokens_include_cost_estimate() -> None:
    """A profile that advertises cost estimation lists the cost_estimate capability."""
    tokens = be._hal_profile_capability_tokens(_local_profile(supports_cost_estimate=True))
    assert "cost_estimate" in tokens


def test_qiskit_aer_backend_unavailable_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Aer backend reports unavailable when the package cannot be imported."""
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    assert be._QiskitAerBackend().is_available() is False


def test_cirq_backend_unavailable_without_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Cirq backend reports unavailable when its adapter cannot be imported."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_control.hardware.cirq_adapter", None)
    assert be._CirqBackend().is_available() is False


def test_braket_backend_unavailable_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Braket backend reports unavailable when the SDK cannot be imported."""
    monkeypatch.setitem(sys.modules, "braket", None)
    assert be._BraketBackend().is_available() is False
