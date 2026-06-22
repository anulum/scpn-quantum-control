# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for feedback provider metadata
"""Branch tests for the feedback provider-metadata introspection helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from scpn_quantum_control.hardware.feedback_provider_metadata import (
    _backend_num_qubits,
    _target_has_operation,
)


def test_target_has_operation_detects_listed_operation() -> None:
    """An operation present in the target operation set is detected."""
    target = SimpleNamespace(operation_names=["x", "cx"])
    assert _target_has_operation(target, "cx") is True
    assert _target_has_operation(target, "rz") is False


def test_backend_num_qubits_requires_positive_value() -> None:
    """A backend without a usable qubit count is rejected."""
    backend = cast(Any, SimpleNamespace(num_qubits=0))
    with pytest.raises(ValueError, match="backend num_qubits must be available"):
        _backend_num_qubits(backend)
