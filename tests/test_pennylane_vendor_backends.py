# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cross-vendor PennyLane backend tests (mocked)
"""Cross-vendor backend tests for the PennyLane adapter.

The docstring of ``pennylane_adapter`` advertises IBM, IonQ, Rigetti,
Quantinuum, Braket, Cirq and Xanadu (photonic). The existing
``test_coverage_100_pennylane_mock`` only exercises ``default.qubit``;
``test_pennylane_adapter`` is skipped when PennyLane is absent and only
hits ``default.qubit`` when present.

This module closes audit item B13: for each advertised vendor string we
verify that the adapter:

1. Calls ``qml.device`` with the exact vendor string.
2. Forwards ``wires``, ``shots`` and vendor-specific kwargs verbatim.
3. Preserves ``device_name`` on the resulting dataclass.
4. Does not gate construction behind a hard-coded allow-list (an unknown
   string must still round-trip through ``qml.device`` — the real
   PennyLane plugin system is what rejects unregistered devices).

All tests run against a spy/mock PennyLane; no real hardware is touched
and no vendor plugins are required for CI.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_quantum_control.hardware import pennylane_adapter as pl_mod


class _SpyQml:
    """PennyLane stub that records every ``qml.device`` call."""

    def __init__(self) -> None:
        self.device_calls: list[dict[str, Any]] = []

    def PauliX(self, wire: int) -> MagicMock:
        m = MagicMock(name=f"PauliX({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name="XX")
        return m

    def PauliY(self, wire: int) -> MagicMock:
        m = MagicMock(name=f"PauliY({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name="YY")
        return m

    def PauliZ(self, wire: int) -> MagicMock:
        return MagicMock(name=f"PauliZ({wire})")

    def Hamiltonian(self, coeffs: list[float], ops: list[Any]) -> MagicMock:
        return MagicMock(name="Hamiltonian")

    def device(
        self,
        name: str,
        wires: int | None = None,
        shots: int | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        self.device_calls.append(
            {"name": name, "wires": wires, "shots": shots, "kwargs": dict(kwargs)},
        )
        dev = MagicMock(name=f"device[{name}]")
        dev.short_name = name
        return dev

    def qnode(self, dev: MagicMock) -> Any:
        def decorator(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> float:
                return 0.1

            wrapper.__name__ = fn.__name__
            return wrapper

        return decorator

    def ApproxTimeEvolution(self, H: Any, dt: float, n: int) -> None:
        return None

    def expval(self, op: Any) -> MagicMock:
        return MagicMock(name="expval")


@pytest.fixture()
def spy_pl(monkeypatch: pytest.MonkeyPatch) -> _SpyQml:
    """Replace the adapter's PennyLane handle with a recording spy."""
    qml = _SpyQml()
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", True)
    monkeypatch.setattr(pl_mod, "qml", qml)
    return qml


# ---------------------------------------------------------------------------
# Vendor matrix — one row per advertised backend
# ---------------------------------------------------------------------------
# Each entry: (device_string, shots, vendor_kwargs, human_label)
# ``shots=None`` is analytic (simulator); hardware backends typically need
# a finite shot count, so those cases supply one.

VENDOR_MATRIX: list[tuple[str, int | None, dict[str, Any], str]] = [
    # IBM Quantum — the primary hardware target for Phase 1.
    ("qiskit.ibmq", 4096, {"backend": "ibm_fez"}, "IBM Heron (ibm_fez)"),
    ("qiskit.remote", 4096, {"backend": "ibm_kingston"}, "IBM Heron (ibm_kingston)"),
    # IonQ — trapped-ion.
    ("ionq.simulator", None, {}, "IonQ simulator"),
    ("ionq.qpu", 1024, {"backend": "ionq_aria_1"}, "IonQ Aria"),
    # Rigetti — superconducting, Aspen family.
    # Note: pennylane-rigetti's own device string uses the ``device`` kwarg
    # which collides with our adapter's ``device=`` positional. Keeping the
    # vendor-kwargs empty here — this is a documented limitation of the
    # adapter surface (tracked in docs/falsification.md §C2).
    ("rigetti.qpu", 2048, {}, "Rigetti QPU"),
    # Quantinuum — trapped-ion (ex-Honeywell).
    ("quantinuum.hqs", 500, {"machine": "H1-1"}, "Quantinuum H1-1"),
    # Amazon Braket.
    (
        "braket.aws.qubit",
        1000,
        {"device_arn": "arn:aws:braket:us-west-1::device/qpu/ionq/Aria-1"},
        "Braket · IonQ Aria-1",
    ),
    # Google Cirq simulator (local).
    ("cirq.simulator", None, {}, "Cirq simulator"),
    # PennyLane built-in simulators — regression anchors.
    ("default.qubit", None, {}, "PennyLane default.qubit"),
    ("lightning.qubit", None, {}, "PennyLane lightning.qubit"),
]


def _minimal_inputs(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i + 1] = 0.5
        K[i + 1, i] = 0.5
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


@pytest.mark.parametrize(
    ("device_str", "shots", "kwargs", "label"),
    VENDOR_MATRIX,
    ids=[row[3] for row in VENDOR_MATRIX],
)
class TestVendorBackendConstruction:
    """Each advertised vendor must reach ``qml.device`` verbatim."""

    def test_device_string_forwarded(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        assert len(spy_pl.device_calls) == 1
        call = spy_pl.device_calls[0]
        assert call["name"] == device_str, f"{label}: wrong device string"

    def test_wires_match_n(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs(n=4)
        pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        assert spy_pl.device_calls[0]["wires"] == 4, f"{label}: wires != N"

    def test_shots_forwarded(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        assert spy_pl.device_calls[0]["shots"] == shots

    def test_vendor_kwargs_forwarded(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        assert spy_pl.device_calls[0]["kwargs"] == kwargs, (
            f"{label}: vendor kwargs dropped or mangled"
        )

    def test_device_name_preserved_on_runner(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs()
        runner = pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        assert runner.device_name == device_str

    def test_device_name_propagates_to_result(
        self,
        spy_pl: _SpyQml,
        device_str: str,
        shots: int | None,
        kwargs: dict[str, Any],
        label: str,
    ) -> None:
        K, omega = _minimal_inputs()
        runner = pl_mod.PennyLaneRunner(K, omega, device=device_str, shots=shots, **kwargs)
        result = runner.run_trotter(t=0.1, reps=1)
        assert result.device_name == device_str


# ---------------------------------------------------------------------------
# Negative + pass-through cases
# ---------------------------------------------------------------------------


class TestVendorPassThroughSemantics:
    def test_unknown_vendor_reaches_qml_device_unfiltered(self, spy_pl: _SpyQml) -> None:
        """Adapter must not maintain a hard-coded allow-list of devices.

        Whether a given string is valid is a PennyLane/plugin concern.
        We only assert that we do not gate at the adapter boundary — an
        unknown string still reaches ``qml.device``.
        """
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device="future.vendor.qubit", shots=100)
        assert spy_pl.device_calls[0]["name"] == "future.vendor.qubit"

    def test_exactly_one_device_call_per_runner(self, spy_pl: _SpyQml) -> None:
        """Each ``PennyLaneRunner`` construction issues exactly one
        ``qml.device`` call — regression guard against an accidental
        double-instantiation on hardware, which would double-bill users.
        """
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device="ionq.qpu", shots=100)
        pl_mod.PennyLaneRunner(K, omega, device="ionq.qpu", shots=100)
        assert len(spy_pl.device_calls) == 2

    def test_shots_none_is_distinct_from_shots_zero(self, spy_pl: _SpyQml) -> None:
        """shots=None (analytic) must not be coerced into shots=0 or
        shots=1 when forwarded to PennyLane — some plugins treat 0 as
        'one sample'."""
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(K, omega, device="default.qubit", shots=None)
        assert spy_pl.device_calls[0]["shots"] is None

    def test_hardware_vendor_with_no_kwargs_is_allowed(self, spy_pl: _SpyQml) -> None:
        """Passing a hardware vendor string without vendor-specific kwargs
        must not raise — the PennyLane plugin is responsible for validating
        its own required configuration."""
        K, omega = _minimal_inputs()
        runner = pl_mod.PennyLaneRunner(K, omega, device="rigetti.qpu", shots=512)
        assert runner.device_name == "rigetti.qpu"
        assert spy_pl.device_calls[0]["kwargs"] == {}

    def test_no_hidden_global_state_between_vendors(self, spy_pl: _SpyQml) -> None:
        """Constructing runners for different vendors must not share
        cached device handles — each call is a fresh plugin invocation."""
        K, omega = _minimal_inputs()
        r1 = pl_mod.PennyLaneRunner(K, omega, device="ionq.simulator")
        r2 = pl_mod.PennyLaneRunner(K, omega, device="braket.aws.qubit", shots=100)
        assert r1.dev is not r2.dev
        assert [c["name"] for c in spy_pl.device_calls] == [
            "ionq.simulator",
            "braket.aws.qubit",
        ]


# ---------------------------------------------------------------------------
# Pipeline smoke — full Knm → vendor runner → result, mocked end-to-end
# ---------------------------------------------------------------------------


class TestVendorPipelineSmoke:
    def test_pipeline_braket_end_to_end_mocked(self, spy_pl: _SpyQml) -> None:
        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        runner = pl_mod.PennyLaneRunner(
            K,
            omega,
            device="braket.aws.qubit",
            shots=1000,
            device_arn="arn:aws:braket:us-west-1::device/qpu/ionq/Aria-1",
        )
        result = runner.run_trotter(t=0.5, reps=1)
        assert result.device_name == "braket.aws.qubit"
        assert result.n_qubits == 3
        assert spy_pl.device_calls[0]["kwargs"]["device_arn"].startswith("arn:aws:braket:")

    def test_pipeline_quantinuum_preserves_machine_kwarg(self, spy_pl: _SpyQml) -> None:
        K, omega = _minimal_inputs()
        pl_mod.PennyLaneRunner(
            K,
            omega,
            device="quantinuum.hqs",
            shots=500,
            machine="H1-1",
        )
        assert spy_pl.device_calls[0]["kwargs"] == {"machine": "H1-1"}
