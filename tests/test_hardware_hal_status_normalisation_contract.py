# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- status normalisation contract tests
"""Contract tests for adapter status canonicalisation fallback policy."""

from __future__ import annotations

from collections.abc import Callable

from hypothesis import given
from hypothesis import strategies as st

from scpn_quantum_control.hardware import (
    hal_azure,
    hal_braket,
    hal_ionq,
    hal_iqm,
    hal_oqc,
    hal_pasqal,
    hal_qbraid,
    hal_qiskit,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_strangeworks,
)

_EXTENDED_NORMALISERS_STRICT: tuple[Callable[[object], str], ...] = (
    hal_pasqal._normalise_status,
    hal_oqc._normalise_status,
    hal_iqm._normalise_status,
    hal_quantinuum._normalise_status,
)

_EXTENDED_NORMALISERS_OVERRIDABLE: tuple[Callable[..., str], ...] = (
    hal_azure._normalise_status,
    hal_braket._normalise_status,
    hal_qbraid._normalise_status,
    hal_qiskit._normalise_status,
    hal_ionq._normalise_status,
    hal_strangeworks._normalise_status,
)

_BASELINE_NORMALISERS_STRICT: tuple[Callable[[object], str], ...] = (
    hal_quera_bloqade._normalise_status,
)

_EXTENDED_FROZEN_ALIAS_MATRIX: dict[str, tuple[str, ...]] = {
    "completed": (
        "complete",
        "completed",
        "success",
        "succeeded",
    ),
    "running": ("running", "in-progress", "inprogress"),
    "submitted": (
        "initializing",
        "initialising",
        "starting",
        "creating",
        "created",
    ),
    "queued": ("queued", "pending"),
    "cancelled": ("cancelled", "canceled", "aborting", "cancelling", "canceling"),
    "failed": ("failed", "error"),
}

_BASELINE_FROZEN_ALIAS_MATRIX: dict[str, tuple[str, ...]] = {
    "completed": (
        "done",
        "complete",
        "completed",
        "finished",
        "success",
        "succeeded",
    ),
    "running": ("running",),
    "queued": ("queued", "pending"),
    "cancelled": ("cancelled", "canceled"),
    "failed": ("failed", "error"),
}


def test_status_normalisers_reject_raw_unknown_status_leakage() -> None:
    unknown_token = "vendor_future_terminal_state"

    assert hal_pasqal._normalise_status(unknown_token) == "unknown"
    assert hal_oqc._normalise_status(unknown_token) == "unknown"
    assert hal_iqm._normalise_status(unknown_token) == "unknown"
    assert hal_quera_bloqade._normalise_status(unknown_token) == "unknown"
    assert hal_quantinuum._normalise_status(unknown_token) == "unknown"

    assert hal_azure._normalise_status(unknown_token) == "unknown"
    assert hal_braket._normalise_status(unknown_token) == "unknown"
    assert hal_qbraid._normalise_status(unknown_token) == "unknown"
    assert hal_qiskit._normalise_status(unknown_token) == "unknown"
    assert hal_ionq._normalise_status(unknown_token) == "unknown"
    assert hal_strangeworks._normalise_status(unknown_token) == "unknown"


def test_status_normalisers_honour_explicit_default_overrides() -> None:
    unknown_token = "vendor_status_v2"

    assert hal_azure._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_braket._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_qbraid._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_qiskit._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_ionq._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_strangeworks._normalise_status(unknown_token, default="queued") == "queued"


def test_status_normalisers_canonicalise_whitespace_and_spacing() -> None:
    completion_tokens = ("  COMPLETED  ", "job status.completed")

    for token in completion_tokens:
        assert hal_pasqal._normalise_status(token) == "completed"
        assert hal_oqc._normalise_status(token) == "completed"
        assert hal_iqm._normalise_status(token) == "completed"
        assert hal_quera_bloqade._normalise_status(token) == "completed"
        assert hal_quantinuum._normalise_status(token) == "completed"
        assert hal_azure._normalise_status(token) == "completed"
        assert hal_braket._normalise_status(token) == "completed"
        assert hal_qbraid._normalise_status(token) == "completed"
        assert hal_qiskit._normalise_status(token) == "completed"
        assert hal_ionq._normalise_status(token) == "completed"
        assert hal_strangeworks._normalise_status(token) == "completed"


def test_status_normalisers_map_failure_and_cancellation_aliases() -> None:
    failure_tokens = ("FAILED", "error")
    cancellation_tokens = ("CANCELED", "cancelled")

    for token in failure_tokens:
        assert hal_pasqal._normalise_status(token) == "failed"
        assert hal_oqc._normalise_status(token) == "failed"
        assert hal_iqm._normalise_status(token) == "failed"
        assert hal_quera_bloqade._normalise_status(token) == "failed"
        assert hal_quantinuum._normalise_status(token) == "failed"
        assert hal_azure._normalise_status(token) == "failed"
        assert hal_braket._normalise_status(token) == "failed"
        assert hal_qbraid._normalise_status(token) == "failed"
        assert hal_qiskit._normalise_status(token) == "failed"
        assert hal_ionq._normalise_status(token) == "failed"
        assert hal_strangeworks._normalise_status(token) == "failed"

    for token in cancellation_tokens:
        assert hal_pasqal._normalise_status(token) == "cancelled"
        assert hal_oqc._normalise_status(token) == "cancelled"
        assert hal_iqm._normalise_status(token) == "cancelled"
        assert hal_quera_bloqade._normalise_status(token) == "cancelled"
        assert hal_quantinuum._normalise_status(token) == "cancelled"
        assert hal_azure._normalise_status(token) == "cancelled"
        assert hal_braket._normalise_status(token) == "cancelled"
        assert hal_qbraid._normalise_status(token) == "cancelled"
        assert hal_qiskit._normalise_status(token) == "cancelled"
        assert hal_ionq._normalise_status(token) == "cancelled"
        assert hal_strangeworks._normalise_status(token) == "cancelled"


def test_status_normalisers_map_running_and_queue_aliases() -> None:
    running_tokens = ("RUNNING", "running")
    queued_tokens = ("QUEUED", "queued", "PENDING", "pending")

    for token in running_tokens:
        assert hal_pasqal._normalise_status(token) == "running"
        assert hal_oqc._normalise_status(token) == "running"
        assert hal_iqm._normalise_status(token) == "running"
        assert hal_quera_bloqade._normalise_status(token) == "running"
        assert hal_quantinuum._normalise_status(token) == "running"
        assert hal_azure._normalise_status(token) == "running"
        assert hal_braket._normalise_status(token) == "running"
        assert hal_qbraid._normalise_status(token) == "running"
        assert hal_qiskit._normalise_status(token) == "running"
        assert hal_ionq._normalise_status(token) == "running"
        assert hal_strangeworks._normalise_status(token) == "running"

    for token in queued_tokens:
        assert hal_pasqal._normalise_status(token) == "queued"
        assert hal_oqc._normalise_status(token) == "queued"
        assert hal_iqm._normalise_status(token) == "queued"
        assert hal_quera_bloqade._normalise_status(token) == "queued"
        assert hal_quantinuum._normalise_status(token) == "queued"
        assert hal_azure._normalise_status(token) == "queued"
        assert hal_braket._normalise_status(token) == "queued"
        assert hal_qbraid._normalise_status(token) == "queued"
        assert hal_qiskit._normalise_status(token) == "queued"
        assert hal_ionq._normalise_status(token) == "queued"
        assert hal_strangeworks._normalise_status(token) == "queued"


def test_status_alias_matrix_is_frozen_across_extended_adapters() -> None:
    """Freeze accepted status aliases so drift requires explicit contract edits."""

    for canonical, aliases in _EXTENDED_FROZEN_ALIAS_MATRIX.items():
        for alias in aliases:
            for normalise in _EXTENDED_NORMALISERS_STRICT:
                assert normalise(alias) == canonical
                assert normalise(alias.upper()) == canonical

            for normalise in _EXTENDED_NORMALISERS_OVERRIDABLE:
                assert normalise(alias) == canonical
                assert normalise(alias.upper()) == canonical


def test_status_alias_matrix_is_frozen_for_baseline_adapters() -> None:
    for canonical, aliases in _BASELINE_FROZEN_ALIAS_MATRIX.items():
        for alias in aliases:
            for normalise in _BASELINE_NORMALISERS_STRICT:
                assert normalise(alias) == canonical
                assert normalise(alias.upper()) == canonical


@given(
    token=st.sampled_from(
        tuple(raw for aliases in _EXTENDED_FROZEN_ALIAS_MATRIX.values() for raw in aliases)
    ),
    left_pad=st.text(alphabet=" \t", min_size=0, max_size=2),
    right_pad=st.text(alphabet=" \t", min_size=0, max_size=2),
    dot_prefix=st.booleans(),
)
def test_extended_status_normalisers_property_whitespace_case_and_dot_prefix(
    token: str,
    left_pad: str,
    right_pad: str,
    dot_prefix: bool,
) -> None:
    """Property-style canonicalisation proof for common provider formatting drift."""

    expected = next(
        canonical
        for canonical, aliases in _EXTENDED_FROZEN_ALIAS_MATRIX.items()
        if token in aliases
    )
    candidate = f"{left_pad}{token.upper()}{right_pad}"
    if dot_prefix:
        candidate = f"provider.status.{candidate}"

    for normalise in _EXTENDED_NORMALISERS_STRICT:
        assert normalise(candidate) == expected

    for normalise in _EXTENDED_NORMALISERS_OVERRIDABLE:
        assert normalise(candidate) == expected
