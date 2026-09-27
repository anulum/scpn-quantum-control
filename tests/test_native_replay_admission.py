# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Mandatory native replay input admission tests
"""Use installed PyO3 replay entry points for conversion refusal and recovery."""

import json
import sys
from importlib import import_module
from threading import Event
from time import monotonic, sleep
from types import FrameType

import numpy as np
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    ExecutionMemoryReservation,
    active_reserved_bytes,
    reserve_execution_memory,
)

IR = json.dumps(
    {
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [
            {
                "name": "%0",
                "producer": 0,
                "version": 0,
                "shape": [],
                "dtype": "float64",
                "effect": 0,
            },
            {
                "name": "%1",
                "producer": 1,
                "version": 0,
                "shape": [],
                "dtype": "float64",
                "effect": 1,
            },
        ],
        "effects": [
            {
                "index": 0,
                "kind": "parameter",
                "target": "%0",
                "inputs": ["p"],
                "version": 0,
                "ordering": 0,
                "operation": "parameter",
            },
            {
                "index": 1,
                "kind": "pure",
                "target": "%1",
                "inputs": ["%0", "%0"],
                "version": 0,
                "ordering": 1,
                "operation": "mul",
            },
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [0],
    }
)
SURFACES = [
    "program_ad_effect_ir_interpret_forward",
    "program_ad_effect_ir_interpret_value_and_gradient",
]


def test_installed_metadata_parser_refuses_beyond_input_copy_budget_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual PyO3 metadata parsing cannot spend only its source-copy allowance."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", str(8 * len(IR) / 1024**3))
        with pytest.raises(DenseAllocationError):
            engine.program_ad_effect_ir_metadata_summary(IR)
        assert active_reserved_bytes() == baseline
    result = json.loads(engine.program_ad_effect_ir_metadata_summary(IR))
    assert result["format"] == "program_ad_effect_ir.v1"
    assert result["ssa_value_count"] == 2
    assert result["effect_count"] == 2
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", SURFACES)
def test_native_replay_refuses_virtual_inputs_before_extraction_and_recovers(
    surface: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real enormous static sequences refuse under the current environment cap.

    Parameters
    ----------
    surface
        Installed forward or value-and-gradient replay entry point.
    monkeypatch
        Scoped real process memory cap.

    """
    engine = import_module("scpn_quantum_engine")
    callback = getattr(engine, surface)
    baseline = active_reserved_bytes()
    virtual = np.ndarray((100_000_000,), dtype=np.float64, buffer=np.array([2.0]), strides=(0,))
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        for values in (range(100_000_000), virtual, range(sys.maxsize + 1)):
            with pytest.raises(DenseAllocationError):
                callback(IR, values)
            assert active_reserved_bytes() == baseline
        result = json.loads(callback(IR, [2.0]))
        assert result["supported"]
        assert result["value"] == 4.0
        if surface.endswith("value_and_gradient"):
            assert result["gradient"] == [4.0]
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", SURFACES)
def test_native_replay_rejects_opaque_and_malformed_inputs_without_poisoning_owner(
    surface: str,
) -> None:
    """Unsupported sequences and genuine parse failures leave the next replay usable.

    Parameters
    ----------
    surface
        Installed forward or value-and-gradient replay entry point.

    """

    class OpaqueFloat(float):
        def __float__(self) -> float:
            """Expose a forbidden conversion protocol if validation lets it through."""
            raise AssertionError("opaque conversion must not execute")

    callback = getattr(import_module("scpn_quantum_engine"), surface)
    baseline = active_reserved_bytes()
    for invalid in (
        iter([2.0]),
        [object()],
        [OpaqueFloat(2.0)],
        [True],
        np.ones((1, 1)),
        np.array([1j]),
    ):
        with pytest.raises(ValueError):
            callback(IR, invalid)
        assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError):
        callback("{", [2.0])
    for values in ([2.0], [np.float64(2.0)], (2.0,), np.array([2.0]), range(2, 3)):
        result = json.loads(callback(IR, values))
        assert result["supported"]
        assert result["value"] == 4.0
    assert active_reserved_bytes() == baseline


def test_native_replay_metadata_source_cap_and_parent_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata parsing inherits the same source cap and active parent's cancellation.

    Parameters
    ----------
    monkeypatch
        Scoped real process memory cap.

    """
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            engine.program_ad_effect_ir_metadata_summary(" " * 1_000_000)
        with pytest.raises(ValueError):
            engine.program_ad_effect_ir_metadata_summary([IR])
        with pytest.raises(ValueError):
            engine.program_ad_effect_ir_metadata_summary("")
        with pytest.raises(ValueError):
            engine.program_ad_effect_ir_metadata_summary("{")
        assert isinstance(json.loads(engine.program_ad_effect_ir_metadata_summary(IR)), dict)
    cancelled = Event()
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (1,), "float64"),))
    with reserve_execution_memory(plan, cancelled=cancelled):
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            engine.program_ad_effect_ir_interpret_forward(IR, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("fault", ["cancelled", "deadline", "callback-error"])
def test_native_replay_mid_execution_checkpoint_preserves_error_and_recovers(
    surface: str,
    fault: str,
) -> None:
    """Installed native replay observes its real Python checkpoint during execution.

    Parameters
    ----------
    surface
        Installed PyO3 forward or gradient entry point.
    fault
        Cancellation or an injected error from a real native lifecycle callback.

    """
    engine = import_module("scpn_quantum_engine")
    native = getattr(engine, surface)
    baseline_count = 0
    previous = sys.getprofile()

    def count_checkpoints(frame: FrameType, event: str, arg: object) -> None:
        nonlocal baseline_count
        if event == "return" and frame.f_code.co_name == "checkpoint":
            baseline_count += 1

    sys.setprofile(count_checkpoints)
    try:
        baseline_result = json.loads(native(IR, [2.0]))
    finally:
        sys.setprofile(previous)
    assert baseline_result["supported"] is True
    assert baseline_result["value"] == 4.0
    assert baseline_count >= 12
    trigger = baseline_count - 4
    cancelled = Event()
    deadline = monotonic() + 5.0 if fault == "deadline" else None
    observations = 0
    injected = False
    previous = sys.getprofile()
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observations, injected
        if event != "return" or frame.f_code.co_name != "checkpoint":
            return
        observations += 1
        if observations == trigger:
            injected = True
            if fault == "cancelled":
                cancelled.set()
            elif fault == "deadline":
                assert deadline is not None
                sleep(max(0.0, deadline - monotonic()) + 0.001)
            else:
                raise RuntimeError("actual native checkpoint callback failure")

    parent = ExecutionMemoryPlan(
        (ExecutionBuffer("native_lifecycle_parent", "forward", (1,), "float64"),)
    )
    with reserve_execution_memory(parent, cancelled=cancelled, deadline_monotonic=deadline):
        sys.setprofile(profile)
        try:
            error = {
                "cancelled": ExecutionCancelledError,
                "deadline": TimeoutError,
                "callback-error": RuntimeError,
            }[fault]
            message = {
                "cancelled": "execution cancellation observed",
                "deadline": "execution deadline elapsed",
                "callback-error": "actual native checkpoint callback failure",
            }[fault]
            with pytest.raises(error, match=message):
                native(IR, [2.0])
        finally:
            sys.setprofile(previous)
    assert injected
    assert observations >= trigger
    assert active_reserved_bytes() == baseline
    retry = json.loads(native(IR, [2.0]))
    assert retry["supported"] is True
    assert retry["value"] == 4.0
    if surface.endswith("value_and_gradient"):
        assert retry["gradient"] == [4.0]
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("outer_surface", SURFACES)
@pytest.mark.parametrize("inner_surface", SURFACES)
def test_native_reentrant_replay_keeps_parent_exception_instance_and_disposes_both_scopes(
    outer_surface: str,
    inner_surface: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real inherited native callback failure reaches child and parent unchanged.

    Parameters
    ----------
    outer_surface
        Installed native entry point owning the initial lifecycle callback.
    inner_surface
        Installed native entry point called reentrantly from the actual callback.
    monkeypatch
        Restores the instrumented production checkpoint method after injection.

    """

    class ParentCallbackFault(RuntimeError):
        """Distinct actual parent callback error used to assert identity transport."""

    engine = import_module("scpn_quantum_engine")
    outer = getattr(engine, outer_surface)
    inner = getattr(engine, inner_surface)
    original_checkpoint = ExecutionMemoryReservation.checkpoint
    marker = ParentCallbackFault("inherited native parent callback failure")
    stage = "outer"
    native_owner: ExecutionMemoryReservation | None = None
    child_observed = False
    baseline = active_reserved_bytes()

    def observed_checkpoint(reservation: ExecutionMemoryReservation) -> None:
        nonlocal stage, native_owner, child_observed
        original_checkpoint(reservation)
        caller = sys._getframe(1).f_code.co_name
        if stage == "outer" and caller == (
            "test_native_reentrant_replay_keeps_parent_exception_instance_and_disposes_both_scopes"
        ):
            native_owner = reservation
            stage = "child"
            try:
                with pytest.raises(ParentCallbackFault) as child_error:
                    inner(IR, [2.0])
                assert child_error.value is marker
                child_observed = True
            finally:
                stage = "unwinding"
        elif stage == "child" and reservation is native_owner:
            raise marker

    with monkeypatch.context() as instrumentation:
        instrumentation.setattr(ExecutionMemoryReservation, "checkpoint", observed_checkpoint)
        with pytest.raises(ParentCallbackFault) as parent_error:
            outer(IR, [2.0])
        assert parent_error.value is marker
    assert child_observed
    assert active_reserved_bytes() == baseline
    for native in (inner, outer):
        retry = json.loads(native(IR, [2.0]))
        assert retry["supported"] is True
        assert retry["value"] == 4.0
        assert active_reserved_bytes() == baseline


def test_native_replay_retained_broadcast_refuses_before_materialisation_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny input cannot bypass the installed engine's retained-array allowance.

    Parameters
    ----------
    monkeypatch
        Scoped real process cap with no mocked allocator or native implementation.

    """
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    payload = json.loads(IR)
    payload["ssa_values"][1]["shape"] = [100_000_000]
    payload["effects"][1].update(inputs=["%0"], operation="broadcast_to")
    payload["ssa_values"].append(
        {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}
    )
    payload["effects"].append(
        {
            "index": 2,
            "kind": "primitive",
            "target": "%2",
            "inputs": ["%1"],
            "version": 0,
            "ordering": 2,
            "operation": "sum",
        }
    )
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            engine.program_ad_effect_ir_interpret_value_and_gradient(json.dumps(payload), [2.0])
        assert active_reserved_bytes() == baseline
        payload["ssa_values"][1]["shape"] = [3]
        result = json.loads(
            engine.program_ad_effect_ir_interpret_value_and_gradient(json.dumps(payload), [2.0])
        )
        assert result["supported"] is True
        assert result["value"] == 6.0
        assert result["gradient"] == [3.0]
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("outer_surface", SURFACES)
@pytest.mark.parametrize("inner_surface", SURFACES)
def test_reentrant_native_workspace_uses_fixed_input_baseline(
    outer_surface: str, inner_surface: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited callbacks add each cumulative native request to the original input charge."""
    engine = import_module("scpn_quantum_engine")
    outer = getattr(engine, outer_surface)
    inner = getattr(engine, inner_surface)
    original_resize = ExecutionMemoryReservation.resize
    baseline = active_reserved_bytes()
    # Source copies, native numeric copies and boxed input remain the fixed
    # baseline; parser metadata is additional storage throughout each replay.
    input_bytes = len(IR) * 8 + 16 + sys.getsizeof(0.0)

    def isolated_charges(surface: str) -> list[int]:
        owner: ExecutionMemoryReservation | None = None
        measured: list[int] = []

        def observe(reservation: ExecutionMemoryReservation, plan: ExecutionMemoryPlan) -> None:
            nonlocal owner
            if owner is None:
                owner = reservation
                assert reservation.decision.bytes_required == input_bytes
            original_resize(reservation, plan)
            if reservation is owner:
                measured.append(reservation.decision.bytes_required - input_bytes)

        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(ExecutionMemoryReservation, "resize", observe)
            result = json.loads(getattr(engine, surface)(IR, [2.0]))
        assert result["supported"] is True
        assert result["value"] == 4.0
        if surface.endswith("value_and_gradient"):
            assert result["gradient"] == [4.0]
        assert measured and measured[0] > 0
        assert measured == sorted(measured)
        assert active_reserved_bytes() == baseline
        return measured

    outer_increments = isolated_charges(outer_surface)
    inner_increments = isolated_charges(inner_surface)
    native_owner: ExecutionMemoryReservation | None = None
    charges: list[int] = []
    stage = "outer"

    def observe_resize(reservation: ExecutionMemoryReservation, plan: ExecutionMemoryPlan) -> None:
        nonlocal native_owner, stage
        if stage == "outer":
            native_owner = reservation
            assert reservation.decision.bytes_required == input_bytes
            stage = "child"
            original_resize(reservation, plan)
            charges.append(reservation.decision.bytes_required)
            result = json.loads(inner(IR, [2.0]))
            assert result["supported"] is True
            assert result["value"] == 4.0
            if inner_surface.endswith("value_and_gradient"):
                assert result["gradient"] == [4.0]
            stage = "complete"
        else:
            original_resize(reservation, plan)
            if reservation is native_owner:
                charges.append(reservation.decision.bytes_required)

    with monkeypatch.context() as instrumentation:
        instrumentation.setattr(ExecutionMemoryReservation, "resize", observe_resize)
        result = json.loads(outer(IR, [2.0]))
    assert result["supported"] is True
    assert result["value"] == 4.0
    if outer_surface.endswith("value_and_gradient"):
        assert result["gradient"] == [4.0]
    assert charges == [
        input_bytes + outer_increments[0],
        *(input_bytes + outer_increments[0] + request for request in inner_increments),
        *(input_bytes + request + inner_increments[-1] for request in outer_increments[1:]),
    ]
    assert active_reserved_bytes() == baseline
    retry = json.loads(outer(IR, [2.0]))
    assert retry["supported"] is True
    assert retry["value"] == 4.0
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", [*SURFACES, "program_ad_effect_ir_metadata_summary"])
def test_native_json_output_budget_refuses_before_encoding_and_recovers(
    surface: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real result bytes join the native charge before an output buffer is created."""
    native = getattr(import_module("scpn_quantum_engine"), surface)
    metadata_only = surface.endswith("metadata_summary")

    def replay() -> str:
        return native(IR) if metadata_only else native(IR, [2.0])

    owner: ExecutionMemoryReservation | None = None
    measured: list[int] = []
    original_resize = ExecutionMemoryReservation.resize

    def observe(reservation: ExecutionMemoryReservation, plan: ExecutionMemoryPlan) -> None:
        nonlocal owner
        if owner is None:
            owner = reservation
        original_resize(reservation, plan)
        if reservation is owner:
            measured.append(reservation.decision.bytes_required)

    with monkeypatch.context() as instrumentation:
        instrumentation.setattr(ExecutionMemoryReservation, "resize", observe)
        expected = replay()
    input_bytes = len(IR) * 8 + (0 if metadata_only else 16 + sys.getsizeof(0.0))
    numeric_bytes = 0 if metadata_only else (40 if surface.endswith("value_and_gradient") else 16)
    encoded_bytes = len(expected.encode("utf-8"))
    python_header = max(sys.getsizeof(c) for c in ("", "a", "\u0080", "\u0100", "\U00010000"))
    # Observe real parser/table declarations, then independently pin the two
    # output increments; B must cover metadata as well as retained numerics.
    assert measured[-3] > input_bytes + numeric_bytes
    assert measured[-2] - measured[-3] == encoded_bytes
    assert measured[-1] - measured[-2] == python_header + 4 * encoded_bytes
    assert measured == sorted(measured)
    total = measured[-1]
    baseline = active_reserved_bytes()
    for budget in (total - 1, total, total + 1):
        with monkeypatch.context() as environment:
            environment.setenv("SCPN_MAX_DENSE_GIB", str(budget / 1024**3))
            if budget < total:
                with pytest.raises(DenseAllocationError):
                    replay()
            else:
                assert replay() == expected
        assert active_reserved_bytes() == baseline
        assert replay() == expected
    if not metadata_only:
        result = json.loads(expected)
        assert result["value"] == 4.0
        if surface.endswith("value_and_gradient"):
            assert result["gradient"] == [4.0]
