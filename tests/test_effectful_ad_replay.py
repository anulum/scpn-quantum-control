# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — effectful whole-program AD replay qualification
"""Aliasing, saved-program replay, captured-state digests and switching surfaces.

Every case runs through the public whole-program AD entry points. Analytic
derivatives are independent closed forms; replay agreement uses the frozen
float64 budget ``1e-10 + 1e-8*|reference|`` and digests compare exactly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import (
    Parameter,
    TraceADArray,
    TraceADScalar,
    WholeProgramReplayMismatchError,
    WholeProgramReplayRecord,
    capture_whole_program_replay,
    replay_whole_program,
    vmap,
    whole_program_value_and_grad,
)
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult

_BUDGET_ABSOLUTE = 1.0e-10
_BUDGET_RELATIVE = 1.0e-8

_FIXTURE_SOURCE = """
import numpy as np

COEFFICIENTS = np.array([0.5, -1.25, 2.0])
SHARED = [COEFFICIENTS, COEFFICIENTS]


def objective(values):
    total = values[0] * SHARED[0][0]
    for index in range(3):
        total = total * values[1] + SHARED[1][index] * values[0]
    return total
"""

_CHILD_SCRIPT = """
import importlib.util
import sys

from scpn_quantum_control import capture_whole_program_replay, whole_program_value_and_grad

spec = importlib.util.spec_from_file_location("replay_fixture_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = whole_program_value_and_grad(module.objective, [1.5, 0.75], trace=False)
print(capture_whole_program_replay(result).to_json())
"""


def _within_budget(actual: object, reference: object) -> bool:
    """Apply the frozen float64 test budget elementwise."""
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return bool(
        np.all(
            np.abs(actual_array - reference_array)
            <= _BUDGET_ABSOLUTE + _BUDGET_RELATIVE * np.abs(reference_array)
        )
    )


def _load_fixture(directory: Path, name: str) -> ModuleType:
    """Write the shared fixture source into ``directory`` and import it as ``name``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(_FIXTURE_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_reference(first: float, second: float) -> tuple[float, tuple[float, float]]:
    """Return the closed-form value and gradient of the fixture objective.

    The loop expands to ``a*(c0*b**3 + c0*b**2 + c1*b + c2)`` with the shared
    coefficients ``c = (0.5, -1.25, 2.0)``.
    """
    c0, c1, c2 = 0.5, -1.25, 2.0
    polynomial = c0 * second**3 + c0 * second**2 + c1 * second + c2
    derivative = 3.0 * c0 * second**2 + 2.0 * c0 * second + c1
    return first * polynomial, (polynomial, first * derivative)


@pytest.mark.parametrize("trace", [False, True])
def test_effectful_ad_replay_01(trace: bool) -> None:
    """An aliased shared variable contributes the adjoint of every path it feeds.

    Parameters
    ----------
    trace
        Whether the public execution also collects source-line evidence.

    """

    def objective(values: TraceADArray) -> object:
        first = values[0]
        alias = first
        paths = [first, alias]
        paths[1] = paths[1] * values[1]
        return paths[0] * paths[1] + alias

    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=trace)
    # f = a*(a*b) + a, so df/da = 2ab + 1 and df/db = a**2.
    assert result.value == 14.0
    np.testing.assert_array_equal(result.gradient, [13.0, 4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [13.0, 4.0])
    record = capture_whole_program_replay(result)
    assert record.trace is trace
    assert record.adjoint_step_count is not None and record.adjoint_step_count > 0
    assert record.parameter_values == (2.0, 3.0)


def test_effectful_ad_replay_01_array_alias_and_frozen_coordinate() -> None:
    """Array aliases accumulate both operands, and a frozen alias stays zero."""

    def objective(values: TraceADArray) -> object:
        view = values
        same = view
        return cast(TraceADArray, view * same).sum() + values[0] * values[2]

    inputs = np.array([1.5, -2.0, 0.25])
    result = whole_program_value_and_grad(
        objective,
        inputs,
        [Parameter("x"), Parameter("y"), Parameter("z", trainable=False)],
    )
    # sum(x_i**2) + x0*x2: gradient (2*x0 + x2, 2*x1, frozen).
    assert result.value == pytest.approx(1.5**2 + 4.0 + 0.0625 + 1.5 * 0.25, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, [3.25, -4.0, 0.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), [3.25, -4.0, 0.0], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(inputs, [1.5, -2.0, 0.25])


def test_effectful_ad_replay_02(tmp_path: Path) -> None:
    """A saved record replayed from its checkpoint agrees with the uninterrupted run."""
    module = _load_fixture(tmp_path / "first", "replay_fixture_first")
    uninterrupted = whole_program_value_and_grad(module.objective, [1.5, 0.75], trace=False)
    value, gradient = _fixture_reference(1.5, 0.75)
    assert _within_budget(uninterrupted.value, value)
    assert _within_budget(uninterrupted.gradient, gradient)

    record = capture_whole_program_replay(uninterrupted)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(record.to_json(), encoding="utf-8")
    restored = WholeProgramReplayRecord.from_json(checkpoint.read_text(encoding="utf-8"))
    assert restored == record
    assert restored.digest == record.digest

    report = replay_whole_program(restored, module.objective)
    assert report.record == record
    assert report.value_error == 0.0
    assert report.gradient_error == 0.0
    assert report.reverse_gradient_error == 0.0
    assert record.value == uninterrupted.value
    assert record.gradient == tuple(uninterrupted.gradient)
    assert record.effects[:2] == (("parameter", "parameter"), ("parameter", "parameter"))
    assert record.program_ir_sha256 == (
        __import__("hashlib")
        .sha256(cast(Any, uninterrupted.program_ir).serialization.encode())
        .hexdigest()
    )

    # The same source loaded from another path binds the same program.
    relocated = _load_fixture(tmp_path / "second", "replay_fixture_second")
    assert replay_whole_program(restored, relocated.objective).record == record


def test_effectful_ad_replay_02_checkpoint_survives_a_new_process(tmp_path: Path) -> None:
    """A record saved by another interpreter process replays exactly here."""
    child_path = tmp_path / "child" / "replay_fixture_child.py"
    child_path.parent.mkdir()
    child_path.write_text(_FIXTURE_SOURCE, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(item for item in sys.path if item)
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and owned fixture
        [sys.executable, "-c", _CHILD_SCRIPT, str(child_path)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    saved = WholeProgramReplayRecord.from_json(completed.stdout.strip())

    module = _load_fixture(tmp_path / "parent", "replay_fixture_parent")
    local = capture_whole_program_replay(
        whole_program_value_and_grad(module.objective, [1.5, 0.75], trace=False)
    )
    assert local.to_json() == saved.to_json()
    report = replay_whole_program(saved, module.objective)
    assert (report.value_error, report.gradient_error) == (0.0, 0.0)


def test_effectful_ad_replay_02_forward_only_program_replays_without_reverse() -> None:
    """A program without a supported reverse adjoint still replays its forward mode."""
    coefficients = np.array([1.5, -2.0])

    def objective(values: TraceADArray) -> object:
        total = values[0]
        for index in range(3):
            total = total * values[1] + coefficients[index % 2] * values[0]
        return total if values[1] > 0.0 else -total

    result = whole_program_value_and_grad(objective, [0.7, 1.3], trace=False)
    assert result.adjoint_result is not None and result.adjoint_result.supported is False
    record = capture_whole_program_replay(result)
    assert record.adjoint_step_count is None
    report = replay_whole_program(record, objective)
    assert report.reverse_gradient_error is None
    assert (report.value_error, report.gradient_error) == (0.0, 0.0)


@pytest.mark.parametrize("case", ["array", "list", "default", "alias", "code"])
def test_effectful_ad_replay_03(case: str) -> None:
    """Mutated captured state is a digest mismatch, refused before execution.

    Parameters
    ----------
    case
        Closure array element, closure list item, default-argument mapping,
        object aliasing with equal values, or replacement code.

    """
    weights = np.array([2.0, -1.0])
    offsets = [0.5]
    shared = [weights, weights]

    def objective(values: TraceADArray, scale: dict[str, float] | None = None) -> object:
        factor = 3.0 if scale is None else scale["factor"]
        return (
            values[0] * shared[0][0] * factor
            + values[1] * shared[1][1]
            + offsets[0]
            + values[0] * weights[1]
        )

    defaults = {"factor": 1.5}
    objective.__defaults__ = (defaults,)
    result = whole_program_value_and_grad(objective, [1.0, 2.0], trace=False)
    record = capture_whole_program_replay(result)
    saved = record.to_json()
    assert replay_whole_program(record, objective).record == record

    def replacement(values: TraceADArray, scale: dict[str, float] | None = None) -> object:
        return values[0] * 4.0

    candidate: Callable[..., object] = objective
    if case == "array":
        weights[0] = 7.0
    elif case == "list":
        offsets[0] = -0.5
    elif case == "default":
        defaults["factor"] = 2.5
    elif case == "alias":
        shared[1] = weights.copy()
    else:
        candidate = replacement

    expired = time.monotonic() - 1.0
    with pytest.raises(WholeProgramReplayMismatchError, match="not executed") as refusal:
        replay_whole_program(record, candidate, deadline_monotonic=expired)
    assert refusal.value.fields == ("captured_state_sha256",)
    assert record.to_json() == saved
    if case != "code":
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_replay_gradient(result)
        with pytest.raises(ValueError, match="captured program state changed"):
            capture_whole_program_replay(result)


def test_effectful_ad_replay_03_restored_state_replays_again() -> None:
    """Restoring the captured content restores the program it identified."""
    weights = np.array([2.0, -1.0])

    def objective(values: TraceADArray) -> object:
        return values[0] * weights[0] + values[1] * weights[1]

    record = capture_whole_program_replay(whole_program_value_and_grad(objective, [1.0, 2.0]))
    weights[0] = 9.0
    with pytest.raises(WholeProgramReplayMismatchError):
        replay_whole_program(record, objective)
    weights[0] = 2.0
    report = replay_whole_program(record, objective)
    assert report.record == record


def _gt(values: TraceADArray) -> object:
    return values[0] * values[0] if values[0] > 1.0 else -values[0]


def _ge(values: TraceADArray) -> object:
    return values[0] * values[0] if values[0] >= 1.0 else -values[0]


def _lt(values: TraceADArray) -> object:
    return values[0] * values[0] if values[0] < 1.0 else -values[0]


def _le(values: TraceADArray) -> object:
    return values[0] * values[0] if values[0] <= 1.0 else -values[0]


def _eq(values: TraceADArray) -> object:
    return values[0] * 3.0 if values[0] == 1.0 else values[0]


def _ne(values: TraceADArray) -> object:
    return values[0] * 3.0 if values[0] != 1.0 else values[0]


def _where_eq(values: TraceADArray) -> object:
    trace = cast(Any, values)
    return cast(TraceADArray, np.where(trace == 1.0, 3.0 * trace, trace)).sum()


def _absolute(values: TraceADArray) -> object:
    return abs(cast(TraceADScalar, values[0]) - 1.0)


def _maximum(values: TraceADArray) -> object:
    return np.maximum(cast(Any, values[0]), 1.0)


def _frozen_eq(values: TraceADArray) -> object:
    return values[0] * 2.0 if values[1] == 1.0 else values[0]


def _math_callback(values: TraceADArray) -> object:
    return math.sin(cast(float, values[0]))


def _float_detach(values: TraceADArray) -> object:
    return float(cast(TraceADScalar, values[0])) * 2.0


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        (_gt, "ordering predicate is non-differentiable at equality"),
        (_ge, "ordering predicate is non-differentiable at equality"),
        (_lt, "ordering predicate is non-differentiable at equality"),
        (_le, "ordering predicate is non-differentiable at equality"),
        (_eq, "equality predicate is non-differentiable on its switching surface"),
        (_ne, "equality predicate is non-differentiable on its switching surface"),
        (_where_eq, "equality predicate is non-differentiable on its switching surface"),
        (_frozen_eq, "equality predicate is non-differentiable on its switching surface"),
        (_absolute, "absolute value is non-differentiable at zero"),
        (_maximum, "maximum is non-differentiable at equal inputs"),
        (_math_callback, "cannot be converted to float without losing derivatives"),
        (_float_detach, "cannot be converted to float without losing derivatives"),
    ],
    ids=[
        "gt",
        "ge",
        "lt",
        "le",
        "eq",
        "ne",
        "where-eq",
        "frozen-eq",
        "abs",
        "maximum",
        "math-callback",
        "float-detach",
    ],
)
def test_effectful_ad_replay_04(objective: Callable[[TraceADArray], object], message: str) -> None:
    """A switching surface or detaching callback refuses instead of a one-sided slope.

    Parameters
    ----------
    objective
        Program evaluated exactly on its switching surface or through a
        derivative-losing conversion.
    message
        Expected refusal text.

    """
    inputs = np.array([1.0, 1.0])
    parameters = [Parameter("x"), Parameter("y", trainable=objective is not _frozen_eq)]
    with pytest.raises(ValueError, match=message):
        whole_program_value_and_grad(objective, inputs, parameters, trace=False)
    np.testing.assert_array_equal(inputs, [1.0, 1.0])


@pytest.mark.parametrize(
    ("objective", "point", "expected"),
    [(_eq, 1.5, 1.0), (_ne, 1.5, 3.0), (_gt, 1.5, 3.0), (_where_eq, 2.0, 1.0)],
    ids=["eq-off-surface", "ne-off-surface", "gt-interior", "where-off-surface"],
)
def test_effectful_ad_replay_04_branch_interiors_keep_local_derivatives(
    objective: Callable[[TraceADArray], object], point: float, expected: float
) -> None:
    """Away from the surface each executed branch keeps its analytic slope.

    Parameters
    ----------
    objective
        Piecewise program from the refusal matrix.
    point
        First coordinate strictly inside one branch.
    expected
        Analytic derivative of the selected branch in the first coordinate.

    """
    result = whole_program_value_and_grad(objective, [point, 2.0], trace=False)
    assert result.gradient[0] == expected


def test_effectful_ad_replay_04_self_equality_is_not_a_surface() -> None:
    """A value compared with itself is constant and keeps the executed derivative."""

    def objective(values: TraceADArray) -> object:
        first = values[0]
        return first * 2.0 if first == first else first

    result = whole_program_value_and_grad(objective, [1.0], trace=False)
    assert result.gradient.tolist() == [2.0]


def test_effectful_ad_replay_04_external_effects_refuse_without_side_effects() -> None:
    """External callbacks and generator draws refuse before running or advancing."""
    calls: list[str] = []
    generator = np.random.default_rng(3)
    before = generator.bit_generator.state

    def external_callback() -> float:
        calls.append("called")
        return 7.0

    def callback_objective(values: TraceADArray) -> object:
        return values[0] ** 2 + external_callback()

    def generator_objective(values: TraceADArray) -> object:
        return values[0] * generator.normal()

    for objective in (callback_objective, generator_objective):
        with pytest.raises(ValueError, match="external_callback"):
            whole_program_value_and_grad(objective, [1.0])
    assert calls == []
    assert generator.bit_generator.state == before


def _sample_record() -> tuple[WholeProgramReplayRecord, Callable[[TraceADArray], object]]:
    """Return a record of a small program with a supported reverse adjoint."""

    def objective(values: TraceADArray) -> object:
        return values[0] * values[1] + values[0]

    return capture_whole_program_replay(whole_program_value_and_grad(objective, [2.0, 3.0])), (
        objective
    )


@pytest.mark.parametrize(
    ("field", "value", "fields"),
    [
        ("program_ir_sha256", "0" * 64, ("program_ir_sha256",)),
        ("effects", (("pure", "mul"),), ("effects",)),
        ("adjoint_step_count", 999, ("adjoint_step_count",)),
        ("value", 7.0, ("value",)),
        ("gradient", (1.0, 1.0), ("gradient", "reverse_gradient")),
        ("parameter_values", (2.5, 3.0), ("value", "gradient", "reverse_gradient")),
    ],
)
def test_replay_reports_every_mismatched_field(
    field: str, value: object, fields: tuple[str, ...]
) -> None:
    """A changed saved field is named rather than accepted as a match.

    Parameters
    ----------
    field
        Saved record field to change.
    value
        Replacement saved value.
    fields
        Field names the refusal must report.

    """
    record, objective = _sample_record()
    with pytest.raises(WholeProgramReplayMismatchError) as refusal:
        replay_whole_program(replace(record, **cast(dict[str, Any], {field: value})), objective)
    assert refusal.value.fields == fields


@pytest.mark.parametrize(
    ("changes", "fields"),
    [
        ({"interpreter": "cpython-0"}, ("interpreter",)),
        ({"numpy_version": "0.0"}, ("numpy_version",)),
        ({"interpreter": "cpython-0", "numpy_version": "0.0"}, ("interpreter", "numpy_version")),
    ],
)
def test_replay_refuses_another_runtime(changes: dict[str, str], fields: tuple[str, ...]) -> None:
    """Code and arithmetic from another runtime are not compared as equal.

    Parameters
    ----------
    changes
        Saved runtime fields that differ from this process.
    fields
        Field names the refusal must report.

    """
    record, objective = _sample_record()
    with pytest.raises(WholeProgramReplayMismatchError) as refusal:
        replay_whole_program(replace(record, **cast(dict[str, Any], changes)), objective)
    assert refusal.value.fields == fields


def test_replay_and_capture_refuse_malformed_inputs(tmp_path: Path) -> None:
    """Malformed records, objectives and results refuse with explicit reasons."""
    record, objective = _sample_record()
    with pytest.raises(ValueError, match="WholeProgramReplayRecord"):
        replay_whole_program(cast(Any, record.to_json()), objective)
    with pytest.raises(ValueError, match="callable"):
        replay_whole_program(record, cast(Any, 3.0))
    with pytest.raises(ValueError, match="WholeProgramADResult"):
        capture_whole_program_replay(cast(Any, record))
    standalone = WholeProgramADResult(
        value=1.0,
        gradient=np.array([1.0]),
        method="whole_program_ad",
        step=0.0,
        evaluations=1,
        parameter_names=("x",),
        trainable=(True,),
        trace_events=(),
        source=None,
        control_flow_observed=False,
        numpy_observed=False,
        polyglot_targets={"python": "standalone historical record"},
        claim_boundary="standalone record",
    )
    with pytest.raises(ValueError, match="live binding"):
        capture_whole_program_replay(standalone)

    unsupported = {1, 2}

    def opaque(values: TraceADArray) -> object:
        return values[0] * len(unsupported)

    with pytest.raises(ValueError, match="no portable encoding"):
        replay_whole_program(record, opaque)


def test_capture_refuses_state_beyond_the_portable_snapshot(tmp_path: Path) -> None:
    """Live binding fits, but the portable digest also hashes code constants.

    An 8.08 MB captured array leaves the live snapshot under its 8 MiB limit;
    a 400 kB string constant in the code pushes the portable snapshot over it.
    """
    source = (
        "import numpy as np\n\n"
        "WEIGHTS = np.zeros(1_010_000)\n\n\n"
        "def objective(values):\n"
        f'    label = "{"x" * 400_000}"\n'
        "    return values[0] * 2.0 + WEIGHTS[0]\n"
    )
    path = tmp_path / "replay_large_fixture.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("replay_large_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = whole_program_value_and_grad(module.objective, [1.0], trace=False)
    assert result.gradient.tolist() == [2.0]
    with pytest.raises(ValueError, match="no portable encoding"):
        capture_whole_program_replay(result)


def test_portable_digest_binds_library_references_by_name() -> None:
    """Library code, typing casts and the package transform are bound by name.

    ``np.identity`` is a plain Python NumPy function, unlike the dispatchers
    behind ``np.sum``, so it exercises the named binding of library code.
    """
    record, _ = _sample_record()

    def library_objective(values: TraceADArray) -> object:
        if "alpha" in {"alpha", "beta"}:
            return cast(Any, values[0]) * np.sum(np.array([1.0, 2.0])) + (lambda z: z)(1.0)
        return vmap, np.identity

    with pytest.raises(WholeProgramReplayMismatchError) as refusal:
        replay_whole_program(record, library_objective)
    assert refusal.value.fields == ("captured_state_sha256",)


def _wire(record: WholeProgramReplayRecord) -> dict[str, Any]:
    """Return the decoded JSON wire object of ``record``."""
    return cast(dict[str, Any], json.loads(record.to_json()))


def _mutated_wire(record: WholeProgramReplayRecord, key: str, value: object) -> str:
    """Return the wire text with one record field replaced and the digest kept."""
    payload = _wire(record)
    payload["record"][key] = value
    return json.dumps(payload)


def test_record_codec_refuses_malformed_and_tampered_text() -> None:
    """Unknown versions, keys, encodings or a stale digest never parse."""
    record, _ = _sample_record()
    payload = _wire(record)
    assert set(payload) == {"schema", "digest", "record"}
    assert payload["record"]["value"] == "4020000000000000"
    cases: list[tuple[object, str]] = [
        (3, "must be a string"),
        ("{", "malformed"),
        ("[]", "exactly schema, digest and record"),
        (json.dumps({**payload, "extra": 1}), "exactly schema, digest and record"),
        (
            json.dumps({**payload, "schema": "whole_program_replay_record.v2"}),
            "unsupported replay record schema",
        ),
        (json.dumps({**payload, "record": []}), "do not match the v1 schema"),
        (
            json.dumps({**payload, "record": {**payload["record"], "extra": 1}}),
            "do not match the v1 schema",
        ),
        (_mutated_wire(record, "parameter_names", "x"), "JSON array"),
        (_mutated_wire(record, "value", "4020"), "16 lowercase hex digits"),
        (_mutated_wire(record, "value", "402000000000000G"), "16 lowercase hex digits"),
        (_mutated_wire(record, "value", 8.0), "16 lowercase hex digits"),
        (_mutated_wire(record, "value", "7ff8000000000000"), "finite float"),
        (_mutated_wire(record, "value", "4022000000000000"), "digest does not match"),
    ]
    for text, message in cases:
        with pytest.raises(ValueError, match=message):
            WholeProgramReplayRecord.from_json(cast(str, text))
    assert WholeProgramReplayRecord.from_json(record.to_json()) == record


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("interpreter", "", "interpreter must be a non-empty string"),
        ("numpy_version", 2, "numpy_version must be a non-empty string"),
        ("program_ir_sha256", "A" * 64, "lowercase SHA-256"),
        ("captured_state_sha256", "0" * 63, "lowercase SHA-256"),
        ("parameter_names", ["x", "y"], "unique non-empty strings"),
        ("parameter_names", ("x", ""), "unique non-empty strings"),
        ("parameter_names", ("x", "x"), "unique non-empty strings"),
        ("trainable", [True, True], "tuple of booleans"),
        ("trainable", (True, 1), "tuple of booleans"),
        ("trace", 1, "trace must be a boolean"),
        ("parameter_values", (2.0, 3), "tuple of finite floats"),
        ("gradient", (math.inf, 1.0), "tuple of finite floats"),
        ("gradient", [4.0, 2.0], "tuple of finite floats"),
        ("gradient", (4.0,), "one entry per parameter"),
        ("value", 8, "value must be a finite float"),
        ("value", math.nan, "value must be a finite float"),
        ("effects", [("pure", "mul")], "effects must be"),
        ("effects", (("pure",),), "effects must be"),
        ("effects", (("", "mul"),), "effects must be"),
        ("effects", ((1, "mul"),), "effects must be"),
        ("effects", (("pure", ""),), "effects must be"),
        ("effects", (("pure", 3),), "effects must be"),
        ("adjoint_step_count", -1, "non-negative integer"),
        ("adjoint_step_count", True, "non-negative integer"),
    ],
)
def test_record_fields_are_validated_on_construction(
    field: str, value: object, message: str
) -> None:
    """Every saved field has an exact type, length and finiteness contract.

    Parameters
    ----------
    field
        Record field given a malformed value.
    value
        Malformed replacement.
    message
        Expected refusal text.

    """
    record, _ = _sample_record()
    with pytest.raises(ValueError, match=message):
        replace(record, **cast(dict[str, Any], {field: value}))


def test_record_accepts_effects_without_an_operation() -> None:
    """An effect row without a recorded operation round-trips as ``None``."""
    record, _ = _sample_record()
    changed = replace(record, effects=(*record.effects, ("mutation", None)))
    assert WholeProgramReplayRecord.from_json(changed.to_json()) == changed
    assert changed.digest != record.digest
