# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source admission tests
"""Actual source files, encodings and callable blocks under owned admission."""

from __future__ import annotations

import codecs
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import compile_whole_program_frontend
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    active_reserved_bytes,
    reserve_execution_memory,
)
from scpn_quantum_control.source_admission import objective_source_block, read_source_lines


@pytest.mark.parametrize(
    "contents",
    [
        b"def objective(values):\n    return values[0] * values[0]\n",
        b"def objective(values):\r\n    return values[0] * values[0]",
        b"# coding: latin-1\n# caf\xe9\ndef objective(values):\n    return values[0]\n",
        b"\xef\xbb\xbfdef objective(values):\n    return values[0]\n",
        b"",
    ],
)
def test_read_source_lines_matches_python_text_decoding(tmp_path: Path, contents: bytes) -> None:
    """Owned reads match real Python decoding and universal newline semantics."""
    path = tmp_path / "source.py"
    path.write_bytes(contents)
    with tokenize.open(path) as source:
        expected = source.readlines()
    if expected and not expected[-1].endswith("\n"):
        expected[-1] += "\n"
    baseline = active_reserved_bytes()
    initial = ExecutionMemoryPlan(
        (ExecutionBuffer("source_owner", "intermediate", (1,), "uint8"),)
    )
    with reserve_execution_memory(initial) as reservation:
        lines, plan = read_source_lines(str(path), path.stat(), reservation)
        assert lines == expected
        assert active_reserved_bytes() == baseline + plan.bytes_required
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "interruption",
    ["truncate", "grow", "overwrite", "cancel", "deadline", "open_cancel", "budget"],
)
def test_public_ad_source_read_interruption_recovers(tmp_path: Path, interruption: str) -> None:
    """Real file reads observe filesystem and ancestor-lifecycle interruptions."""
    path = tmp_path / "interrupted_source.py"
    original = (
        "#" + "x" * (128 * 1024) + "\ndef objective(values):\n    return values[0] * values[0]\n"
    )
    path.write_text(original, encoding="utf-8")
    program = r"""
import importlib.util
import json
import sys
from pathlib import Path
from threading import Event
from time import monotonic
import numpy as np
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import whole_program_value_and_grad
from scpn_quantum_control.execution_reservations import ExecutionCancelledError, active_reserved_bytes

path = Path(sys.argv[1])
interruption = sys.argv[2]
original = path.read_bytes()
spec = importlib.util.spec_from_file_location("interrupted_source", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
baseline = active_reserved_bytes()
cancelled = Event()
deadline = monotonic() + 5.0
fired = False
reads = 0
opens = 0
armed = True
with path.open("r+b") as writer:
    def source_open(event, args):
        global fired, opens
        if armed and event == "open" and args[0] == str(path):
            opens += 1
            if interruption == "open_cancel":
                fired = True
                cancelled.set()
    def observe_read(frame, event, method):
        global fired, reads
        if event != "c_call" or getattr(method, "__name__", None) != "readinto":
            return
        source = getattr(method, "__self__", None)
        if getattr(source, "name", None) != str(path):
            return
        reads += 1
        if fired:
            return
        fired = True
        if interruption == "truncate":
            writer.truncate(0)
            writer.flush()
        elif interruption == "grow":
            writer.seek(0, 2)
            writer.write(b"\n# source grew\n")
            writer.flush()
        elif interruption == "overwrite":
            writer.seek(1)
            writer.write(b"y")
            writer.flush()
        elif interruption == "cancel":
            cancelled.set()
        elif interruption == "deadline":
            Event().wait(max(0.0, deadline - monotonic()) + 0.01)
        else:
            raise AssertionError("unexpected first source read")
    sys.addaudithook(source_open)
    sys.setprofile(observe_read)
    try:
        whole_program_value_and_grad(
            module.objective, [2.0], trace=False,
            max_execution_gib=0.001 if interruption == "budget" else 0.1,
            cancelled=cancelled,
            deadline_monotonic=deadline if interruption == "deadline" else None,
        )
    except (DenseAllocationError, ExecutionCancelledError, TimeoutError) as exc:
        error_type = type(exc).__name__
        error = str(exc)
    else:
        raise AssertionError("interrupted source read succeeded")
    finally:
        sys.setprofile(None)
        armed = False
if interruption == "budget":
    assert not fired and opens == 0
else:
    assert fired and opens > 0
assert active_reserved_bytes() == baseline
path.write_bytes(original)
cancelled.clear()
result = whole_program_value_and_grad(
    module.objective, [2.0], trace=False, max_execution_gib=0.1, cancelled=cancelled
)
assert result.value == 4.0
np.testing.assert_array_equal(result.gradient, [4.0])
assert active_reserved_bytes() == baseline
print(json.dumps({"error_type": error_type, "error": error, "reads": reads, "opens": opens}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), interruption],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    if interruption in {"cancel", "open_cancel"}:
        assert payload["error_type"] == "ExecutionCancelledError"
    elif interruption == "deadline":
        assert payload["error_type"] == "TimeoutError"
    elif interruption == "budget":
        assert payload["error_type"] == "DenseAllocationError"
        assert "execution memory" in payload["error"]
        assert payload["opens"] == 0
    else:
        assert payload["error_type"] == "DenseAllocationError"
        assert "during bounded read" in payload["error"]
    if interruption in {"open_cancel", "budget"}:
        expected_reads = 0
    elif interruption in {"truncate", "cancel", "deadline"}:
        expected_reads = 1
    else:
        expected_reads = (len(original.encode("utf-8")) + 65535) // 65536
    assert payload["reads"] == expected_reads


def test_read_source_lines_refuses_changed_file_and_recovers(tmp_path: Path) -> None:
    """A changed real file cannot reuse an earlier size observation."""
    path = tmp_path / "changed.py"
    path.write_text("value = 1\n", encoding="utf-8")
    observation = path.stat()
    path.write_text("value = 1000\n", encoding="utf-8")
    baseline = active_reserved_bytes()
    initial = ExecutionMemoryPlan(
        (ExecutionBuffer("source_owner", "intermediate", (1,), "uint8"),)
    )
    with (
        reserve_execution_memory(initial) as reservation,
        pytest.raises(DenseAllocationError, match="changed before bounded read"),
    ):
        read_source_lines(str(path), observation, reservation)
    assert active_reserved_bytes() == baseline
    with reserve_execution_memory(initial) as reservation:
        lines, _ = read_source_lines(str(path), path.stat(), reservation)
        assert lines == ["value = 1000\n"]
    assert active_reserved_bytes() == baseline


def test_objective_source_blocks_and_frontend_match_actual_inspection(tmp_path: Path) -> None:
    """Functions, lambdas, decorators and lexical classes use the same real blocks."""
    path = tmp_path / "source_blocks.py"
    path.write_text(
        "from functools import wraps\n"
        "def decorate(fn):\n"
        "    @wraps(fn)\n"
        "    def wrapper(values):\n"
        "        return fn(values)\n"
        "    return wrapper\n"
        "@decorate\n"
        "def objective(values):\n"
        "    return values[0] * values[0]\n"
        "linear = lambda values: values[0]\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        def objective(self, values):\n"
        "            return values[0]\n"
        "def factory():\n"
        "    class Local:\n"
        "        pass\n"
        "    return Local\n",
        encoding="utf-8",
    )
    name = "source_admission_actual_blocks"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    baseline = active_reserved_bytes()
    try:
        spec.loader.exec_module(module)
        initial = ExecutionMemoryPlan(
            (ExecutionBuffer("source_owner", "intermediate", (1,), "uint8"),)
        )
        with reserve_execution_memory(initial) as reservation:
            lines, _ = read_source_lines(str(path), path.stat(), reservation)
            for objective in (
                module.objective,
                module.linear,
                module.Outer,
                module.Outer.Inner,
                module.Outer.Inner().objective,
                module.factory(),
            ):
                assert objective_source_block(objective, lines) == inspect.getsourcelines(
                    objective
                )
            with pytest.raises(OSError, match="code object"):
                objective_source_block(len, lines)
        report = compile_whole_program_frontend(module.objective)
        assert report.source_available
        assert "source_bytecode_mismatch" in report.hard_gaps
        assert report.source_start_line == inspect.getsourcelines(module.objective)[1]
    finally:
        del sys.modules[name]
    assert active_reserved_bytes() == baseline


def test_source_reader_refuses_nonregular_observation(tmp_path: Path) -> None:
    """A real directory observation cannot authorize source-buffer materialisation."""
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("owner", "forward", (1,), "uint8"),))
    with (
        reserve_execution_memory(plan) as owner,
        pytest.raises(DenseAllocationError, match="regular file"),
    ):
        read_source_lines(str(tmp_path), tmp_path.stat(), owner)
    assert active_reserved_bytes() == baseline


def test_source_reader_refuses_expanding_registered_codec_and_recovers(tmp_path: Path) -> None:
    """An actual codec cannot produce text larger than its admitted character bound."""

    def decode(data: bytes | bytearray | memoryview, errors: str = "strict") -> tuple[str, int]:
        return bytes(data).decode("ascii", errors) * 2, len(data)

    def lookup(name: str) -> codecs.CodecInfo | None:
        if name == "scpn_expanding":
            return codecs.CodecInfo(
                name=name,
                encode=codecs.getencoder("ascii"),
                decode=decode,
            )
        return None

    path = tmp_path / "expanding_source.py"
    path.write_bytes(b"# coding: scpn-expanding\nvalue = 1\n")
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("owner", "forward", (1,), "uint8"),))
    codecs.register(lookup)
    try:
        with (
            reserve_execution_memory(plan) as owner,
            pytest.raises(DenseAllocationError, match="codec exceeded"),
        ):
            read_source_lines(str(path), path.stat(), owner)
    finally:
        codecs.unregister(lookup)
    assert active_reserved_bytes() == baseline
    path.write_text("value = 1\n")
    with reserve_execution_memory(plan) as owner:
        lines, _ = read_source_lines(str(path), path.stat(), owner)
        assert lines == ["value = 1\n"]
    assert active_reserved_bytes() == baseline


def test_source_block_refuses_missing_class_and_displaced_function_line(tmp_path: Path) -> None:
    """Actual callable metadata must locate a valid block in the observed source."""
    path = tmp_path / "displaced_source.py"
    path.write_text("# header\n\ndef objective(values):\n    return values[0]\n")
    spec = importlib.util.spec_from_file_location("displaced_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = path.read_text().splitlines(keepends=True)
    with pytest.raises(OSError, match="class definition"):
        objective_source_block(type("Absent", (), {}), lines)
    original = module.objective.__code__
    module.objective.__code__ = original.replace(co_firstlineno=len(lines) + 10)
    with pytest.raises(OSError, match="outside source"):
        objective_source_block(module.objective, lines)
    report = compile_whole_program_frontend(module.objective)
    assert not report.source_available
    module.objective.__code__ = original.replace(co_firstlineno=4)
    assert objective_source_block(module.objective, lines) == (lines[2:], 3)
    module.objective.__code__ = original
    assert compile_whole_program_frontend(module.objective).source_available
