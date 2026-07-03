# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto external-solver adapters
"""Tests for :mod:`kuramoto_external_competitors`.

The SciPy adapter runs against the real installed solver; every subprocess
adapter (the Julia packages and JIT-compiled-C ``jitcdde``) is driven through
injected runners and monkeypatched ``subprocess``/``shutil`` so the suite is
deterministic and needs neither a Julia toolchain nor a C compiler on the runner,
while still pinning the fail-closed contract each adapter must honour.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import kuramoto_external_competitors as ext
from scpn_quantum_control.benchmarks.kuramoto_competitive_types import (
    KuramotoProblem,
    build_default_problem,
)


def _small_problem() -> KuramotoProblem:
    """A tiny but non-trivial deterministic problem for fast real runs."""
    return build_default_problem(n_oscillators=6, t_max=0.5, dt=0.1, seed=7)


def _ok_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Injected subprocess runner returning a fixed valid result."""
    return {"r_final": 0.5, "elapsed_ms": 2.5, "version": "1.2.3"}


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------- #
# module-presence + payload + result parsing
# --------------------------------------------------------------------------- #


def test_python_module_present_true_and_false() -> None:
    assert ext._python_module_present("numpy") is True
    assert ext._python_module_present("definitely_not_a_real_module_xyz") is False


def test_problem_payload_carries_problem_fields() -> None:
    payload = json.loads(ext._problem_payload(_small_problem()))
    assert set(payload) == {"K", "omega", "theta0", "t_max"}
    assert len(payload["omega"]) == 6
    assert payload["t_max"] == 0.5


def test_parse_subprocess_result_happy_and_errors() -> None:
    parsed = ext._parse_subprocess_result(
        'noise\n{"r_final": 0.4, "elapsed_ms": 1.0, "version": "v"}\n'
    )
    assert parsed == {"r_final": 0.4, "elapsed_ms": 1.0, "version": "v"}
    with pytest.raises(RuntimeError, match="could not parse"):
        ext._parse_subprocess_result("not json")
    with pytest.raises(RuntimeError, match="could not parse"):
        ext._parse_subprocess_result("")


# --------------------------------------------------------------------------- #
# _run_julia_script (subprocess boundary mocked)
# --------------------------------------------------------------------------- #


def test_run_julia_script_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        ext._run_julia_script("script", _small_problem(), 5.0)


def test_run_julia_script_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"r_final": 0.42, "elapsed_ms": 3.1, "version": "7.17.0"})
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout=f"x\n{payload}\n"))
    result = ext._run_julia_script("script", _small_problem(), 5.0)
    assert result == {"r_final": 0.42, "elapsed_ms": 3.1, "version": "7.17.0"}


def test_run_julia_script_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(1, stderr="Package X not found")
    )
    with pytest.raises(RuntimeError, match="julia subprocess failed"):
        ext._run_julia_script("script", _small_problem(), 5.0)


def test_run_julia_script_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="julia", timeout=5.0)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", _raise)
    with pytest.raises(RuntimeError, match="timed out"):
        ext._run_julia_script("script", _small_problem(), 5.0)


def test_run_julia_script_unparsable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout="not json"))
    with pytest.raises(RuntimeError, match="could not parse"):
        ext._run_julia_script("script", _small_problem(), 5.0)


# --------------------------------------------------------------------------- #
# _run_jitcdde (module presence + subprocess boundary mocked)
# --------------------------------------------------------------------------- #


def test_run_jitcdde_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "_python_module_present", lambda _m: False)
    with pytest.raises(FileNotFoundError, match="jitcdde is not installed"):
        ext._run_jitcdde(_small_problem(), 5.0)


def test_run_jitcdde_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"r_final": 0.77, "elapsed_ms": 18.0, "version": "1.8.3"})
    monkeypatch.setattr(ext, "_python_module_present", lambda _m: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout=payload))
    result = ext._run_jitcdde(_small_problem(), 5.0)
    assert result == {"r_final": 0.77, "elapsed_ms": 18.0, "version": "1.8.3"}


def test_run_jitcdde_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "_python_module_present", lambda _m: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(1, stderr="compile error"))
    with pytest.raises(RuntimeError, match="jitcdde subprocess failed"):
        ext._run_jitcdde(_small_problem(), 5.0)


def test_run_jitcdde_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="python", timeout=5.0)

    monkeypatch.setattr(ext, "_python_module_present", lambda _m: True)
    monkeypatch.setattr(subprocess, "run", _raise)
    with pytest.raises(RuntimeError, match="timed out"):
        ext._run_jitcdde(_small_problem(), 5.0)


# --------------------------------------------------------------------------- #
# default runners route to their own embedded programs
# --------------------------------------------------------------------------- #


def test_julia_default_runners_route_to_their_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def _capture(script: str, problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
        seen["script"] = script
        return {"r_final": 0.1, "elapsed_ms": 1.0, "version": "v"}

    monkeypatch.setattr(ext, "_run_julia_script", _capture)
    ext.default_julia_runner(_small_problem(), 5.0)
    assert seen["script"] is ext._JULIA_DIFFEQ_SCRIPT
    ext.default_dynamicalsystems_runner(_small_problem(), 5.0)
    assert seen["script"] is ext._JULIA_DYNAMICALSYSTEMS_SCRIPT
    ext.default_networkdynamics_runner(_small_problem(), 5.0)
    assert seen["script"] is ext._JULIA_NETWORKDYNAMICS_SCRIPT
    ext.default_scimlsensitivity_runner(_small_problem(), 5.0)
    assert seen["script"] is ext._JULIA_SCIMLSENSITIVITY_SCRIPT


def test_jitcdde_default_runner_routes_to_run_jitcdde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ext, "_run_jitcdde", lambda p, t: {"r_final": 0.7, "elapsed_ms": 9.0, "version": "1.8.3"}
    )
    assert ext.default_jitcdde_runner(_small_problem(), 5.0)["version"] == "1.8.3"


# --------------------------------------------------------------------------- #
# SciPy row (real surface)
# --------------------------------------------------------------------------- #


def test_scipy_row_available_and_matches_reference() -> None:
    row = ext.scipy_row(_small_problem())
    assert row.available is True
    assert row.family == "external" and row.language == "python"
    assert row.version is not None
    assert row.r_final is not None and 0.0 <= row.r_final <= 1.0


def test_scipy_row_fails_closed_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "_python_module_present", lambda _module: False)
    row = ext.scipy_row(_small_problem())
    assert row.available is False
    assert row.unavailable_reason == "scipy not installed"
    assert row.install_command == ext.INSTALL_COMMANDS["scipy_solve_ivp"]


# --------------------------------------------------------------------------- #
# subprocess row builders (available + fail-closed, per competitor)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("builder", "method", "language"),
    [
        (ext.julia_diffeq_row, "julia_diffeq", "julia"),
        (ext.dynamicalsystems_row, "dynamicalsystems_jl", "julia"),
        (ext.networkdynamics_row, "networkdynamics_jl", "julia"),
        (ext.scimlsensitivity_row, "scimlsensitivity_jl", "julia"),
        (ext.jitcdde_row, "jitcdde", "python"),
    ],
)
def test_subprocess_rows_available_with_injected_runner(
    builder: Any, method: str, language: str
) -> None:
    row = builder(_small_problem(), timeout=5.0, runner=_ok_runner)
    assert row.available is True
    assert row.method == method
    assert row.family == "external" and row.language == language
    assert row.version == "1.2.3" and row.r_final == 0.5 and row.elapsed_ms == 2.5


@pytest.mark.parametrize(
    ("builder", "method"),
    [
        (ext.julia_diffeq_row, "julia_diffeq"),
        (ext.dynamicalsystems_row, "dynamicalsystems_jl"),
        (ext.networkdynamics_row, "networkdynamics_jl"),
        (ext.scimlsensitivity_row, "scimlsensitivity_jl"),
        (ext.jitcdde_row, "jitcdde"),
    ],
)
def test_subprocess_rows_fail_closed_on_missing_and_error(builder: Any, method: str) -> None:
    def _missing(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
        raise FileNotFoundError("toolchain not found")

    def _boom(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
        raise RuntimeError("package not installed")

    missing = builder(_small_problem(), timeout=5.0, runner=_missing)
    assert missing.available is False
    assert missing.unavailable_reason == "toolchain not found"
    assert missing.install_command == ext.INSTALL_COMMANDS[method]

    errored = builder(_small_problem(), timeout=5.0, runner=_boom)
    assert errored.available is False
    assert errored.unavailable_reason == "package not installed"
