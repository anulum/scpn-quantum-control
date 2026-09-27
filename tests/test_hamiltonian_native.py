# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native Hamiltonian admission tests
"""Mandatory installed-engine Hamiltonian boundaries and independent XY oracles."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    ExecutionMemoryReservation,
    active_reserved_bytes,
    reserve_execution_memory,
)


@pytest.mark.parametrize("surface", ["build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"])
@pytest.mark.parametrize("invalid_n", [sys.maxsize, sys.maxsize.bit_length() + 1])
def test_native_hamiltonian_impossible_dimension_refuses_before_allocation(
    surface: str, invalid_n: int
) -> None:
    """Small real inputs cannot trigger a wrapped shift or a dense giant allocation."""
    engine = import_module("scpn_quantum_engine")
    width = sys.maxsize.bit_length() + 1
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="native addressability"):
        getattr(engine, surface)(np.zeros(width * width), np.zeros(width), invalid_n)
    assert active_reserved_bytes() == baseline


def test_native_dense_hamiltonian_byte_product_refuses_before_allocation() -> None:
    """A representable Hilbert dimension still rejects an unaddressable dense matrix."""
    engine = import_module("scpn_quantum_engine")
    n = (sys.maxsize.bit_length() + 1) // 2
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="bytes exceed native addressability"):
        engine.build_xy_hamiltonian_dense(np.zeros(n * n), np.zeros(n), n)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"])
@pytest.mark.parametrize("frequency", [-1e-15, 1e-15, float(np.nextafter(1e-15, np.inf))])
def test_native_hamiltonian_matches_canonical_frequency_cutoff(
    surface: str, frequency: float
) -> None:
    """The canonical inclusive frequency cutoff also applies at direct native entries."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    effective = frequency if abs(frequency) > 1e-15 else 0.0
    if surface == "build_xy_hamiltonian_dense":
        actual = np.asarray(
            getattr(engine, surface)(np.zeros(1), np.array([frequency]), 1)
        ).reshape(2, 2)
    else:
        rows, columns, values = getattr(engine, surface)(np.zeros(1), np.array([frequency]), 1)
        actual = np.zeros((2, 2))
        np.add.at(actual, (rows, columns), values)
    np.testing.assert_array_equal(actual, np.diag([-effective, effective]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("coupling", [0.0, 1.0])
def test_native_sparse_hamiltonian_entry_and_byte_products_refuse(coupling: float) -> None:
    """Sparse triplet counts and byte products reject native overflow before vector growth."""
    engine = import_module("scpn_quantum_engine")
    n = sys.maxsize.bit_length()
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="(entries|bytes) exceed native addressability"):
        engine.build_sparse_xy_hamiltonian(np.full(n * n, coupling), np.zeros(n), n)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"])
def test_native_hamiltonian_obeys_actual_environment_budget_and_recovers(
    surface: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed kernels honor the real policy environment and release refused charges."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    k = np.array([0.0, 0.5, 0.5, 0.0])
    omega = np.array([1.0, 0.0])
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "1e-9")
        with pytest.raises(DenseAllocationError):
            getattr(engine, surface)(k, omega, 2)
    assert active_reserved_bytes() == baseline
    expected = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, -1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if surface == "build_xy_hamiltonian_dense":
        actual = np.asarray(getattr(engine, surface)(k, omega, 2)).reshape(4, 4)
    else:
        rows, columns, values = getattr(engine, surface)(k, omega, 2)
        actual = np.zeros((4, 4))
        np.add.at(actual, (rows, columns), values)
    np.testing.assert_array_equal(actual, expected)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"])
def test_native_hamiltonian_inherits_real_owner_lifecycle(surface: str) -> None:
    """Native direct calls cannot bypass a cancelled or expired active owner."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    cancelled = Event()
    plan = ExecutionMemoryPlan((ExecutionBuffer("native_owner", "forward", (1,), "uint8"),))
    with reserve_execution_memory(plan, cancelled=cancelled):
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            getattr(engine, surface)(np.zeros(4), np.zeros(2), 2)
    assert active_reserved_bytes() == baseline
    deadline = monotonic() + 5.0
    with reserve_execution_memory(plan, deadline_monotonic=deadline):
        Event().wait(max(0.0, deadline - monotonic()) + 0.01)
        with pytest.raises(TimeoutError):
            getattr(engine, surface)(np.zeros(4), np.zeros(2), 2)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"])
def test_native_hamiltonian_refuses_nonfinite_constructed_output(surface: str) -> None:
    """Finite frequencies whose sum overflows cannot publish an infinite matrix."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="Hamiltonian output.*not finite"):
        getattr(engine, surface)(np.zeros(4), np.array([1e308, 1e308]), 2)
    assert active_reserved_bytes() == baseline


@pytest.fixture(scope="module")
def installed_hamiltonian_wheels(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Install exact-source control/toolkit wheels and the ABI-matched CI engine."""
    repo = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("hamiltonian_installed_wheels")
    wheels = directory / "wheels"
    wheels.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for project in (repo, repo / "oscillatools"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheels),
                str(project),
            ],
            check=True,
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
        )
    engines = sorted((repo / "dist").glob("scpn_quantum_engine-*.whl"))
    assert len(engines) == 1, "the native CI job must supply one current ABI-matched wheel"
    venv = directory / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
    )
    python = venv / "bin" / "python"
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *map(str, sorted(wheels.glob("*.whl"))),
            str(engines[0]),
        ],
        check=True,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
    )
    return python, directory


def test_native_hamiltonian_installed_wheels_share_admission_and_recover(
    installed_hamiltonian_wheels: tuple[Path, Path],
) -> None:
    """Installed artifacts, without source imports, retain real budget refusal and recovery."""
    python, directory = installed_hamiltonian_wheels
    program = r"""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
from packaging.requirements import Requirement
import scpn_quantum_engine as engine

assert "scpn_quantum_control" not in sys.modules
cold_k = np.array([0.0, 0.5, 0.5, 0.0])
cold_w = np.array([1.0, 0.0])
previous = os.environ.get("SCPN_MAX_DENSE_GIB")
os.environ["SCPN_MAX_DENSE_GIB"] = "0.1"
cold_result = engine.build_xy_hamiltonian_dense(cold_k, cold_w, 2)
assert "scpn_quantum_control.execution_reservations" in sys.modules
import scpn_quantum_control
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_reservations import active_reserved_bytes

prefix = Path(sys.prefix).resolve()
for module in (scpn_quantum_control, engine):
    assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
requirements = [Requirement(value) for value in importlib.metadata.requires("scpn-quantum-engine")]
control = [value for value in requirements if value.name == "scpn-quantum-control"]
assert len(control) == 1
assert importlib.metadata.version("scpn-quantum-control") in control[0].specifier
baseline = active_reserved_bytes()
k = np.array([0.0, 0.5, 0.5, 0.0])
w = np.array([1.0, 0.0])
expected = np.diag([-1.0, 1.0, -1.0, 1.0])
expected[1, 2] = expected[2, 1] = -1.0
np.testing.assert_array_equal(np.asarray(cold_result).reshape(4, 4), expected)
try:
    for name in ("build_xy_hamiltonian_dense", "build_sparse_xy_hamiltonian"):
        os.environ["SCPN_MAX_DENSE_GIB"] = "1e-9"
        try:
            getattr(engine, name)(k, w, 2)
        except DenseAllocationError:
            pass
        else:
            raise AssertionError("installed native entry ignored the real budget")
        assert active_reserved_bytes() == baseline
        os.environ["SCPN_MAX_DENSE_GIB"] = "0.1"
        result = getattr(engine, name)(k, w, 2)
        if name == "build_xy_hamiltonian_dense":
            actual = np.asarray(result).reshape(4, 4)
        else:
            rows, columns, values = result
            actual = np.zeros((4, 4))
            np.add.at(actual, (rows, columns), values)
        np.testing.assert_array_equal(actual, expected)
        assert active_reserved_bytes() == baseline
finally:
    if previous is None:
        os.environ.pop("SCPN_MAX_DENSE_GIB", None)
    else:
        os.environ["SCPN_MAX_DENSE_GIB"] = previous
print(json.dumps({"installed_paths_verified": True, "ledger_recovered": True}))
"""
    result = subprocess.run(
        [str(python), "-I", "-c", program],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "installed_paths_verified": True,
        "ledger_recovered": True,
    }


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "dtype"])
def test_dense_consumer_refuses_corrupted_real_native_output_and_recovers(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport corruption after a real compiled result cannot escape FFI validation."""
    engine = import_module("scpn_quantum_engine")
    original = engine.build_xy_hamiltonian_dense
    completed: list[int] = []
    baseline = active_reserved_bytes()

    def corrupt(*arguments: object) -> object:
        result = np.asarray(original(*arguments))
        completed.append(result.size)
        if fault == "shape":
            return result.reshape(2, 2)
        if fault == "nonfinite":
            result[0] = np.inf
            return result
        return result.astype(np.complex128)

    with monkeypatch.context() as transport:
        transport.setattr(engine, "build_xy_hamiltonian_dense", corrupt)
        with pytest.raises(ValueError, match="malformed output shape|invalid numeric data"):
            knm_to_dense_matrix(np.zeros((1, 1)), np.array([1.0]), backend="rust")
    assert completed == [4]
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(
        knm_to_dense_matrix(np.zeros((1, 1)), np.array([1.0]), backend="rust"),
        np.diag([-1.0, 1.0]),
    )
    assert active_reserved_bytes() == baseline


def test_dense_native_entry_disappearing_after_admission_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loss of an actual compiled capability after admission refuses explicit native dispatch."""
    engine = import_module("scpn_quantum_engine")
    baseline = active_reserved_bytes()
    original_checkpoint = ExecutionMemoryReservation.checkpoint
    removed: list[bool] = []

    def checkpoint(owner: ExecutionMemoryReservation) -> None:
        original_checkpoint(owner)
        if not removed and active_reserved_bytes() > baseline:
            boundary.delattr(engine, "build_xy_hamiltonian_dense")
            removed.append(True)

    with monkeypatch.context() as boundary:
        boundary.setattr(ExecutionMemoryReservation, "checkpoint", checkpoint)
        with pytest.raises(RuntimeError, match="became unavailable"):
            knm_to_dense_matrix(np.zeros((1, 1)), np.array([1.0]), backend="rust")
    assert removed == [True]
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(
        knm_to_dense_matrix(np.zeros((1, 1)), np.array([1.0]), backend="rust"),
        np.diag([-1.0, 1.0]),
    )
    assert active_reserved_bytes() == baseline
