# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Container memory checker contracts
"""Exercise the real checker entrypoint with controller files, not a container.

Positive cases use public dense admission unchanged. Fault injection separately
proves checker refusals; only hosted Docker execution proves kernel integration.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import scpn_quantum_control.dense_budget as dense_budget
from tools.check_container_memory_budget import main

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools/check_container_memory_budget.py"
LIMIT = 1024**3


@pytest.fixture
def controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use real flat controller files, leaving process hierarchy to its own tests."""
    monkeypatch.setattr(dense_budget, "DEFAULT_PROC_ROOT", tmp_path / "absent-proc")
    monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
    monkeypatch.delenv(dense_budget.DEFAULT_DENSE_BUDGET_ENV, raising=False)
    (tmp_path / "memory.max").write_text(str(LIMIT), encoding="utf-8")
    (tmp_path / "memory.current").write_text(str(LIMIT // 4), encoding="utf-8")
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, limit: str = str(LIMIT)) -> None:
    """Execute exactly the script entrypoint named by Docker's Python command."""
    monkeypatch.setattr(sys, "argv", [str(CHECKER), "--limit-bytes", limit])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(CHECKER), run_name="__main__")
    assert result.value.code == 0


@pytest.mark.parametrize("version", [1, 2])
def test_real_admission_with_controller_files(
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: int,
) -> None:
    """Both controller versions admit tiny metadata and refuse above-limit bytes."""
    if version == 1:
        (controller / "memory.max").unlink()
        memory = controller / "memory"
        memory.mkdir()
        (memory / "memory.limit_in_bytes").write_text(str(LIMIT), encoding="utf-8")
        (memory / "memory.usage_in_bytes").write_text(str(LIMIT // 4), encoding="utf-8")
    _run(monkeypatch)
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["cgroup_headroom_bytes"] == 3 * LIMIT // 4
    assert 0 < evidence["default_budget_bytes"] <= int(0.3 * 3 * LIMIT // 4)
    assert evidence["small_admitted_bytes"] == 2
    assert evidence["refused_bytes"] == 2 * LIMIT
    assert evidence["limit_bytes"] == LIMIT


@pytest.mark.parametrize(
    ("limit", "usage"),
    [("max", "0"), (str(LIMIT), "bad"), (str(LIMIT), str(LIMIT)), (str(4 * LIMIT), "0")],
)
def test_unknown_empty_or_oversized_allowance_refuses(
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    usage: str,
) -> None:
    """Do not certify missing headroom or a container larger than declared."""
    (controller / "memory.max").write_text(limit, encoding="utf-8")
    (controller / "memory.current").write_text(usage, encoding="utf-8")
    with pytest.raises(RuntimeError, match="positive cgroup headroom"):
        _run(monkeypatch)


def test_environment_override_refuses(
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit budget cannot masquerade as default discovery evidence."""
    monkeypatch.setenv(dense_budget.DEFAULT_DENSE_BUDGET_ENV, "1")
    with pytest.raises(RuntimeError, match="override masks"):
        _run(monkeypatch)


def test_explicit_arguments_with_controller_files(
    controller: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The callable CLI honours explicit byte arguments without global argv changes."""
    assert main(["--limit-bytes", str(LIMIT)]) == 0
    assert json.loads(capsys.readouterr().out)["limit_bytes"] == LIMIT


@pytest.mark.parametrize("budget", [0, LIMIT])
def test_bad_default_budget_is_detected(
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
) -> None:
    """Fault injection proves refusal of disabled or excessive default admission."""
    monkeypatch.setattr(dense_budget, "dense_budget_bytes", lambda: budget)
    with pytest.raises(RuntimeError, match="admission bound"):
        _run(monkeypatch)


def test_wrongly_admitted_large_estimate_is_detected(
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real explicit-override admission must not pass the default refusal check."""
    original = dense_budget.require_dense_allocation

    def overridden(
        n_qubits: int, *, rank: int, dtype: str
    ) -> dense_budget.DenseAllocationEstimate:
        """Inject a real override-backed admission, not fabricated estimate data."""
        return original(n_qubits, rank=rank, dtype=dtype, max_gib=8)

    monkeypatch.setattr(dense_budget, "require_dense_allocation", overridden)
    with pytest.raises(RuntimeError, match="above-limit dense allocation was admitted"):
        _run(monkeypatch)


@pytest.mark.parametrize("limit", ["0", "-1", "1048575", "8589934593", "bad"])
def test_invalid_cli_limit_exits_two(monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    """Invalid or unbounded operator limits fail before probing admission."""
    monkeypatch.setattr(sys, "argv", [str(CHECKER), "--limit-bytes", limit])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(CHECKER), run_name="__main__")
    assert result.value.code == 2


def test_process_help_imports_real_package() -> None:
    """The standalone process can import dependencies and exposes the required flag."""
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0
    assert "--limit-bytes" in result.stdout
    assert result.stderr == ""


def test_docker_workflow_runs_bounded_checker_before_suite() -> None:
    """Keep both real-container checks fail-closed ahead of the existing suite."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text())
    job = workflow["jobs"]["docker-test"]
    assert job["timeout-minutes"] == 45
    steps = job["steps"]
    check = next(
        step for step in steps if step.get("name") == "Verify memory-limited dense admission"
    )
    command = check["run"]
    assert check["timeout-minutes"] == 3
    assert "continue-on-error" not in check
    assert "if" not in check
    assert "set -euo pipefail" in command
    assert "for limit in 1073741824 2147483648" in command
    assert "docker run --rm --network none --cpus 1" in command
    assert '--memory "$limit" --memory-swap "$limit" --pids-limit 128' in command
    assert "--env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1" in command
    assert (
        'timeout 60s python tools/check_container_memory_budget.py --limit-bytes "$limit"'
        in command
    )
    names = [step.get("name") for step in steps]
    assert (
        names.index("Build Docker image") < steps.index(check) < names.index("Run tests in Docker")
    )
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dockerfile.splitlines()[:2] == [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
    ]
    assert dockerfile.splitlines()[6] == "# SCPN Quantum Control — Container reproduction image"
    assert "COPY tools/ tools/" in dockerfile
    assert "ENV PYTHONPATH=/app/src:/app/oscillatools/src:/app" in dockerfile
