# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive dispatch CLI tests
"""Tests for the ``scpn-studio-run`` executive dispatch CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
pytest.importorskip("qiskit", reason="qiskit not installed")

from scpn_quantum_control.studio.executive import ExecutionPlan  # noqa: E402
from scpn_quantum_control.studio.executive_cli import (  # noqa: E402
    EXIT_FAILED,
    EXIT_GATED,
    EXIT_REQUEST_ERROR,
    EXIT_SUCCEEDED,
    _load_parameters,
    build_default_registry,
    main,
    run,
)
from scpn_quantum_control.studio.executive_compile import (  # noqa: E402
    CompileActionHandler,
)

_COMPILE_PARAMS: dict[str, Any] = {
    "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
    "omega": [-0.1, 0.05, 0.05],
    "time": 0.1,
    "trotter_steps": 1,
    "trotter_order": 1,
}

_SIMULATE_PARAMS: dict[str, Any] = {
    "K_nm": [[0.0, 0.4], [0.4, 0.0]],
    "omega": [-0.1, 0.1],
    "t_max": 0.2,
    "dt": 0.1,
    "trotter_per_step": 1,
    "trotter_order": 1,
}

_EXECUTE_PARAMS: dict[str, Any] = {
    "provider": "ibm-quantum",
    "endpoint": "ibm_brisbane",
    "circuit_digest": "sha256:abc123",
    "circuit_ref": "data/studio/xy_compile_recompute_unit_20260708.json",
    "shots": 4096,
}


def _compile_argv(*extra: str) -> list[str]:
    return [
        "compile",
        "--action-id",
        "cli-compile-3node",
        "--params",
        json.dumps(_COMPILE_PARAMS),
        *extra,
    ]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_default_registry_registers_all_shipped_verbs() -> None:
    registry = build_default_registry()
    assert registry.verbs() == (
        "analyse",
        "benchmark",
        "compile",
        "differentiate",
        "execute",
        "simulate",
        "validate",
    )


# --------------------------------------------------------------------------- #
# end-to-end dispatch
# --------------------------------------------------------------------------- #
def test_cli_compile_succeeds_and_prints_sealed_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(_compile_argv()) == EXIT_SUCCEEDED
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["status"] == "succeeded"
    assert record["result"]["outputs"]["verified"] is True
    assert record["digest"].startswith("sha256:")
    assert record["script"]["filename"].endswith(".py")


def test_cli_preview_prints_plan_without_executing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "simulate",
        "--action-id",
        "cli-sim-preview",
        "--params",
        json.dumps(_SIMULATE_PARAMS),
        "--preview",
    ]
    assert run(argv) == EXIT_SUCCEEDED
    plan = json.loads(capsys.readouterr().out)
    assert plan["verb"] == "simulate"
    assert "claim_boundary" in plan
    assert "result" not in plan


def test_cli_execute_without_approval_is_gated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "execute",
        "--action-id",
        "cli-deploy",
        "--params",
        json.dumps(_EXECUTE_PARAMS),
    ]
    assert run(argv) == EXIT_GATED
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["result"]["status"] == "gated"
    assert "gated" in captured.err


def test_cli_execute_with_approval_builds_no_submit_dossier(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "execute",
        "--action-id",
        "cli-deploy",
        "--params",
        json.dumps(_EXECUTE_PARAMS),
        "--approve",
    ]
    assert run(argv) == EXIT_SUCCEEDED
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["status"] == "succeeded"
    assert record["result"]["outputs"]["submitted"] is False


def test_cli_writes_script_into_script_dir(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script_dir = tmp_path / "scripts"
    assert run(_compile_argv("--script-dir", str(script_dir))) == EXIT_SUCCEEDED
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    target = script_dir / record["script"]["filename"]
    assert target.read_text(encoding="utf-8") == record["script"]["source"]
    assert str(target) in captured.err


def test_cli_without_script_dir_writes_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert run(_compile_argv()) == EXIT_SUCCEEDED
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []


def test_cli_failed_execution_returns_failure_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: CompileActionHandler, plan: ExecutionPlan) -> Any:
        raise RuntimeError("synthetic execution fault")

    monkeypatch.setattr(CompileActionHandler, "execute", _boom)
    assert run(_compile_argv()) == EXIT_FAILED
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["result"]["status"] == "failed"
    assert "failed" in captured.err


# --------------------------------------------------------------------------- #
# request errors
# --------------------------------------------------------------------------- #
def test_cli_rejects_unknown_verb(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["teleport", "--action-id", "x", "--params", "{}"]
    assert run(argv) == EXIT_REQUEST_ERROR
    assert "error" in capsys.readouterr().err


def test_cli_rejects_undeclared_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(_compile_argv("--backend", "abacus")) == EXIT_REQUEST_ERROR
    assert "is not declared" in capsys.readouterr().err


def test_cli_rejects_invalid_json_params(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["compile", "--action-id", "x", "--params", "{not json"]
    assert run(argv) == EXIT_REQUEST_ERROR
    assert "not valid JSON" in capsys.readouterr().err


def test_cli_rejects_non_object_params(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["compile", "--action-id", "x", "--params", "[1, 2]"]
    assert run(argv) == EXIT_REQUEST_ERROR
    assert "JSON object" in capsys.readouterr().err


def test_cli_rejects_missing_params_file(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["compile", "--action-id", "x", "--params-file", "/nonexistent/params.json"]
    assert run(argv) == EXIT_REQUEST_ERROR
    assert "cannot read" in capsys.readouterr().err


def test_cli_reads_params_from_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    params_file = tmp_path / "compile.json"
    params_file.write_text(json.dumps(_COMPILE_PARAMS), encoding="utf-8")
    argv = ["compile", "--action-id", "cli-file", "--params-file", str(params_file)]
    assert run(argv) == EXIT_SUCCEEDED
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "argv",
    [
        ["compile", "--action-id", "x"],
        ["compile", "--action-id", "x", "--params", "{}", "--params-file", "p.json"],
    ],
)
def test_cli_requires_exactly_one_params_source(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(argv)
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# _load_parameters unit branches
# --------------------------------------------------------------------------- #
def test_load_parameters_inline_and_file(tmp_path: Path) -> None:
    assert _load_parameters('{"a": 1}', None) == {"a": 1}
    params_file = tmp_path / "p.json"
    params_file.write_text('{"b": 2}', encoding="utf-8")
    assert _load_parameters(None, str(params_file)) == {"b": 2}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def test_main_exits_with_run_code(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["scpn-studio-run", *_compile_argv()])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == EXIT_SUCCEEDED
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["status"] == "succeeded"
