# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the benchmark CLI
"""Policy and diff-summary branch tests for the scpn-bench CLI.

Covers the harness execution-policy guards, the missing-script handling, the
diff-summary clean and error paths and the console entry point.
"""

from __future__ import annotations

import dataclasses
import subprocess
from typing import Any

import pytest

from scpn_quantum_control import bench_cli as bench
from scpn_quantum_control.bench_cli import (
    HARNESS_REGISTRY,
    OFFLINE_HARNESS_POLICY,
    Harness,
    _print_diff_summary,
    _run_harness,
    _validate_harness_policy,
    main,
)


def _harness(*, script: str = "scripts/benchmark_rust_core_methods.py", **policy: Any) -> Harness:
    return Harness(
        label="probe",
        script=script,
        groups=frozenset({"methods"}),
        policy=dataclasses.replace(OFFLINE_HARNESS_POLICY, **policy),
    )


def test_policy_rejects_network_access() -> None:
    """A harness that allows network access is refused."""
    with pytest.raises(PermissionError, match="allows network access"):
        _validate_harness_policy(_harness(network_allowed=True))


def test_policy_rejects_credential_access() -> None:
    """A harness that allows credential access is refused."""
    with pytest.raises(PermissionError, match="allows credential access"):
        _validate_harness_policy(_harness(credential_allowed=True))


def test_policy_rejects_hardware_submission() -> None:
    """A harness that allows hardware submission is refused."""
    with pytest.raises(PermissionError, match="allows hardware submission"):
        _validate_harness_policy(_harness(hardware_submission_allowed=True))


def test_policy_rejects_disabled_subprocess() -> None:
    """A harness that disallows subprocess execution is refused."""
    with pytest.raises(PermissionError, match="disallows subprocess execution"):
        _validate_harness_policy(_harness(subprocess_allowed=False))


def test_policy_rejects_write_root_outside_repository() -> None:
    """An allowed write root resolving outside the repository is refused."""
    with pytest.raises(ValueError, match="allowed write root must stay inside repository"):
        _validate_harness_policy(_harness(allowed_write_roots=("../../../../tmp",)))


def test_run_harness_returns_two_for_missing_script() -> None:
    """A missing harness script is reported with exit code 2."""
    assert _run_harness(_harness(script="scripts/does_not_exist_harness.py")) == 2


def test_print_diff_summary_reports_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean artefact tree returns success."""

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("scpn_quantum_control.bench_cli.subprocess.run", _fake_run)
    assert _print_diff_summary() == 0


def test_print_diff_summary_propagates_name_only_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing name-only git diff propagates its non-zero return code."""

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        code = 3 if "--name-only" in command else 0
        return subprocess.CompletedProcess(command, code, stdout="", stderr="")

    monkeypatch.setattr("scpn_quantum_control.bench_cli.subprocess.run", _fake_run)
    assert _print_diff_summary() == 3


def test_main_delegates_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console entry point delegates to run()."""
    monkeypatch.setattr(bench, "run", lambda: 0)
    assert main() == 0


def test_registry_scripts_exist_for_offline_harnesses() -> None:
    """Sanity: the first registry harness ships an on-disk script."""
    first = HARNESS_REGISTRY[0]
    assert (bench.REPO_ROOT / first.script).exists()
