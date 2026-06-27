# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for hardware.provenance
"""Multi-angle tests for `hardware.provenance.capture_provenance`.

Closes the testing half of audit item C8. The `save_result`
side of the same item is tested through existing hardware-runner
tests; this file gates the provenance function itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scpn_quantum_control.hardware.provenance import capture_provenance


class TestCapturedFields:
    """Structural checks for the top-level provenance payload."""

    def test_all_top_level_keys_present(self) -> None:
        """The provenance payload exposes all required top-level groups."""
        prov = capture_provenance()
        for key in ("captured_at_utc", "git", "versions", "runtime"):
            assert key in prov, f"missing top-level provenance key: {key}"

    def test_git_block_shape(self) -> None:
        """The git block carries every stable provenance field."""
        git = capture_provenance()["git"]
        for key in ("commit", "short", "branch", "describe", "dirty"):
            assert key in git
        assert isinstance(git["dirty"], bool)

    def test_versions_block_shape(self) -> None:
        """The versions block records every package as a string value."""
        versions = capture_provenance()["versions"]
        for pkg in (
            "scpn_quantum_control",
            "scpn_quantum_engine",
            "qiskit",
            "qiskit_ibm_runtime",
            "numpy",
            "scipy",
        ):
            assert pkg in versions
            assert isinstance(versions[pkg], str)

    def test_runtime_block_shape(self) -> None:
        """The runtime block records Python, platform, machine, and host fields."""
        runtime = capture_provenance()["runtime"]
        for key in ("python", "implementation", "platform", "machine", "hostname"):
            assert key in runtime
            assert isinstance(runtime[key], str)


class TestFreshnessAndSerialisation:
    """Freshness and JSON compatibility checks for provenance payloads."""

    def test_timestamp_is_iso8601_utc(self) -> None:
        """The capture timestamp is ISO-8601 with an explicit UTC timezone."""
        ts = capture_provenance()["captured_at_utc"]
        # Must include a `T` separator and a timezone suffix
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_json_round_trip(self) -> None:
        """The payload survives a JSON encode/decode round trip exactly."""
        prov = capture_provenance()
        round_tripped = json.loads(json.dumps(prov))
        assert round_tripped == prov

    def test_python_version_is_dotted(self) -> None:
        """The Python version field uses a dotted interpreter version string."""
        v = capture_provenance()["runtime"]["python"]
        # e.g. 3.12.3
        parts = v.split(".")
        assert len(parts) >= 2
        for p in parts:
            assert p.isdigit() or p == "dev" or p.startswith("rc")


class TestHostnameAnonymisation:
    """Hostname anonymisation checks for provenance capture."""

    def test_raw_hostname_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The raw hostname is recorded when anonymisation is disabled."""
        monkeypatch.delenv("SCPN_ANONYMOUS_HOSTNAME", raising=False)
        # SCPNConfig is cached; flush before every env-dependent read.
        from scpn_quantum_control.config import reload_config

        reload_config()
        try:
            expected = socket.gethostname()
        except OSError:
            pytest.skip("gethostname unavailable")
        assert capture_provenance()["runtime"]["hostname"] == expected

    def test_hashed_hostname_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hostname is hashed when the anonymisation toggle is enabled."""
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
        # SCPNConfig is cached; flush so the new env var takes effect.
        from scpn_quantum_control.config import reload_config

        reload_config()
        host = capture_provenance()["runtime"]["hostname"]
        assert host.startswith("h")
        assert len(host) == 9  # "h" + 8 hex chars
        # Must be the first 8 chars of the SHA-256 of the real hostname
        try:
            real = socket.gethostname()
        except OSError:
            pytest.skip("gethostname unavailable")
        assert host[1:] == hashlib.sha256(real.encode("utf-8")).hexdigest()[:8]


class TestGracefulFailure:
    """Fail-closed checks for optional provenance dependencies."""

    def test_no_exception_when_git_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing git should produce unknown git fields instead of raising."""
        # Force a non-git directory by pointing the HEAD resolution at
        # an unlikely-to-be-a-repo path via PATH manipulation: remove
        # `git` from PATH so subprocess.run raises FileNotFoundError.
        monkeypatch.setenv("PATH", "")
        prov = capture_provenance()
        git = prov["git"]
        for key in ("commit", "short", "branch", "describe"):
            assert git[key] == "unknown"
        # dirty is a bool; with `git` missing the porcelain call returns
        # "unknown" which != "" so dirty evaluates True. The contract
        # is that the field is present, not that it reads False.
        assert isinstance(git["dirty"], bool)

    def test_git_command_uses_resolved_absolute_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The git probe should launch only an absolute admitted executable."""
        prov_mod = importlib.import_module("scpn_quantum_control.hardware.provenance")
        seen: list[Sequence[str]] = []

        def fake_run(
            command: Sequence[str],
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            seen.append(command)
            return subprocess.CompletedProcess(
                args=list(command),
                returncode=0,
                stdout="abc123\n",
                stderr="",
            )

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.subprocess.run",
            fake_run,
        )
        git_probe = cast(Callable[[str], str], prov_mod.__dict__["_git"])

        assert git_probe("rev-parse") == "abc123"
        assert seen
        executable = Path(seen[0][0])
        assert executable.is_absolute()
        assert executable.name == "git"

    def test_git_resolution_rejects_unresolvable_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken git lookup result is rejected before subprocess launch."""
        prov_mod = importlib.import_module("scpn_quantum_control.hardware.provenance")
        resolve_git = cast(Callable[[], str | None], prov_mod.__dict__["_resolve_git_executable"])

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.shutil.which",
            lambda _: "\0bad",
        )

        assert resolve_git() is None

    def test_git_resolution_rejects_non_executable_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A non-executable git path is rejected before subprocess launch."""
        prov_mod = importlib.import_module("scpn_quantum_control.hardware.provenance")
        resolve_git = cast(Callable[[], str | None], prov_mod.__dict__["_resolve_git_executable"])
        fake_git = tmp_path / "git"
        fake_git.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_git.chmod(0o600)

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.shutil.which",
            lambda _: str(fake_git),
        )

        assert resolve_git() is None

    def test_git_probe_returns_unknown_after_subprocess_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A subprocess error after admission is recorded as unknown."""
        prov_mod = importlib.import_module("scpn_quantum_control.hardware.provenance")
        git_probe = cast(Callable[[str], str], prov_mod.__dict__["_git"])

        def fail_run(command: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
            raise OSError(f"cannot execute {command[0]}")

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.subprocess.run",
            fail_run,
        )

        assert git_probe("rev-parse") == "unknown"

    def test_anonymous_hostname_falls_back_when_config_import_breaks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The anonymous-hostname toggle falls back to the legacy env var."""
        prov_mod = importlib.import_module("scpn_quantum_control.hardware.provenance")
        anonymous_enabled = cast(
            Callable[[], bool], prov_mod.__dict__["_anonymous_hostname_enabled"]
        )
        monkeypatch.setitem(sys.modules, "scpn_quantum_control.config", None)
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")

        assert anonymous_enabled() is True

    def test_missing_optional_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing optional engine is recorded without aborting capture."""
        # Simulate scpn_quantum_engine not installed by shadowing the
        # import. `importlib.import_module` is called inside
        # `_optional_engine_version`; raising ImportError there must
        # not abort the rest of the capture.

        def _raise(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.importlib.import_module", _raise
        )
        prov = capture_provenance()
        assert prov["versions"]["scpn_quantum_engine"] in (
            "not installed",
            "unknown",
        )

    def test_missing_package_metadata_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing package metadata is represented as a not-installed value."""
        metadata = importlib.import_module("importlib.metadata")

        original_version = importlib.metadata.version

        def version_or_missing(name: str) -> str:
            if name == "numpy":
                raise importlib.metadata.PackageNotFoundError(name)
            return original_version(name)

        monkeypatch.setattr(metadata, "version", version_or_missing)
        prov = capture_provenance()
        assert prov["versions"]["numpy"] == "not installed"

    def test_optional_engine_module_version_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An installed optional engine module version is preserved."""
        original_import_module = importlib.import_module

        def import_engine(name: str) -> object:
            if name == "scpn_quantum_engine":
                return SimpleNamespace(__version__="9.8.7")
            return original_import_module(name)

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.importlib.import_module",
            import_engine,
        )
        prov = capture_provenance()
        assert prov["versions"]["scpn_quantum_engine"] == "9.8.7"

    def test_hostname_lookup_failure_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hostname lookup failures are recorded as unknown."""

        def fail_hostname() -> str:
            raise OSError("hostname unavailable")

        monkeypatch.setattr(
            "scpn_quantum_control.hardware.provenance.socket.gethostname",
            fail_hostname,
        )
        prov = capture_provenance()
        assert prov["runtime"]["hostname"] == "unknown"
