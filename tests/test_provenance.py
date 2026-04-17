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
import json
import socket

import pytest

from scpn_quantum_control.hardware.provenance import capture_provenance


class TestCapturedFields:
    def test_all_top_level_keys_present(self):
        prov = capture_provenance()
        for key in ("captured_at_utc", "git", "versions", "runtime"):
            assert key in prov, f"missing top-level provenance key: {key}"

    def test_git_block_shape(self):
        git = capture_provenance()["git"]
        for key in ("commit", "short", "branch", "describe", "dirty"):
            assert key in git
        assert isinstance(git["dirty"], bool)

    def test_versions_block_shape(self):
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

    def test_runtime_block_shape(self):
        runtime = capture_provenance()["runtime"]
        for key in ("python", "implementation", "platform", "machine", "hostname"):
            assert key in runtime
            assert isinstance(runtime[key], str)


class TestFreshnessAndSerialisation:
    def test_timestamp_is_iso8601_utc(self):
        ts = capture_provenance()["captured_at_utc"]
        # Must include a `T` separator and a timezone suffix
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_json_round_trip(self):
        prov = capture_provenance()
        round_tripped = json.loads(json.dumps(prov))
        assert round_tripped == prov

    def test_python_version_is_dotted(self):
        v = capture_provenance()["runtime"]["python"]
        # e.g. 3.12.3
        parts = v.split(".")
        assert len(parts) >= 2
        for p in parts:
            assert p.isdigit() or p == "dev" or p.startswith("rc")


class TestHostnameAnonymisation:
    def test_raw_hostname_by_default(self, monkeypatch):
        monkeypatch.delenv("SCPN_ANONYMOUS_HOSTNAME", raising=False)
        try:
            expected = socket.gethostname()
        except OSError:
            pytest.skip("gethostname unavailable")
        assert capture_provenance()["runtime"]["hostname"] == expected

    def test_hashed_hostname_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
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
    def test_no_exception_when_git_absent(self, monkeypatch):
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

    def test_missing_optional_engine(self, monkeypatch):
        # Simulate scpn_quantum_engine not installed by shadowing the
        # import. `importlib.import_module` is called inside
        # `_optional_engine_version`; raising ImportError there must
        # not abort the rest of the capture.
        import scpn_quantum_control.hardware.provenance as prov_mod

        def _raise(name: str):
            raise ImportError(name)

        monkeypatch.setattr(prov_mod.importlib, "import_module", _raise)
        prov = capture_provenance()
        assert prov["versions"]["scpn_quantum_engine"] in (
            "not installed",
            "unknown",
        )
