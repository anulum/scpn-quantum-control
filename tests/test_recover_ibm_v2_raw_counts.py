# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the IBM v2 raw-count recovery script
"""Tests for scripts/recover_ibm_v2_raw_counts.py.

Exercises the pub-count extraction (success and the no-register error), the
per-job retrieval for iterable and single-result shapes, pack assembly and its
job-count guard, the fail-closed token-leak check, calibration retrieval
(available and absent), cluster enumeration with submission-order reversal and
the creation-date fallback, and the CLI dry-run and write paths against a fully
stubbed Qiskit Runtime service — no network or real identifiers anywhere.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from scripts import recover_ibm_v2_raw_counts as script

TOKEN = "secret-token-never-written"


class _Register:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class _PubData:
    def __init__(self, counts: dict[str, int] | None) -> None:
        if counts is not None:
            self.meas = _Register(counts)


class _Pub:
    def __init__(self, counts: dict[str, int] | None) -> None:
        self.data = _PubData(counts)


class _Job:
    def __init__(self, job_id: str, pubs: list[_Pub], created: str | None = "2026-03-29") -> None:
        self._id = job_id
        self._pubs = pubs
        self._created = created

    def job_id(self) -> str:
        return self._id

    def creation_date(self) -> str:
        if self._created is None:
            raise RuntimeError("no creation date")
        return self._created

    def result(self) -> list[_Pub]:
        return self._pubs


def _nine_jobs() -> list[_Job]:
    # newest-first order (as IBM returns); reversed by retrieve_cluster.
    return [_Job(f"job{i}", [_Pub({"0000": 900, "1111": 100})]) for i in range(9)]


class _Backend:
    def __init__(self, props: Any) -> None:
        self._props = props

    def properties(self, datetime: Any = None) -> Any:  # noqa: A002 - mirrors qiskit API
        return self._props


class _Service:
    def __init__(self, jobs: list[_Job], props: Any) -> None:
        self._jobs = jobs
        self._backend = _Backend(props)

    def jobs(self, **_kwargs: Any) -> list[_Job]:
        return self._jobs

    def backend(self, _name: str) -> _Backend:
        return self._backend


class TestCountsFromPub:
    def test_extracts_counts(self) -> None:
        assert script.counts_from_pub(_Pub({"0000": 5, "1111": 3})) == {"0000": 5, "1111": 3}

    def test_missing_register_raises(self) -> None:
        with pytest.raises(ValueError, match="no classical register"):
            script.counts_from_pub(_Pub(None))

    def test_skips_non_register_field(self) -> None:
        # a public attribute (alphabetically before the register) without
        # get_counts() must be skipped, not crash — dir() is sorted, so "aaa"
        # is examined before "meas" and exercises the hasattr-False branch.
        class _Data:
            def __init__(self) -> None:
                self.aaa = {"shots": 8192}
                self.meas = _Register({"0000": 4})

        pub = types.SimpleNamespace(data=_Data())
        assert script.counts_from_pub(pub) == {"0000": 4}


class TestJobRawCounts:
    def test_iterable_result(self) -> None:
        job = _Job("j", [_Pub({"0000": 1}), _Pub({"1111": 2})])
        assert script.job_raw_counts(job) == [{"0000": 1}, {"1111": 2}]

    def test_non_iterable_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        single = _Pub({"0000": 7})

        class _SingleJob:
            def result(self) -> _Pub:
                return single

        assert script.job_raw_counts(_SingleJob()) == [{"0000": 7}]


class TestBuildPack:
    def _records(self) -> list[dict[str, Any]]:
        return [
            {
                "ibm_job_id": f"id{i}",
                "creation_date": "2026-03-29",
                "per_pub_counts": [{"0000": 900, "1111": 100}],
            }
            for i in range(9)
        ]

    def test_assembles_nine_experiments(self) -> None:
        committed = {name: {"mean": 0.9} for name in script.EXPERIMENT_ORDER}
        pack = script.build_pack(self._records(), {"available": True}, committed)
        assert [e["experiment"] for e in pack["experiments"]] == list(script.EXPERIMENT_ORDER)
        assert pack["experiments"][0]["total_shots"] == 1000
        assert pack["experiments"][0]["committed_aggregate_mean"] == 0.9
        assert pack["calibration_snapshot"] == {"available": True}

    def test_wrong_job_count_raises(self) -> None:
        with pytest.raises(ValueError, match="expected 9 jobs"):
            script.build_pack(self._records()[:5], {}, {})


class TestLeakGuard:
    def test_token_present_raises(self) -> None:
        with pytest.raises(RuntimeError, match="API token found"):
            script.assert_no_token_leak({"x": f"prefix-{TOKEN}-suffix"}, TOKEN)

    def test_token_absent_passes(self) -> None:
        script.assert_no_token_leak({"x": "clean"}, TOKEN)

    def test_no_token_is_noop(self) -> None:
        script.assert_no_token_leak({"x": "anything"}, None)


class TestRetrieval:
    def test_retrieve_calibration_available(self) -> None:
        props = types.SimpleNamespace(last_update_date="2026-03-29", qubits=[0] * 156)
        cal = script.retrieve_calibration(_Service([], props))
        assert cal["available"] is True
        assert cal["num_qubits"] == 156

    def test_retrieve_calibration_absent(self) -> None:
        cal = script.retrieve_calibration(_Service([], None))
        assert cal["available"] is False

    def test_retrieve_cluster_reverses_and_reads(self) -> None:
        service = _Service(_nine_jobs(), None)
        records = script.retrieve_cluster(service)
        assert len(records) == 9
        # newest-first job8 becomes the last record; job0 the first (submission order)
        assert records[0]["ibm_job_id"] == "job8"
        assert records[-1]["ibm_job_id"] == "job0"
        assert records[0]["per_pub_counts"] == [{"0000": 900, "1111": 100}]

    def test_retrieve_cluster_creation_date_fallback(self) -> None:
        job = _Job("j", [_Pub({"0000": 1})], created=None)
        records = script.retrieve_cluster(_Service([job], None))
        assert records[0]["creation_date"] is None


class TestSha256:
    def test_digest(self, tmp_path: Path) -> None:
        f = tmp_path / "x.json"
        f.write_text("abc", encoding="utf-8")
        assert script.sha256_file(f) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )


class TestLoadCommittedAggregates:
    def test_reads_committed_v2(self) -> None:
        agg = script.load_committed_aggregates()
        # the committed aggregate-only pack carries the C-experiment means
        assert agg["C_fim"]["mean"] == pytest.approx(0.9158, abs=1e-3)
        assert agg["C_xy"]["mean"] == pytest.approx(0.8484, abs=1e-3)


class TestLoadRuntimeService:
    def test_builds_service_with_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class _FakeRuntimeService:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake_module = types.ModuleType("qiskit_ibm_runtime")
        fake_module.QiskitRuntimeService = _FakeRuntimeService  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "qiskit_ibm_runtime", fake_module)

        from scripts import prepare_s1_ibm_live_readiness as readiness

        monkeypatch.setattr(readiness, "_parse_vault", lambda _p: ("tok", "crn-instance"))
        vault = tmp_path / "vault.md"
        vault.write_text("stub", encoding="utf-8")
        service, token = script._load_runtime_service(vault, None)
        assert token == "tok"
        assert captured == {"channel": "ibm_cloud", "token": "tok", "instance": "crn-instance"}
        assert isinstance(service, _FakeRuntimeService)

    def test_no_vault_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeRuntimeService:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake_module = types.ModuleType("qiskit_ibm_runtime")
        fake_module.QiskitRuntimeService = _FakeRuntimeService  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "qiskit_ibm_runtime", fake_module)
        service, token = script._load_runtime_service(None, "crn-explicit")
        assert token is None
        assert captured == {"instance": "crn-explicit"}
        assert isinstance(service, _FakeRuntimeService)

    def test_token_without_instance(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeRuntimeService:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake_module = types.ModuleType("qiskit_ibm_runtime")
        fake_module.QiskitRuntimeService = _FakeRuntimeService  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "qiskit_ibm_runtime", fake_module)
        from scripts import prepare_s1_ibm_live_readiness as readiness

        monkeypatch.setattr(readiness, "_parse_vault", lambda _p: ("tok", None))
        vault = tmp_path / "vault.md"
        vault.write_text("stub", encoding="utf-8")
        _service, token = script._load_runtime_service(vault, None)
        assert token == "tok"
        # neither an explicit nor a vault instance -> no instance kwarg
        assert captured == {"channel": "ibm_cloud", "token": "tok"}


class TestCli:
    @pytest.fixture()
    def _stub_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        props = types.SimpleNamespace(last_update_date="2026-03-29", qubits=[0] * 156)
        service = _Service(_nine_jobs(), props)
        monkeypatch.setattr(script, "_load_runtime_service", lambda *_a, **_k: (service, TOKEN))
        committed = {name: {"mean": 0.9} for name in script.EXPERIMENT_ORDER}
        monkeypatch.setattr(script, "load_committed_aggregates", lambda *_a, **_k: committed)

    def test_dry_run(self, _stub_service: None, capsys: pytest.CaptureFixture[str]) -> None:
        assert script.main(["--dry-run"]) == 0
        assert "dry-run" in capsys.readouterr().out

    def test_writes_pack(
        self, _stub_service: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert script.main(["--output-dir", str(tmp_path)]) == 0
        pack = json.loads((tmp_path / "recovered_raw_counts.json").read_text(encoding="utf-8"))
        assert len(pack["experiments"]) == 9
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert "recovered_raw_counts.json" in manifest["files"]
        # the token must never reach disk
        assert TOKEN not in (tmp_path / "recovered_raw_counts.json").read_text(encoding="utf-8")
