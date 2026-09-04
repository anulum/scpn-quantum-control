# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the IQM DLA submission journal
"""Contract tests for the IQM DLA runner's crash-safe submission journal.

The tests exercise the real CLI and filesystem state machine with an injected
provider boundary. They prove that a process interruption or ambiguous network
failure cannot silently trigger a second paid submission, and that recovery
accepts only an exact provider-payload digest match.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import qiskit

from scripts import iqm_depth_ordering_protocol as protocol
from scripts import run_iqm_dla_powered_block as script


class FakeCircuit:
    """Minimal transpiled circuit with deterministic IQM serialisation."""

    def __init__(self, name: str = "", depth: int = 20) -> None:
        self.name = name
        self._depth = depth

    def depth(self) -> int:
        """Return the injected transpiled depth."""
        return self._depth

    def count_ops(self) -> dict[str, int]:
        """Return a stable operation inventory for dry-run compatibility."""
        return {"cz": 2, "measure": 4}

    def model_dump(self, *, mode: str, exclude: set[str]) -> dict[str, object]:
        """Return a JSON-ready representation used by the payload digest."""
        assert mode == "json"
        assert "move_validation_mode" in exclude
        return {"name": self.name, "depth": self._depth}


@dataclass(frozen=True)
class FakeIQMOperation:
    """Dataclass-shaped IQM native operation used by the installed client."""

    name: str
    locus: tuple[str, ...]
    args: dict[str, float]


@dataclass(frozen=True)
class FakeIQMCircuit:
    """Dataclass-shaped IQM circuit used to exercise live payload serialisation."""

    name: str
    instructions: tuple[FakeIQMOperation, ...]
    metadata: dict[str, str] | None = None


class FakeHelper:
    """Build the named campaign circuit through the production helper contract."""

    @staticmethod
    def _build_circuit(payload: dict[str, Any]) -> FakeCircuit:
        return FakeCircuit(str(payload["circuit_name"]))


class FakeParameters:
    """Execution parameters returned by a provider job payload."""

    def __init__(self, shots: int) -> None:
        self.calibration_set_id = UUID(int=99)
        self.qubit_mapping = None
        self.shots = shots
        self.max_circuit_duration_over_t2 = None
        self.heralding_mode = "none"
        self.move_gate_validation = "strict"
        self.move_gate_frame_tracking = "full"
        self.active_reset_cycles = None
        self.dd_mode = "disabled"
        self.dd_strategy = None


class FakeRunRequest(FakeParameters):
    """Validated provider request created before submission."""

    def __init__(self, circuits: list[FakeCircuit], shots: int) -> None:
        super().__init__(shots)
        self.circuits = circuits


class FakeProviderJob:
    """Provider job that exposes a UUID and its exact submitted payload."""

    def __init__(self, job_id: UUID, request: FakeRunRequest) -> None:
        self.job_id = job_id
        self._request = request

    def payload(self) -> tuple[list[object], object]:
        """Return the circuits and parameters for recovery verification."""
        return list(self._request.circuits), FakeParameters(self._request.shots)


class FakeClient:
    """IQM client fake with deterministic failures at provider-call boundaries."""

    def __init__(self) -> None:
        self.calls = 0
        self.failures: dict[int, BaseException] = {}
        self.forced_job_ids: dict[int, UUID] = {}
        self.requests: dict[int, FakeRunRequest] = {}
        self.jobs: dict[UUID, FakeProviderJob] = {}

    def submit_run_request(
        self, run_request: object, use_timeslot: bool = False
    ) -> FakeProviderJob:
        """Record the exact request, then succeed or fail at the selected call."""
        assert use_timeslot is False
        assert isinstance(run_request, FakeRunRequest)
        self.calls += 1
        self.requests[self.calls] = run_request
        failure = self.failures.get(self.calls)
        if failure is not None:
            raise failure
        job_id = self.forced_job_ids.get(self.calls, UUID(int=self.calls))
        job = FakeProviderJob(job_id, run_request)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: UUID) -> FakeProviderJob:
        """Return the dashboard-selected job for recovery verification."""
        return self.jobs[job_id]


class FakeBackend:
    """Live-backend fake retaining the production request-building boundary."""

    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def create_run_request(self, run_input: list[Any], *, shots: int) -> FakeRunRequest:
        """Build the validated payload without provider submission."""
        assert all(isinstance(circuit, FakeCircuit) for circuit in run_input)
        return FakeRunRequest(list(run_input), shots)

    def retrieve_job(self, job_id: str) -> Any:
        """Retrieval is outside the submit-journal tests."""
        raise AssertionError(f"unexpected retrieval of {job_id}")


class FakeCountsResult:
    """Result object returning one or several exact count maps."""

    def __init__(self, counts: dict[str, int] | list[dict[str, int]]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int] | list[dict[str, int]]:
        """Return the injected count payload."""
        return self._counts


class FakePollingJob:
    """Polling job with deterministic completion and result behaviour."""

    def __init__(
        self,
        counts: dict[str, int] | list[dict[str, int]],
        *,
        waits: int = 0,
    ) -> None:
        self._result = FakeCountsResult(counts)
        self._waits = waits
        self._polls = 0

    def done(self) -> bool:
        """Reach completion after the selected number of polls."""
        self._polls += 1
        return self._polls > self._waits

    def status(self) -> str:
        """Return the queued status used in progress output."""
        return "QUEUED"

    def result(self) -> FakeCountsResult:
        """Return the injected result."""
        return self._result


class FakeRetrievalBackend:
    """Backend exposing only pre-existing polling jobs."""

    def __init__(self, jobs: dict[str, FakePollingJob]) -> None:
        self.jobs = jobs

    def retrieve_job(self, job_id: str) -> FakePollingJob:
        """Return the selected polling job."""
        return self.jobs[job_id]


class FakeDryBackend:
    """Noisy-simulator fake used through the real dry-run CLI."""

    def run(self, _circuit: FakeCircuit, *, shots: int) -> FakePollingJob:
        """Return shot-conserving deterministic counts."""
        return FakePollingJob({"0000": shots})


def identity_transpile(circuit: FakeCircuit, **kwargs: object) -> FakeCircuit:
    """Preserve the fake circuit while asserting the live layout is supplied."""
    assert kwargs["initial_layout"] in (
        list(script.PRIMARY_LAYOUT),
        list(script.FALLBACK_LAYOUT),
    )
    assert kwargs["optimization_level"] == 1
    return circuit


@pytest.fixture()
def hardware_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeClient, FakeBackend]:
    """Inject one provider boundary while leaving the production CLI intact."""
    client = FakeClient()
    backend = FakeBackend(client)
    monkeypatch.setattr(script, "_load_helper", lambda: FakeHelper())
    monkeypatch.setattr(script, "_live_backend", lambda _name: backend)
    monkeypatch.setattr(qiskit, "transpile", identity_transpile)
    return client, backend


def submit_args(path: Path, *extra: str) -> list[str]:
    """Return the exact window-variability submission arguments used by the tests."""
    return [
        "submit",
        "--campaign",
        "window-variability",
        "--quantum-computer",
        "garnet",
        "--window",
        "3",
        "--date",
        "2026-07-26",
        "--out",
        str(path),
        *extra,
    ]


def powered_epoch_args(
    path: Path,
    calibration_path: Path | None,
    *,
    epoch: int = 1,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Return owner-gated arguments for one confirmatory calibration epoch."""
    args = [
        "submit",
        "--campaign",
        "depth-profile-powered-epoch",
        "--quantum-computer",
        "garnet",
        "--epoch",
        str(epoch),
        "--date",
        "2026-09-05",
        "--out",
        str(path),
        "--i-have-owner-go",
    ]
    if calibration_path is not None:
        args.extend(("--calibration", str(calibration_path)))
    args.extend(extra)
    return args


def write_calibration(path: Path, calibration_id: UUID, *, date: str = "2026-09-05") -> None:
    """Write a structurally complete primary-layout calibration snapshot."""
    path.write_text(
        json.dumps(
            {
                "source": "IQM Resonance garnet",
                "date": date,
                "calibration_set_id": str(calibration_id),
                "calibration": {
                    "num_qubits": 20,
                    "edges": [[2, 7], [7, 12], [12, 13]],
                    "edge_fidelity": {},
                    "readout_error": {},
                },
            }
        ),
        encoding="utf-8",
    )


def write_epoch_counts(
    path: Path,
    calibration_id: UUID,
    *,
    epoch: int,
    date: str = "2026-09-05",
) -> None:
    """Write one complete retrieved confirmatory count matrix."""
    counts: dict[str, dict[str, int]] = {}
    for label in protocol.expected_count_labels():
        shots = protocol.READOUT_SHOTS if label.startswith("readout_") else protocol.MAIN_SHOTS
        counts[label] = {"0000": shots - 1, "0001": 1}
    path.write_text(
        json.dumps(
            {
                "schema": protocol.RETRIEVED_COUNTS_SCHEMA,
                "campaign": protocol.CAMPAIGN_ID,
                "backend": "garnet",
                "date": date,
                "repetition": 1,
                "window": 0,
                "epoch": epoch,
                "calibration_set_id": str(calibration_id),
                "design_sha256": protocol.FROZEN_DESIGN_SHA256,
                "layout": list(script.PRIMARY_LAYOUT),
                "job_ids": [str(UUID(int=epoch * 2 + 10)), str(UUID(int=epoch * 2 + 11))],
                "counts": counts,
            }
        ),
        encoding="utf-8",
    )


def load_record(path: Path) -> dict[str, Any]:
    """Load a test journal as an object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def journal_job(
    *,
    group_id: str = "main",
    state: str = "prepared",
    job_id: str | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Return one structurally valid version-two journal group."""
    entry: dict[str, object] = {
        "group_id": group_id,
        "state": state,
        "job_id": job_id,
        "shots": 1024 if group_id == "main" else 2048,
        "labels": [f"{group_id}-label"],
        "circuit_names": [f"{group_id}-circuit"],
        "payload_sha256": "0" * 64,
    }
    entry.update(overrides)
    return entry


class TestFrozenMatrix:
    def test_window_matrix_and_budget_are_exact(self) -> None:
        rows = script.build_powered_plan(
            layout=script.PRIMARY_LAYOUT,
            depths=script.CAMPAIGNS["window-variability"]["depths"],
        )
        mains = [row for row in rows if row["kind"] == "dla_parity"]
        readout = [row for row in rows if row["kind"] == "readout_baseline"]
        assert len(mains) == 32
        assert len(readout) == 4
        assert sum(int(row["shots"]) for row in rows) == 40_960

    def test_powered_depth_ordering_matrix_and_budget_are_exact(self) -> None:
        campaign = script.CAMPAIGNS["depth-profile-powered-epoch"]
        rows = script.build_powered_plan(
            layout=script.PRIMARY_LAYOUT,
            depths=campaign["depths"],
            repetitions=campaign["repetitions"],
        )
        assert len([row for row in rows if row["kind"] == "dla_parity"]) == 48
        assert len([row for row in rows if row["kind"] == "readout_baseline"]) == 4
        assert sum(int(row["shots"]) for row in rows) == 57_344


class TestDryRunAndBootstrap:
    def test_real_helper_loads_and_missing_spec_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        helper = script._load_helper()
        assert callable(helper._build_circuit)
        monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args: None)
        with pytest.raises(RuntimeError, match="cannot load IQM circuit helper"):
            script._load_helper()

    def test_dry_run_writes_full_window_and_reuses_unique_circuits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(script, "_load_helper", lambda: FakeHelper())
        monkeypatch.setattr(script, "_fake_backend", lambda: FakeDryBackend())
        monkeypatch.setattr(qiskit, "transpile", identity_transpile)
        output = tmp_path / "dry.json"
        assert (
            script.main(
                [
                    "dry-run",
                    "--campaign",
                    "window-variability",
                    "--layout",
                    "fallback",
                    "--date",
                    "2026-07-26",
                    "--out",
                    str(output),
                ]
            )
            == 0
        )
        payload = load_record(output)
        assert payload["circuit_count"] == 36
        assert payload["shot_count"] == 40_960
        assert payload["layout"] == list(script.FALLBACK_LAYOUT)
        assert len({row["circuit_name"] for row in payload["records"]}) == 12
        assert len(payload["counts"]) == 36
        assert "all circuits inside" in capsys.readouterr().out

    def test_dry_run_reports_envelope_violation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(script, "_load_helper", lambda: FakeHelper())
        monkeypatch.setattr(script, "_fake_backend", lambda: FakeDryBackend())

        def too_deep(circuit: FakeCircuit, **_kwargs: object) -> FakeCircuit:
            circuit._depth = 999
            return circuit

        monkeypatch.setattr(qiskit, "transpile", too_deep)
        output = tmp_path / "dry.json"
        assert (
            script.main(
                [
                    "dry-run",
                    "--date",
                    "2026-07-26",
                    "--out",
                    str(output),
                ]
            )
            == 1
        )
        assert load_record(output)["envelope_violations"]
        assert "DEPTH ENVELOPE VIOLATIONS" in capsys.readouterr().err

    def test_powered_depth_ordering_dry_run_covers_the_frozen_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(script, "_load_helper", lambda: FakeHelper())
        monkeypatch.setattr(script, "_fake_backend", lambda: FakeDryBackend())
        monkeypatch.setattr(qiskit, "transpile", identity_transpile)
        output = tmp_path / "powered-depth-ordering-readiness.json"

        assert (
            script.main(
                [
                    "dry-run",
                    "--campaign",
                    "depth-profile-powered-epoch",
                    "--date",
                    "2026-09-05",
                    "--out",
                    str(output),
                ]
            )
            == 0
        )

        payload = load_record(output)
        assert payload["circuit_count"] == 52
        assert payload["shot_count"] == 57_344
        assert payload["design_sha256"] == protocol.FROZEN_DESIGN_SHA256
        assert len(payload["counts"]) == 52

    def test_powered_depth_ordering_dry_run_rejects_design_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Changed frozen-design bytes stop before simulator execution."""
        monkeypatch.setattr(script, "_load_helper", lambda: FakeHelper())

        def reject_design(_path: Path) -> protocol.FrozenDesign:
            raise ValueError("changed design")

        monkeypatch.setattr(protocol, "validate_frozen_design", reject_design)
        output = tmp_path / "powered-depth-ordering-readiness.json"
        assert (
            script.main(
                [
                    "dry-run",
                    "--campaign",
                    "depth-profile-powered-epoch",
                    "--date",
                    "2026-09-05",
                    "--out",
                    str(output),
                ]
            )
            == 2
        )
        assert "No readiness simulation was attempted" in capsys.readouterr().err
        assert not output.exists()

    def test_credentials_and_dynamic_backend_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        vault = tmp_path / "credentials.md"
        vault.write_text(
            "# Vault\n## Other\n- token: ignored\n## IQM Resonance\n"
            "- URL: https://example.invalid\n- Token: secret\n## Next\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(script, "VAULT_PATH", vault)
        assert script._load_credentials() == ("https://example.invalid", "secret")

        backend = object()

        class FakeProvider:
            def __init__(self, url: str, *, quantum_computer: str, token: str) -> None:
                assert (url, quantum_computer, token) == (
                    "https://example.invalid",
                    "garnet",
                    "secret",
                )

            def get_backend(self) -> object:
                return backend

        module = types.SimpleNamespace(IQMProvider=FakeProvider)
        monkeypatch.setattr(importlib, "import_module", lambda _name: module)
        assert script._live_backend("garnet") is backend

        vault.write_text("## IQM Resonance\n- URL: https://example.invalid\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="missing IQM Resonance"):
            script._load_credentials()

    def test_dynamic_fake_backend_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = object()
        module = types.SimpleNamespace(IQMFakeGarnet=lambda: backend)
        monkeypatch.setattr(importlib, "import_module", lambda _name: module)
        assert script._fake_backend() is backend


class TestSubmissionJournal:
    def test_payload_digest_serialises_installed_iqm_dataclass_shape(self) -> None:
        circuit = FakeIQMCircuit(
            "native",
            (FakeIQMOperation("prx", ("QB1",), {"angle": 0.5}),),
            {"purpose": "journal"},
        )
        parameters = FakeParameters(1024)
        first = script._payload_digest([circuit], parameters)
        second = script._payload_digest([circuit], parameters)
        assert first == second
        assert len(first) == 64

    def test_owner_go_and_window_range_fail_before_provider_contact(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        record = tmp_path / "submission.json"
        assert script.main(submit_args(record)) == 2
        assert "owner GO" in capsys.readouterr().err
        invalid = submit_args(record, "--i-have-owner-go")
        invalid[invalid.index("3")] = "11"
        assert script.main(invalid) == 2
        assert "range 1..10" in capsys.readouterr().err
        assert client.calls == 0
        assert not record.exists()

    def test_first_powered_epoch_binds_design_and_calibration_before_submit(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
    ) -> None:
        client, _ = hardware_boundary
        calibration = tmp_path / "calibration.json"
        write_calibration(calibration, UUID(int=99))
        record_path = tmp_path / "powered-epoch.json"

        assert script.main(powered_epoch_args(record_path, calibration)) == 0

        record = load_record(record_path)
        assert record["epoch"] == 1
        assert record["calibration_set_id"] == str(UUID(int=99))
        assert record["design_sha256"] == protocol.FROZEN_DESIGN_SHA256
        assert [(entry["group_id"], len(entry["labels"])) for entry in record["jobs"]] == [
            ("main", 48),
            ("readout", 4),
        ]
        assert client.calls == 2

    def test_powered_epoch_requires_calibration_before_provider_contact(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The confirmatory campaign cannot submit without a fresh snapshot."""
        client, _ = hardware_boundary
        record_path = tmp_path / "powered-epoch.json"
        assert script.main(powered_epoch_args(record_path, None)) == 2
        assert "requires --calibration" in capsys.readouterr().err
        assert client.calls == 0
        assert not record_path.exists()

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda payload: payload.update(date="2026-09-06"), "dates differ"),
            (lambda payload: payload["calibration"].update(edges=[]), "lacks primary-layout"),
        ],
    )
    def test_powered_epoch_rejects_invalid_calibration_without_provider_contact(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        mutate: Any,
        message: str,
    ) -> None:
        client, _ = hardware_boundary
        calibration = tmp_path / "calibration.json"
        write_calibration(calibration, UUID(int=99))
        payload = load_record(calibration)
        mutate(payload)
        calibration.write_text(json.dumps(payload), encoding="utf-8")
        record_path = tmp_path / "powered-epoch.json"

        assert script.main(powered_epoch_args(record_path, calibration)) == 2
        assert message in capsys.readouterr().err
        assert client.calls == 0
        assert not record_path.exists()

    def test_powered_epoch_rejects_design_evidence_and_live_payload_drift(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        design = load_record(protocol.DESIGN_PATH)
        excluded_id = UUID(design["source"]["excluded_calibration_set_ids"][0])
        calibration = tmp_path / "calibration.json"
        write_calibration(calibration, excluded_id)
        record_path = tmp_path / "powered-epoch.json"

        assert script.main(powered_epoch_args(record_path, calibration)) == 2
        assert "used as design evidence" in capsys.readouterr().err
        assert client.calls == 0

        write_calibration(calibration, UUID(int=100))
        assert script.main(powered_epoch_args(record_path, calibration)) == 2
        assert "provider payload calibration set differs" in capsys.readouterr().err
        assert client.calls == 0
        assert not record_path.exists()

    def test_later_powered_epoch_requires_complete_prior_retrieval_custody(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        prior_calibration = tmp_path / "prior-calibration.json"
        prior_counts = tmp_path / "prior-counts.json"
        current_calibration = tmp_path / "current-calibration.json"
        write_calibration(prior_calibration, UUID(int=98))
        write_epoch_counts(prior_counts, UUID(int=98), epoch=1)
        write_calibration(current_calibration, UUID(int=99))
        record_path = tmp_path / "powered-epoch-2.json"
        args = powered_epoch_args(
            record_path,
            current_calibration,
            epoch=2,
            extra=(
                "--prior-calibrations",
                str(prior_calibration),
                "--prior-epoch-counts",
                str(prior_counts),
            ),
        )

        broken = load_record(prior_counts)
        broken["design_sha256"] = "0" * 64
        prior_counts.write_text(json.dumps(broken), encoding="utf-8")
        assert script.main(args) == 2
        assert "wrong design_sha256" in capsys.readouterr().err
        assert client.calls == 0

        write_epoch_counts(prior_counts, UUID(int=98), epoch=1)
        assert script.main(args) == 0
        assert load_record(record_path)["epoch"] == 2
        assert client.calls == 2

    def test_depth_violation_fails_before_journal_or_provider_call(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary

        def too_deep(circuit: FakeCircuit, **_kwargs: object) -> FakeCircuit:
            circuit._depth = 999
            return circuit

        monkeypatch.setattr(qiskit, "transpile", too_deep)
        record = tmp_path / "submission.json"
        assert script.main(submit_args(record, "--i-have-owner-go")) == 1
        assert "DEPTH ENVELOPE VIOLATION" in capsys.readouterr().err
        assert client.calls == 0
        assert not record.exists()

    def test_success_persists_each_group_and_restart_is_contact_free(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        record_path = tmp_path / "submission.json"
        args = submit_args(record_path, "--i-have-owner-go")
        assert script.main(args) == 0
        record = load_record(record_path)
        assert record["schema"] == script.JOURNAL_SCHEMA
        assert record["status"] == "submitted"
        assert [(job["group_id"], job["shots"], len(job["labels"])) for job in record["jobs"]] == [
            ("main", 1024, 32),
            ("readout", 2048, 4),
        ]
        assert all(job["state"] == "submitted" for job in record["jobs"])
        assert all(len(job["payload_sha256"]) == 64 for job in record["jobs"])
        assert client.calls == 2

        def forbidden_provider_contact(_name: str) -> FakeBackend:
            raise AssertionError("completed journal must not contact IQM")

        monkeypatch.setattr(script, "_live_backend", forbidden_provider_contact)
        assert script.main(args) == 0
        assert "without provider contact" in capsys.readouterr().out
        assert client.calls == 2

    def test_non_batch_repetition_submits_mains_only(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
    ) -> None:
        client, _ = hardware_boundary
        record_path = tmp_path / "powered-rep2.json"
        assert (
            script.main(
                [
                    "submit",
                    "--campaign",
                    "powered",
                    "--quantum-computer",
                    "garnet",
                    "--repetition",
                    "2",
                    "--date",
                    "2026-07-26",
                    "--out",
                    str(record_path),
                    "--i-have-owner-go",
                ]
            )
            == 0
        )
        record = load_record(record_path)
        assert [(job["group_id"], len(job["labels"])) for job in record["jobs"]] == [("main", 6)]
        assert client.calls == 1

    def test_process_interrupt_leaves_ambiguous_group_and_restart_refuses(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.failures[2] = KeyboardInterrupt()
        record_path = tmp_path / "submission.json"
        args = submit_args(record_path, "--i-have-owner-go")
        with pytest.raises(KeyboardInterrupt):
            script.main(args)
        record = load_record(record_path)
        assert [(job["group_id"], job["state"]) for job in record["jobs"]] == [
            ("main", "submitted"),
            ("readout", "submitting"),
        ]
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "No provider submission was attempted" in capsys.readouterr().err
        assert client.calls == 2

    def test_ambiguous_exception_recovers_exact_job_then_resumes_remaining_group(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.failures[1] = TimeoutError("response lost")
        record_path = tmp_path / "submission.json"
        args = submit_args(record_path, "--i-have-owner-go")
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        record = load_record(record_path)
        assert record["jobs"][0]["state"] == "recovery_required"
        assert "Do not resubmit" in capsys.readouterr().err
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert client.calls == 1

        recovered_id = UUID(int=101)
        client.jobs[recovered_id] = FakeProviderJob(recovered_id, client.requests[1])
        assert (
            script.main(
                [
                    "recover",
                    "--record",
                    str(record_path),
                    "--group",
                    "main",
                    "--job-id",
                    str(recovered_id),
                ]
            )
            == 2
        )
        assert "requires --i-confirm-provider-job" in capsys.readouterr().err
        assert (
            script.main(
                [
                    "recover",
                    "--record",
                    str(record_path),
                    "--group",
                    "main",
                    "--job-id",
                    str(recovered_id),
                    "--i-confirm-provider-job",
                ]
            )
            == 0
        )
        assert load_record(record_path)["status"] == "partially_submitted"

        client.failures.clear()
        assert script.main(args) == 0
        final = load_record(record_path)
        assert final["status"] == "submitted"
        assert final["jobs"][0]["job_id"] == str(recovered_id)
        assert final["jobs"][0]["recovery_method"] == "provider_payload_digest_match"
        assert client.calls == 2

    def test_recovery_rejects_mismatched_payload_and_duplicate_binding(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.failures[1] = TimeoutError("response lost")
        record_path = tmp_path / "submission.json"
        assert (
            script.main(submit_args(record_path, "--i-have-owner-go"))
            == script.RECOVERY_REQUIRED_EXIT
        )
        wrong_id = UUID(int=202)
        wrong = FakeRunRequest([FakeCircuit("different")], shots=1024)
        client.jobs[wrong_id] = FakeProviderJob(wrong_id, wrong)
        recovery = [
            "recover",
            "--record",
            str(record_path),
            "--group",
            "main",
            "--job-id",
            str(wrong_id),
            "--i-confirm-provider-job",
        ]
        assert script.main(recovery) == script.RECOVERY_REQUIRED_EXIT
        assert "does not match" in capsys.readouterr().err
        assert load_record(record_path)["jobs"][0]["state"] == "recovery_required"

    def test_provider_duplicate_job_id_fails_closed(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.forced_job_ids[2] = UUID(int=1)
        record_path = tmp_path / "submission.json"
        assert (
            script.main(submit_args(record_path, "--i-have-owner-go"))
            == script.RECOVERY_REQUIRED_EXIT
        )
        record = load_record(record_path)
        assert record["jobs"][0]["state"] == "submitted"
        assert record["jobs"][1]["state"] == "recovery_required"
        assert "reused job ID" in capsys.readouterr().err

    def test_existing_legacy_record_is_never_overwritten(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        record_path = tmp_path / "submission.json"
        original = {"campaign": "legacy", "jobs": [{"job_id": "old"}]}
        record_path.write_text(json.dumps(original), encoding="utf-8")
        assert (
            script.main(submit_args(record_path, "--i-have-owner-go"))
            == script.RECOVERY_REQUIRED_EXIT
        )
        assert "existing submission journal is invalid" in capsys.readouterr().err
        assert load_record(record_path) == original
        assert client.calls == 0


class TestJournalValidation:
    def test_journal_lock_uses_linked_worktree_common_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        common_dir = tmp_path / "main.git"
        worktree_admin = common_dir / "worktrees" / "candidate"
        worktree_admin.mkdir(parents=True)
        (worktree_admin / "commondir").write_text("../..\n", encoding="utf-8")
        worktree_root = tmp_path / "candidate"
        worktree_root.mkdir()
        (worktree_root / ".git").write_text(f"gitdir: {worktree_admin}\n", encoding="utf-8")
        monkeypatch.setattr(script, "REPO_ROOT", worktree_root)

        journal = tmp_path / "submission.json"
        with script._journal_lock(journal):
            lock_root = common_dir / "scpn-qpu-journal-locks"
            assert lock_root.is_dir()
            assert len(list(lock_root.glob("*.lock"))) == 1

    def test_journal_status_covers_all_state_classes(self) -> None:
        assert script._journal_status([{"state": "prepared"}]) == "prepared"
        assert (
            script._journal_status([{"state": "submitted"}, {"state": "prepared"}])
            == "partially_submitted"
        )
        assert script._journal_status([{"state": "submitting"}]) == "recovery_required"

    def test_payload_digest_rejects_missing_parameter(self) -> None:
        parameters = types.SimpleNamespace(shots=1)
        with pytest.raises(ValueError, match="missing execution parameter"):
            script._payload_digest([FakeCircuit("x")], parameters)

    def test_json_default_supports_uuid_and_containers_and_rejects_unknown(self) -> None:
        class Mode(Enum):
            STRICT = "strict"

        assert script._json_default(FakeCircuit("x")) == {"name": "x", "depth": 20}
        assert script._json_default(Mode.STRICT) == "strict"
        assert script._json_default(UUID(int=7)) == str(UUID(int=7))
        assert script._json_default((1, 2)) == [1, 2]
        with pytest.raises(TypeError, match="cannot serialise"):
            script._json_default(object())

    def test_job_identifier_supports_method_and_rejects_empty(self) -> None:
        valid = str(UUID(int=7))
        assert script._job_identifier(types.SimpleNamespace(job_id=lambda: valid)) == valid
        with pytest.raises(RuntimeError, match="without a durable identifier"):
            script._job_identifier(types.SimpleNamespace(job_id=None))
        with pytest.raises(RuntimeError, match="non-UUID"):
            script._job_identifier(types.SimpleNamespace(job_id="job-1"))

    def test_json_loader_rejects_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a JSON object"):
            script._load_json_object(path)

    @pytest.mark.parametrize(
        "jobs, message",
        [
            ([], "at least one"),
            (["not-an-object"], "must be an object"),
            ([{"state": "prepared", "job_id": None}], "has no group_id"),
            (
                [
                    journal_job(),
                    journal_job(),
                ],
                "duplicate submission journal group_id",
            ),
            ([journal_job(group_id="other")], "unknown group_id"),
            ([journal_job(state="unknown")], "invalid state"),
            ([journal_job(shots=0)], "invalid shots"),
            ([journal_job(labels=[])], "invalid labels"),
            ([journal_job(circuit_names=[])], "invalid circuit_names"),
            ([journal_job(payload_sha256="bad")], "invalid payload_sha256"),
            ([journal_job(state="submitted")], "has no job_id"),
            ([journal_job(state="submitted", job_id="not-a-uuid")], "non-UUID job_id"),
            (
                [
                    journal_job(state="submitted", job_id=str(UUID(int=1))),
                    journal_job(
                        group_id="readout",
                        state="submitted",
                        job_id=str(UUID(int=1)),
                    ),
                ],
                "bound more than once",
            ),
            (
                [journal_job(job_id=str(UUID(int=2)))],
                "already has a job_id",
            ),
        ],
    )
    def test_journal_job_validation_rejects_malformed_states(
        self,
        jobs: list[object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            script._journal_jobs({"jobs": jobs})

    def test_group_plan_rejects_missing_group_and_field_drift(self) -> None:
        main = journal_job()
        with pytest.raises(ValueError, match="job groups"):
            script._validate_group_plan({"jobs": [main]}, [main, {**main, "group_id": "readout"}])
        with pytest.raises(ValueError, match="freshly prepared"):
            script._validate_group_plan({"jobs": [main]}, [{**main, "shots": 2048}])

    def test_submission_group_builder_rejects_empty_and_invalid_request(self) -> None:
        client = FakeClient()
        backend = FakeBackend(client)
        with pytest.raises(ValueError, match="no IQM submission groups"):
            script._build_submission_groups(backend, {1024: [], 2048: []})

        class BadBackend:
            def __init__(self, fake_client: FakeClient) -> None:
                self.client = fake_client

            def create_run_request(self, run_input: list[Any], *, shots: int) -> object:
                return object()

            def retrieve_job(self, job_id: str) -> Any:
                raise AssertionError(f"unexpected retrieval of {job_id}")

        with pytest.raises(ValueError, match="has no circuit list"):
            script._build_submission_groups(
                BadBackend(client),
                {1024: [("a", FakeCircuit("a"))]},
            )

    def test_incomplete_journal_refuses_retrieval_before_provider_contact(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.failures[1] = TimeoutError("response lost")
        record_path = tmp_path / "submission.json"
        assert (
            script.main(submit_args(record_path, "--i-have-owner-go"))
            == script.RECOVERY_REQUIRED_EXIT
        )
        calls_before = client.calls
        assert (
            script.main(
                [
                    "retrieve",
                    "--record",
                    str(record_path),
                    "--out",
                    str(tmp_path / "counts.json"),
                ]
            )
            == script.RECOVERY_REQUIRED_EXIT
        )
        assert "journal is incomplete" in capsys.readouterr().err
        assert client.calls == calls_before

    def test_partial_journal_payload_drift_is_refused_before_submission(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, _ = hardware_boundary
        client.failures[2] = KeyboardInterrupt()
        record_path = tmp_path / "submission.json"
        args = submit_args(record_path, "--i-have-owner-go")
        with pytest.raises(KeyboardInterrupt):
            script.main(args)
        record = load_record(record_path)
        record["jobs"][1]["state"] = "prepared"
        record["jobs"][1]["payload_sha256"] = "1" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "prepared payload does not match journal" in capsys.readouterr().err
        assert client.calls == 2


class TestRecoveryValidation:
    def _ambiguous_record(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
    ) -> tuple[FakeClient, Path]:
        client, _ = hardware_boundary
        client.failures[1] = TimeoutError("response lost")
        record_path = tmp_path / "submission.json"
        assert (
            script.main(submit_args(record_path, "--i-have-owner-go"))
            == script.RECOVERY_REQUIRED_EXIT
        )
        return client, record_path

    def test_recovery_rejects_invalid_record_and_unknown_group(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        args = [
            "recover",
            "--record",
            str(bad),
            "--group",
            "main",
            "--job-id",
            str(UUID(int=1)),
            "--i-confirm-provider-job",
        ]
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "invalid recovery journal" in capsys.readouterr().err

        client, record = self._ambiguous_record(hardware_boundary, tmp_path)
        recovered_id = UUID(int=301)
        client.jobs[recovered_id] = FakeProviderJob(recovered_id, client.requests[1])
        args[args.index(str(bad))] = str(record)
        args[args.index("main")] = "readout"
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "not an ambiguous call" in capsys.readouterr().err

    def test_recovery_rejects_group_absent_from_valid_journal(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _client, _ = hardware_boundary
        record_path = tmp_path / "record.json"
        record_path.write_text(
            json.dumps(
                {
                    "schema": script.JOURNAL_SCHEMA,
                    "quantum_computer": "garnet",
                    "jobs": [journal_job(state="recovery_required")],
                }
            ),
            encoding="utf-8",
        )
        assert (
            script.main(
                [
                    "recover",
                    "--record",
                    str(record_path),
                    "--group",
                    "readout",
                    "--job-id",
                    str(UUID(int=1)),
                    "--i-confirm-provider-job",
                ]
            )
            == script.RECOVERY_REQUIRED_EXIT
        )
        assert "no unique group" in capsys.readouterr().err

    def test_recovery_rejects_invalid_uuid_and_provider_lookup_failure(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _client, record = self._ambiguous_record(hardware_boundary, tmp_path)
        base = [
            "recover",
            "--record",
            str(record),
            "--group",
            "main",
            "--job-id",
            "not-a-uuid",
            "--i-confirm-provider-job",
        ]
        assert script.main(base) == script.RECOVERY_REQUIRED_EXIT
        assert "not an IQM UUID" in capsys.readouterr().err
        base[base.index("not-a-uuid")] = str(UUID(int=302))
        assert script.main(base) == script.RECOVERY_REQUIRED_EXIT
        assert "could not retrieve and verify" in capsys.readouterr().err

    def test_recovery_is_idempotent_and_rejects_rebinding(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client, record = self._ambiguous_record(hardware_boundary, tmp_path)
        recovered_id = UUID(int=303)
        client.jobs[recovered_id] = FakeProviderJob(recovered_id, client.requests[1])
        args = [
            "recover",
            "--record",
            str(record),
            "--group",
            "main",
            "--job-id",
            str(recovered_id),
            "--i-confirm-provider-job",
        ]
        assert script.main(args) == 0
        assert script.main(args) == 0
        assert "already bound" in capsys.readouterr().out
        args[args.index(str(recovered_id))] = str(UUID(int=304))
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "already bound" in capsys.readouterr().err

    def test_recovery_rejects_job_already_bound_to_other_group(
        self,
        hardware_boundary: tuple[FakeClient, FakeBackend],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _client, record_path = self._ambiguous_record(hardware_boundary, tmp_path)
        record = load_record(record_path)
        duplicate_id = str(UUID(int=305))
        record["jobs"][1]["state"] = "submitted"
        record["jobs"][1]["job_id"] = duplicate_id
        record_path.write_text(json.dumps(record), encoding="utf-8")
        assert (
            script.main(
                [
                    "recover",
                    "--record",
                    str(record_path),
                    "--group",
                    "main",
                    "--job-id",
                    duplicate_id,
                    "--i-confirm-provider-job",
                ]
            )
            == script.RECOVERY_REQUIRED_EXIT
        )
        assert "already bound to another group" in capsys.readouterr().err


class TestRetrieval:
    def test_complete_journal_polls_and_writes_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record_path = tmp_path / "record.json"
        main_id = str(UUID(int=401))
        readout_id = str(UUID(int=402))
        record = {
            "schema": script.JOURNAL_SCHEMA,
            "campaign": "campaign",
            "quantum_computer": "garnet",
            "date": "2026-07-26",
            "repetition": 1,
            "window": 3,
            "layout": list(script.PRIMARY_LAYOUT),
            "jobs": [
                journal_job(
                    state="submitted",
                    job_id=main_id,
                    labels=["a", "b"],
                    circuit_names=["a-circuit", "b-circuit"],
                ),
                journal_job(
                    group_id="readout",
                    state="submitted",
                    job_id=readout_id,
                    labels=["r"],
                    circuit_names=["r-circuit"],
                ),
            ],
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        backend = FakeRetrievalBackend(
            {
                main_id: FakePollingJob([{"0000": 2}, {"0001": 3}], waits=1),
                readout_id: FakePollingJob({"0011": 4}),
            }
        )
        monkeypatch.setattr(script, "_live_backend", lambda _name: backend)
        output = tmp_path / "counts.json"
        assert (
            script.main(
                [
                    "retrieve",
                    "--record",
                    str(record_path),
                    "--out",
                    str(output),
                    "--poll-seconds",
                    "0",
                ]
            )
            == 0
        )
        payload = load_record(output)
        assert payload["job_ids"] == [main_id, readout_id]
        assert payload["counts"] == {"a": {"0000": 2}, "b": {"0001": 3}, "r": {"0011": 4}}
        assert "QUEUED" in capsys.readouterr().out

    def test_legacy_missing_jobs_and_timeout_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record_path = tmp_path / "legacy.json"
        record_path.write_text(json.dumps({"quantum_computer": "garnet", "jobs": []}))
        output = tmp_path / "counts.json"
        args = ["retrieve", "--record", str(record_path), "--out", str(output)]
        assert script.main(args) == script.RECOVERY_REQUIRED_EXIT
        assert "legacy submission record has no jobs" in capsys.readouterr().err

        legacy = {
            "campaign": "campaign",
            "quantum_computer": "garnet",
            "date": "2026-07-26",
            "repetition": 1,
            "layout": list(script.PRIMARY_LAYOUT),
            "jobs": [{"job_id": "job", "labels": ["a"]}],
        }
        record_path.write_text(json.dumps(legacy), encoding="utf-8")
        backend = FakeRetrievalBackend({"job": FakePollingJob({"0000": 1}, waits=99)})
        monkeypatch.setattr(script, "_live_backend", lambda _name: backend)
        assert script.main([*args, "--timeout-minutes", "-1"]) == 3
        assert "not finished within timeout" in capsys.readouterr().err
