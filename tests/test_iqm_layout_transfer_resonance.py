# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer Resonance runner tests
"""Tests for the owner-gated IQM Garnet layout-transfer submission boundary."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_runner() -> ModuleType:
    return importlib.reload(importlib.import_module("scripts.iqm_layout_transfer_resonance"))


def _circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.cz(0, 1)
    circuit.measure((0, 1), (0, 1))
    return circuit


def _matrix() -> tuple[list[str], list[QuantumCircuit]]:
    labels: list[str] = []
    for n in (8, 12, 16):
        for arm in ("optimised", "default", "naive"):
            labels.extend(f"main_n{n}_{arm}_rep{repetition}" for repetition in (1, 2, 3, 4))
        labels.extend((f"readout_n{n}_zeros", f"readout_n{n}_ones"))
    return labels, [_circuit() for _label in labels]


class _QpyWrapper:
    def __init__(self, circuits: list[QuantumCircuit]) -> None:
        self._circuits = circuits

    def reviewed_qpy_load_circuits(self, _path: Path) -> list[QuantumCircuit]:
        return self._circuits


class _Job:
    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _Job:
        self.calls.append((len(circuits), shots))
        return _Job(f"job-{len(self.calls)}")


class _RetrievedJob:
    def __init__(self, counts: Any, *, done_after: int = 0) -> None:
        self._counts = counts
        self._done_after = done_after
        self.done_calls = 0

    def done(self) -> bool:
        self.done_calls += 1
        return self.done_calls > self._done_after

    def status(self) -> str:
        return "QUEUED"

    def result(self) -> SimpleNamespace:
        return SimpleNamespace(get_counts=lambda: self._counts)


class _RetrievalBackend:
    def __init__(self, jobs: dict[str, _RetrievedJob]) -> None:
        self.jobs = jobs

    def retrieve_job(self, job_id: str) -> _RetrievedJob:
        return self.jobs[job_id]


def _write_inputs(tmp_path: Path, labels: list[str], campaign: str) -> tuple[Path, Path, Path]:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    circuits_path = tmp_path / "circuits.qpy"
    circuits_path.write_bytes(b"reviewed-qpy-fixture")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "campaign": campaign,
                "circuit_count": 42,
                "all_gates_pass": True,
                "main_shots": 2048,
                "readout_shots": 1024,
                "calibration_set_id": "calibration-test",
            }
        ),
        encoding="utf-8",
    )
    return labels_path, circuits_path, plan_path


def _args(
    tmp_path: Path,
    labels_path: Path,
    circuits_path: Path,
    plan_path: Path,
    *,
    owner_go: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        i_have_owner_go=owner_go,
        labels=str(labels_path),
        circuits=str(circuits_path),
        plan=str(plan_path),
        only_n=None,
        all_sizes=True,
        quantum_computer="garnet:mock",
        date="2026-07-26",
        out=str(tmp_path / "submission.json"),
    )


def test_all_sizes_submits_exact_frozen_two_job_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    labels_path, circuits_path, plan_path = _write_inputs(
        tmp_path, labels, runner.IQM_LAYOUT_TRANSFER_CAMPAIGN
    )
    backend = _Backend()
    monkeypatch.setattr(runner, "_load_qpy_wrapper", lambda: _QpyWrapper(circuits))
    monkeypatch.setattr(runner, "_backend", lambda _target: backend)
    monkeypatch.setattr("qiskit.transpile", lambda circuit, **_kwargs: circuit)

    assert runner._submit(_args(tmp_path, labels_path, circuits_path, plan_path)) == 0
    assert backend.calls == [(36, 2048), (6, 1024)]
    record = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert record["block"] == "all_sizes"
    assert record["calibration_set_id"] == "calibration-test"
    assert len(record["live_two_qubit_depths"]) == 36
    assert [len(job["labels"]) for job in record["jobs"]] == [36, 6]
    assert all(record[key] for key in ("plan_sha256", "labels_sha256", "circuits_sha256"))


def test_all_sizes_rejects_plan_outside_per_size_campaign(tmp_path: Path) -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    with pytest.raises(ValueError, match="restricted to the frozen per-size"):
        runner._select_submission_matrix(
            labels,
            circuits,
            {"campaign": "legacy", "all_gates_pass": True, "circuit_count": 42},
            only_n=None,
            all_sizes=True,
        )


def test_submit_refuses_without_current_owner_go(tmp_path: Path) -> None:
    runner = _load_runner()
    labels, _circuits = _matrix()
    labels_path, circuits_path, plan_path = _write_inputs(
        tmp_path, labels, runner.IQM_LAYOUT_TRANSFER_CAMPAIGN
    )
    assert (
        runner._submit(
            _args(
                tmp_path,
                labels_path,
                circuits_path,
                plan_path,
                owner_go=False,
            )
        )
        == 2
    )


def test_live_depth_gate_rejects_unbalanced_size() -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    for label, circuit in zip(labels, circuits, strict=True):
        if label == "main_n12_naive_rep4":
            for _index in range(3):
                circuit.cz(0, 1)
    with pytest.raises(ValueError, match="depth-parity violation at n=12"):
        runner._validate_iqm_layout_transfer_live_depths(list(zip(labels, circuits, strict=True)))


def test_reviewed_standalone_loaders_resolve_real_modules() -> None:
    runner = _load_runner()
    assert runner._load_adapter()._qubit_index("QB12") == 11
    assert callable(runner._load_qpy_wrapper().reviewed_qpy_load_circuits)


@pytest.mark.parametrize("loader_name", ["_load_adapter", "_load_qpy_wrapper"])
def test_standalone_loaders_fail_closed_when_module_cannot_resolve(
    loader_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(ImportError, match="cannot load"):
        getattr(runner, loader_name)()


def test_credentials_are_scoped_to_iqm_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    vault = tmp_path / "credentials.md"
    test_token = "".join(("te", "st"))
    vault.write_text(
        "## Unrelated\n- token:\n"
        "## IQM Resonance\n- URL: https://example.invalid\n- Token: " + test_token + "\n"
        "## Following\n- token:\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "VAULT_PATH", vault)
    assert runner._load_credentials() == ("https://example.invalid", test_token)
    vault.write_text("## IQM Resonance\n- URL: https://example.invalid\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing IQM Resonance"):
        runner._load_credentials()


@pytest.mark.parametrize(
    ("labels_transform", "circuits_transform", "plan_update", "only_n", "all_sizes", "message"),
    [
        (lambda labels: labels[:-1], lambda circuits: circuits, {}, None, True, "labels but"),
        (lambda labels: labels, lambda circuits: circuits, {}, 8, True, "mutually exclusive"),
        (
            lambda labels: labels,
            lambda circuits: circuits,
            {"all_gates_pass": False},
            None,
            True,
            "42-circuit plan",
        ),
        (
            lambda labels: [*labels[:-1], "other"],
            lambda circuits: circuits,
            {},
            None,
            True,
            "partition",
        ),
        (
            lambda labels: [*labels[:-1], labels[-2]],
            lambda circuits: circuits,
            {},
            None,
            True,
            "unique",
        ),
        (lambda labels: labels, lambda circuits: circuits, {}, None, False, "choose exactly one"),
        (lambda labels: labels, lambda circuits: circuits, {}, 99, False, "no circuits match"),
    ],
)
def test_submission_matrix_fails_closed(
    labels_transform: Any,
    circuits_transform: Any,
    plan_update: dict[str, Any],
    only_n: int | None,
    all_sizes: bool,
    message: str,
) -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    plan: dict[str, Any] = {
        "campaign": runner.IQM_LAYOUT_TRANSFER_CAMPAIGN,
        "all_gates_pass": True,
        "circuit_count": 42,
    }
    plan.update(plan_update)
    with pytest.raises(ValueError, match=message):
        runner._select_submission_matrix(
            labels_transform(labels),
            circuits_transform(circuits),
            plan,
            only_n=only_n,
            all_sizes=all_sizes,
        )


def test_legacy_one_size_selection_remains_available() -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    block, selected = runner._select_submission_matrix(
        labels,
        circuits,
        {"campaign": "legacy"},
        only_n=12,
        all_sizes=False,
    )
    assert block == "n12"
    assert len(selected) == 14


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda labels, circuits: labels.__setitem__(0, "main_n8_unknown_rep1"),
            "unexpected per-size layout-transfer main label",
        ),
        (
            lambda labels, circuits: circuits.__setitem__(0, QuantumCircuit(2, 2)),
            "has no two-qubit depth",
        ),
        (
            lambda labels, circuits: (labels.pop(0), circuits.pop(0)),
            "12 main circuits per size",
        ),
    ],
)
def test_live_depth_matrix_rejects_malformed_content(mutator: Any, message: str) -> None:
    runner = _load_runner()
    labels, circuits = _matrix()
    mutator(labels, circuits)
    with pytest.raises(ValueError, match=message):
        runner._validate_iqm_layout_transfer_live_depths(list(zip(labels, circuits, strict=True)))


def test_provider_constructors_receive_vault_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("client", args, kwargs))

    class Provider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("provider", args, kwargs))

        def get_backend(self) -> str:
            return "backend"

    def fake_import(name: str) -> SimpleNamespace:
        if name == "iqm.iqm_client":
            return SimpleNamespace(IQMClient=Client)
        return SimpleNamespace(IQMProvider=Provider)

    monkeypatch.setattr(runner, "_load_credentials", lambda: ("url", "token"))
    monkeypatch.setattr(runner.importlib, "import_module", fake_import)
    assert isinstance(runner._client("garnet"), Client)
    assert runner._backend("garnet") == "backend"
    assert calls == [
        ("client", ("url",), {"token": "token", "quantum_computer": "garnet"}),
        (
            "provider",
            ("url",),
            {"quantum_computer": "garnet", "token": "token"},
        ),
    ]


def _calibration_fixture(*, missing: str | None = None) -> tuple[Any, Any]:
    cz_impl = SimpleNamespace(loci=[("QB1", "QB2")])
    measure_impl = SimpleNamespace(loci=[("QB1",), ("QB2",)])
    architecture = SimpleNamespace(
        calibration_set_id="calibration-id",
        qubits=["QB1", "QB2"],
        gates={
            "cz": SimpleNamespace(
                default_implementation="impl", implementations={"impl": cz_impl}
            ),
            "measure": SimpleNamespace(
                default_implementation="impl", implementations={"impl": measure_impl}
            ),
        },
    )
    metrics = SimpleNamespace(
        get_gate_fidelity=lambda *_args: None if missing == "cz" else 0.99,
        get_measure_errors=lambda *_args: None if missing == "measure" else (0.02, 0.04),
    )
    return architecture, metrics


def test_dump_calibration_writes_adapter_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    architecture, metrics = _calibration_fixture()
    client = SimpleNamespace(
        get_dynamic_quantum_architecture=lambda: architecture,
        get_calibration_quality_metrics=lambda _calibration_id: metrics,
    )
    monkeypatch.setattr(runner, "_client", lambda _target: client)
    out = tmp_path / "calibration.json"
    args = argparse.Namespace(quantum_computer="garnet", date="2026-07-26", out=str(out))
    assert runner._dump_calibration(args) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["calibration_set_id"] == "calibration-id"
    assert payload["calibration"] == {
        "num_qubits": 2,
        "edges": [[0, 1]],
        "edge_fidelity": {"0-1": 0.99},
        "readout_error": {"0": 0.03, "1": 0.03},
    }


@pytest.mark.parametrize(("missing", "message"), [("cz", "no CZ"), ("measure", "no measure")])
def test_dump_calibration_rejects_incomplete_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    message: str,
) -> None:
    runner = _load_runner()
    architecture, metrics = _calibration_fixture(missing=missing)
    client = SimpleNamespace(
        get_dynamic_quantum_architecture=lambda: architecture,
        get_calibration_quality_metrics=lambda _calibration_id: metrics,
    )
    monkeypatch.setattr(runner, "_client", lambda _target: client)
    with pytest.raises(RuntimeError, match=message):
        runner._dump_calibration(
            argparse.Namespace(
                quantum_computer="garnet", date="2026-07-26", out=str(tmp_path / "out.json")
            )
        )


def test_legacy_submit_skips_empty_readout_group_and_accepts_property_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    labels = ["main_n8_optimised_rep1"]
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    circuits_path = tmp_path / "circuits.qpy"
    circuits_path.write_bytes(b"fixture")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"campaign": "legacy", "main_shots": 10, "readout_shots": 20}),
        encoding="utf-8",
    )

    class PropertyBackend:
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> SimpleNamespace:
            assert len(circuits) == 1 and shots == 10
            return SimpleNamespace(job_id="property-job")

    monkeypatch.setattr(runner, "_load_qpy_wrapper", lambda: _QpyWrapper([_circuit()]))
    monkeypatch.setattr(runner, "_backend", lambda _target: PropertyBackend())
    monkeypatch.setattr("qiskit.transpile", lambda circuit, **_kwargs: circuit)
    args = argparse.Namespace(
        i_have_owner_go=True,
        labels=str(labels_path),
        circuits=str(circuits_path),
        plan=str(plan_path),
        only_n=8,
        all_sizes=False,
        quantum_computer="garnet:mock",
        date="2026-07-26",
        out=str(tmp_path / "submission.json"),
    )
    assert runner._submit(args) == 0
    record = json.loads(Path(args.out).read_text(encoding="utf-8"))
    assert record["jobs"] == [{"job_id": "property-job", "shots": 10, "labels": labels}]


def test_retrieve_polls_and_normalises_single_and_list_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    record = {
        "quantum_computer": "garnet:mock",
        "date": "2026-07-26",
        "block": "fixture",
        "jobs": [
            {"job_id": "one", "labels": ["a"]},
            {"job_id": "two", "labels": ["b", "c"]},
        ],
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    backend = _RetrievalBackend(
        {
            "one": _RetrievedJob({"0": 2}, done_after=1),
            "two": _RetrievedJob([{"1": 3}, {"00": 4}]),
        }
    )
    monkeypatch.setattr(runner, "_backend", lambda _target: backend)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    out = tmp_path / "counts.json"
    args = argparse.Namespace(
        record=str(record_path), out=str(out), timeout_minutes=1.0, poll_seconds=0.0
    )
    assert runner._retrieve(args) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["job_ids"] == ["one", "two"]
    assert payload["counts"] == {"a": {"0": 2}, "b": {"1": 3}, "c": {"00": 4}}


def test_retrieve_timeout_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(
            {
                "quantum_computer": "garnet:mock",
                "date": "2026-07-26",
                "block": "fixture",
                "jobs": [{"job_id": "slow", "labels": ["a"]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_backend",
        lambda _target: _RetrievalBackend({"slow": _RetrievedJob({}, done_after=100)}),
    )
    monotonic_values = iter((10.0, 11.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))
    args = argparse.Namespace(
        record=str(record_path),
        out=str(tmp_path / "never.json"),
        timeout_minutes=0.0,
        poll_seconds=0.0,
    )
    assert runner._retrieve(args) == 3
    assert not Path(args.out).exists()


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (
            ["dump-calibration", "--date", "2026-07-26", "--out", "calibration.json"],
            "_dump_calibration",
        ),
        (
            [
                "submit",
                "--circuits",
                "circuits.qpy",
                "--labels",
                "labels.json",
                "--plan",
                "plan.json",
                "--all-sizes",
                "--date",
                "2026-07-26",
                "--out",
                "submission.json",
            ],
            "_submit",
        ),
        (
            ["retrieve", "--record", "submission.json", "--out", "counts.json"],
            "_retrieve",
        ),
    ],
)
def test_main_dispatches_each_cli_boundary(
    argv: list[str], target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    calls: list[argparse.Namespace] = []

    def dispatched(args: argparse.Namespace) -> int:
        calls.append(args)
        return 7

    monkeypatch.setattr(runner, target, dispatched)
    assert runner.main(argv) == 7
    assert len(calls) == 1


def test_main_rejects_non_integer_subcommand_result(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "_retrieve", lambda _args: "invalid")
    with pytest.raises(TypeError, match="integer process exit code"):
        runner.main(["retrieve", "--record", "submission.json", "--out", "counts.json"])
