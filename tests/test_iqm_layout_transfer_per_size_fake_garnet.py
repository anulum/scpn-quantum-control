# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM Garnet per-size fake-backend boundary tests
"""Tests for descriptive fail-closed per-size fake-backend diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

CAMPAIGN = "iqm_layout_transfer_per_size_prereg_2026-07-22"


def _runner() -> ModuleType:
    return importlib.reload(
        importlib.import_module("scripts.iqm_layout_transfer_per_size_fake_garnet")
    )


def _install_iqm_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module_names = (
        "iqm",
        "iqm.qiskit_iqm",
        "iqm.qiskit_iqm.fake_backends",
        "iqm.qiskit_iqm.fake_backends.fake_garnet",
    )
    modules = {name: ModuleType(name) for name in module_names}
    modules[module_names[-1]].__dict__["IQMFakeGarnet"] = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_optional_loaders_resolve_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    _install_iqm_stub(monkeypatch)
    assert runner._load_fake_backend_type() is object
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(ImportError, match="cannot load reviewed QPY"):
        runner._load_qpy_wrapper()


def _args(
    tmp_path: Path,
    *,
    labels: list[str],
    campaign: str = CAMPAIGN,
    circuit_count: int = 42,
    all_gates_pass: bool = True,
) -> tuple[argparse.Namespace, list[object]]:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    circuits_path = tmp_path / "circuits.qpy"
    circuits_path.write_bytes(b"reviewed-qpy-fixture")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "campaign": campaign,
                "circuit_count": circuit_count,
                "all_gates_pass": all_gates_pass,
                "main_shots": 2048,
                "readout_shots": 1024,
            }
        ),
        encoding="utf-8",
    )
    return (
        argparse.Namespace(
            circuits=str(circuits_path),
            labels=str(labels_path),
            plan=str(plan_path),
            out=str(tmp_path / "counts.json"),
            date="2026-07-26",
            resume=True,
        ),
        [object() for _label in labels],
    )


@pytest.mark.parametrize(
    ("campaign", "circuit_count", "all_gates_pass", "labels", "message"),
    [
        ("legacy", 42, True, ["label"], "outside the frozen per-size campaign"),
        (CAMPAIGN, 41, True, ["label"], "per-size plan must contain 42 circuits"),
        (CAMPAIGN, 42, False, ["label"], "per-size plan must contain 42 circuits"),
        (CAMPAIGN, 42, True, [f"other_{index}" for index in range(42)], "partition"),
    ],
)
def test_run_uses_descriptive_fail_closed_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign: str,
    circuit_count: int,
    all_gates_pass: bool,
    labels: list[str],
    message: str,
) -> None:
    runner = _runner()
    args, circuits = _args(
        tmp_path,
        labels=labels,
        campaign=campaign,
        circuit_count=circuit_count,
        all_gates_pass=all_gates_pass,
    )
    monkeypatch.setattr(
        runner,
        "_load_qpy_wrapper",
        lambda: SimpleNamespace(reviewed_qpy_load_circuits=lambda _path: circuits),
    )
    monkeypatch.setattr(runner, "_load_fake_backend_type", lambda: object)

    with pytest.raises(ValueError, match=message):
        runner._run(args)


def test_main_exposes_per_size_fake_backend_responsibility() -> None:
    runner = _runner()
    assert runner.main.__doc__ == ("Run the provider-free, resumable per-size fake-backend gate.")
