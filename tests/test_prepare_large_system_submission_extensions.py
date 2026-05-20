# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- large-system submission-extension tests
"""Contract tests for larger-system submission-extension readiness planning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from qiskit.providers.fake_provider import GenericBackendV2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module {name}")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


large_system = _load_script_module("prepare_large_system_submission_extensions")


def test_candidate_matrix_covers_submission_papers_and_scales() -> None:
    specs = large_system.candidate_specs()

    assert [spec.candidate_id for spec in specs] == [
        "phase3_reduced_pauli_n6",
        "fim_replication_zne_n6",
        "methods_ansatz_energy_n6",
        "phase3_reduced_pauli_n8",
        "fim_replication_zne_n8",
        "methods_ansatz_energy_n8",
        "methods_gpu_scaling_n16",
        "methods_gpu_scaling_n20",
    ]
    assert {spec.paper_target for spec in specs} == {
        "submission_003_rust_vqe_methods",
        "submission_004_scpn_fim_hamiltonian",
        "submission_005_phase3_reduced_pauli_entanglement",
    }
    assert {spec.n_qubits for spec in specs} == {6, 8, 16, 20}


def test_qpu_readiness_accepts_n8_readout_ceiling_and_rejects_n9_full_readout() -> None:
    n8_spec = next(
        spec
        for spec in large_system.candidate_specs()
        if spec.candidate_id == "phase3_reduced_pauli_n8"
    )
    backend = large_system._line_backend(32)

    estimate = large_system.estimate_candidate(
        n8_spec,
        backend=backend,
        qpu_seconds_ceiling=large_system.QPU_SECONDS_CEILING,
        gpu_seconds_ceiling=large_system.GPU_SECONDS_CEILING,
    )

    assert estimate.readout_circuits == 256
    assert "full readout requires 256 calibration states" not in estimate.decision_reasons
    assert estimate.status == "ready_for_qpu_preregistration"

    n9_spec = large_system.CandidateSpec(
        **{**n8_spec.__dict__, "candidate_id": "synthetic_full_readout_n9", "n_qubits": 9}
    )
    rejected = large_system.estimate_candidate(
        n9_spec,
        backend=backend,
        qpu_seconds_ceiling=large_system.QPU_SECONDS_CEILING,
        gpu_seconds_ceiling=large_system.GPU_SECONDS_CEILING,
    )

    assert rejected.status == "blocked_or_needs_reduction"
    assert "full readout requires 512 calibration states" in rejected.decision_reasons


def test_methods_n16_n20_are_classical_gpu_lanes_not_qpu_jobs() -> None:
    backend = GenericBackendV2(num_qubits=32, seed=large_system.SEED_TRANSPILER)

    estimates = {
        spec.candidate_id: large_system.estimate_candidate(
            spec,
            backend=backend,
            qpu_seconds_ceiling=large_system.QPU_SECONDS_CEILING,
            gpu_seconds_ceiling=large_system.GPU_SECONDS_CEILING,
        )
        for spec in large_system.candidate_specs()
        if spec.candidate_id.startswith("methods_gpu_scaling_")
    }

    assert set(estimates) == {"methods_gpu_scaling_n16", "methods_gpu_scaling_n20"}
    assert all(row.status == "ready_for_gpu_execution" for row in estimates.values())
    assert all(row.estimated_qpu_seconds == 0.0 for row in estimates.values())
    assert all(row.shots == 0 for row in estimates.values())
    assert estimates["methods_gpu_scaling_n20"].statevector_bytes == 16 * 2**20


def test_methods_qpu_lanes_use_submitter_conservative_seconds_per_circuit() -> None:
    spec = next(
        spec
        for spec in large_system.candidate_specs()
        if spec.candidate_id == "methods_ansatz_energy_n8"
    )
    estimate = large_system.estimate_candidate(
        spec,
        backend=large_system._line_backend(32),
        qpu_seconds_ceiling=large_system.QPU_SECONDS_CEILING,
        gpu_seconds_ceiling=large_system.GPU_SECONDS_CEILING,
    )

    assert estimate.total_circuits == 265
    assert estimate.estimated_qpu_seconds == 265.0


def test_payload_and_markdown_record_ready_order_and_budget(tmp_path: Path) -> None:
    args = large_system._parse_args(
        [
            "--backend",
            "generic_line",
            "--backend-qubits",
            "32",
            "--out-dir",
            str(tmp_path / "data"),
            "--docs-dir",
            str(tmp_path / "docs"),
        ]
    )
    payload = large_system.build_payload(args)

    assert payload["hardware_submission"] is False
    assert payload["ready_qpu_minutes_total"] > 0.0
    assert payload["two_backend_ready_qpu_seconds_total"] == (
        2.0 * payload["ready_qpu_seconds_total"]
    )
    assert payload["one_backend_fits_live_budget"] is None
    assert payload["two_backend_fits_live_budget"] is None
    assert payload["recommended_order"] == [
        "phase3_reduced_pauli_n6",
        "phase3_reduced_pauli_n8",
        "fim_replication_zne_n6",
        "fim_replication_zne_n8",
        "methods_ansatz_energy_n6",
        "methods_ansatz_energy_n8",
        "methods_gpu_scaling_n16",
        "methods_gpu_scaling_n20",
    ]

    digest = large_system.write_markdown(tmp_path / "docs" / "readiness.md", payload)
    written = (tmp_path / "docs" / "readiness.md").read_text(encoding="utf-8")

    assert len(digest) == 64
    assert "Larger-System Submission Extension Readiness" in written
    assert "Ready QPU estimate, Fez+Marrakesh pair" in written
    assert "`phase3_reduced_pauli_n6`" in written
    assert "`methods_gpu_scaling_n20`" in written


def test_usage_budget_summary_computes_live_fit_fields(monkeypatch: Any) -> None:
    args = large_system._parse_args(["--backend", "generic_line", "--backend-qubits", "32"])
    usage = {
        "usage_allocation_seconds": 3600,
        "usage_consumed_seconds": 0,
        "usage_remaining_seconds": 3600,
    }
    monkeypatch.setattr(
        large_system,
        "load_backend_and_usage",
        lambda *args, **kwargs: (large_system._line_backend(32), usage),
    )

    payload = large_system.build_payload(args)

    assert payload["usage_budget"]["remaining_seconds"] == 3600.0
    assert payload["one_backend_fits_live_budget"] is True
    assert payload["two_backend_fits_live_budget"] is True
    assert payload["remaining_after_two_backends_seconds"] == (
        3600.0 - payload["two_backend_ready_qpu_seconds_total"]
    )
