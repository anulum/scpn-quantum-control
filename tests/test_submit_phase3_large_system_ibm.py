# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — submit phase3 large system IBM tests
# scpn-quantum-control -- Phase 3 larger-system submitter tests
"""Contract tests for the Phase 3 larger-system IBM submitter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

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


phase3_large = _load_script_module("submit_phase3_large_system_ibm")


def test_source_specs_generalise_phase3_scope_to_larger_n() -> None:
    specs = phase3_large.source_specs(6)

    assert [spec.label for spec in specs] == [
        "dla_even_shallow",
        "dla_odd_shallow",
        "dla_even_signal",
        "dla_odd_signal",
        "fim_lambda0_reference",
        "fim_lambda4_feedback",
    ]
    assert {len(spec.initial_bitstring) for spec in specs} == {6}
    assert specs[0].initial_bitstring == "000011"
    assert specs[1].initial_bitstring == "000001"
    assert specs[-1].lambda_fim == 4.0


def test_observable_settings_cover_left_middle_and_right_transverse_edges() -> None:
    assert phase3_large.phase3_observables(8) == (
        "XXIIIIII",
        "YYIIIIII",
        "IIIXXIII",
        "IIIYYIII",
        "IIIIIIXX",
        "IIIIIIYY",
    )


def test_build_entries_has_expected_zne_and_full_readout_counts() -> None:
    entries = phase3_large.build_entries(
        n_qubits=6,
        repetitions=3,
        noise_scales=[1, 3, 5],
    )

    main = [entry for entry in entries if entry.block == "main"]
    readout = [entry for entry in entries if entry.block == "readout_calibration"]

    assert len(main) == 324
    assert len(readout) == 64
    assert all(entry.circuit.num_qubits == 6 for entry in entries)
    assert {entry.noise_scale for entry in main} == {1, 3, 5}
    assert {entry.basis_setting for entry in main} == set(phase3_large.phase3_observables(6))


def test_readiness_payload_is_budget_and_width_aware() -> None:
    args = phase3_large._parse_args(
        [
            "--backend",
            "fake_backend",
            "--n-qubits",
            "6",
            "--physical-qubits",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--max-qpu-seconds",
            "1000",
        ]
    )
    entries = phase3_large.build_entries(
        n_qubits=args.n_qubits,
        repetitions=args.repetitions,
        noise_scales=args.noise_scales,
    )
    circuits = [entry.circuit for entry in entries]

    payload = phase3_large.readiness_payload(
        args=args,
        backend=object(),
        entries=entries,
        isa_circuits=circuits,
        physical_qubits=args.physical_qubits,
    )

    assert payload["status"] == "ready_for_submission"
    assert payload["n_qubits"] == 6
    assert payload["main_circuits"] == 324
    assert payload["readout_calibration_circuits"] == 64
    assert payload["estimated_qpu_seconds"] == 213.4
    assert "n=6" in payload["claim_boundary"]


def test_default_layouts_include_kingston_larger_width_replicates() -> None:
    assert phase3_large.DEFAULT_LAYOUTS[("ibm_kingston", 6)] == (
        141,
        142,
        143,
        144,
        145,
        146,
    )
    assert phase3_large.DEFAULT_LAYOUTS[("ibm_kingston", 8)] == (
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
    )
