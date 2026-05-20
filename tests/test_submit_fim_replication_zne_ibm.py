# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM replication/ZNE lane tests
"""Contract tests for the FIM replication/ZNE IBM lane."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from qiskit import QuantumCircuit

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


fim_zne = _load_script_module("submit_fim_replication_zne_ibm")


def test_local_folding_preserves_measurement_count_and_scales_unitaries() -> None:
    qc = QuantumCircuit(2, 2)
    qc.rx(0.2, 0)
    qc.rzz(0.3, 0, 1)
    qc.measure(range(2), range(2))

    folded = fim_zne.locally_fold_circuit(qc, 3)
    ops = folded.count_ops()

    assert ops["measure"] == 2
    assert ops["rx"] == 3
    assert ops["rzz"] == 3


def test_build_entries_includes_main_scales_and_full_readout_calibration() -> None:
    entries = fim_zne.build_entries(
        states=["0000"],
        depths=[2],
        replicates=2,
        noise_scales=[1, 3],
    )

    main = [entry for entry in entries if entry.block == "main"]
    readout = [entry for entry in entries if entry.block == "readout_calibration"]

    assert len(main) == 8
    assert len(readout) == 16
    assert {entry.noise_scale for entry in main} == {1, 3}
    assert {entry.lambda_fim for entry in main} == {0.0, 4.0}


def test_summarise_synthetic_rows_reports_zne_delta() -> None:
    rows = []
    metadata_index = 0
    for state in fim_zne.computational_basis_labels(4):
        rows.append(
            {
                "metadata": {
                    "block": "readout_calibration",
                    "observed_target_bitstring": state,
                },
                "counts": {state: 32},
            }
        )
    for scale in [1, 3, 5]:
        for lambda_fim, counts in [
            (0.0, {"0000": 24, "0001": 8}),
            (4.0, {"0000": 16, "0001": 16}),
        ]:
            for replicate in [0, 1]:
                rows.append(
                    {
                        "metadata": {
                            "circuit_index": metadata_index,
                            "block": "main",
                            "initial_bitstring": "0000",
                            "observed_target_bitstring": "0000",
                            "lambda_fim": lambda_fim,
                            "depth": 2,
                            "zne_noise_scale": scale,
                            "replicate": replicate,
                            "magnetisation": 4,
                            "popcount": 0,
                            "transpiled_depth": 10,
                            "transpiled_two_qubit_gates": 4,
                        },
                        "counts": counts,
                    }
                )
                metadata_index += 1

    summary = fim_zne._summarise(
        {
            "experiment_id": "synthetic",
            "backend": "fake_backend",
            "job_ids": ["fake_job"],
            "result_rows": rows,
        }
    )

    channel = next(
        row for row in summary["channel_rows"] if row["observable"] == "raw_magnetisation_leakage"
    )
    assert channel["scale1_delta"] == pytest.approx(0.25)
    assert channel["linear_zne_delta"] == pytest.approx(0.25)
    assert summary["readout_model"]["condition_number"] == pytest.approx(1.0)
