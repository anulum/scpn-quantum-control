# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 entanglement/tomography analysis tests
"""Tests for Phase 3 entanglement/tomography raw-count analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyse_phase3_entanglement_tomography.py"
    )
    spec = importlib.util.spec_from_file_location("analyse_phase3_entanglement_tomography", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement/tomography analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pauli_expectation_from_counts_uses_non_identity_parity() -> None:
    module = _load_module()

    assert module.pauli_expectation_from_counts({"0000": 10}, "ZZII") == 1.0
    assert module.pauli_expectation_from_counts({"1000": 10}, "ZZII") == -1.0
    assert module.pauli_expectation_from_counts({"0000": 5, "1000": 5}, "ZZII") == 0.0


def test_analyse_counts_artifact_groups_repetitions_against_reference(tmp_path: Path) -> None:
    module = _load_module()
    counts_path = tmp_path / "counts.json"
    refs_path = tmp_path / "refs.csv"
    counts_path.write_text(
        json.dumps(
            {
                "schema": "scpn_phase3_entanglement_tomography_live_v1",
                "backend": "fake_heron",
                "status": "completed",
                "circuits": [
                    {
                        "job_id": "job-main",
                        "meta": {
                            "block": "main",
                            "family": "dla_parity",
                            "label": "dla_even_signal",
                            "initial": "0011",
                            "depth": 10,
                            "lambda_fim": None,
                            "basis_setting": "ZZII",
                            "rep": 0,
                        },
                        "counts": {"0000": 8, "1000": 2},
                    },
                    {
                        "job_id": "job-main",
                        "meta": {
                            "block": "main",
                            "family": "dla_parity",
                            "label": "dla_even_signal",
                            "initial": "0011",
                            "depth": 10,
                            "lambda_fim": None,
                            "basis_setting": "ZZII",
                            "rep": 1,
                        },
                        "counts": {"0000": 10},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with refs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "label",
                "initial",
                "depth",
                "lambda_fim",
                "observable",
                "pauli_label",
                "basis_setting",
                "exact_expectation",
                "half_chain_purity",
                "parity_survival_ideal",
                "shots_per_setting",
                "repetitions",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "family": "dla_parity",
                "label": "dla_even_signal",
                "initial": "0011",
                "depth": "10",
                "lambda_fim": "",
                "observable": "ZZ_q0q1",
                "pauli_label": "ZZII",
                "basis_setting": "ZZII",
                "exact_expectation": "0.9",
                "half_chain_purity": "0.55",
                "parity_survival_ideal": "1.0",
                "shots_per_setting": "2048",
                "repetitions": "3",
            }
        )

    rows, summary = module.analyse_counts_artifact(counts_path, refs_path)

    assert summary["backend"] == "fake_heron"
    assert summary["n_observable_rows"] == 1
    assert rows[0]["mean_expectation"] == 0.8
    assert rows[0]["exact_expectation"] == 0.9
    assert rows[0]["absolute_deviation"] == pytest.approx(0.1)
