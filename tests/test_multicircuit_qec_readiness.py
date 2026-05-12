# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for multi-circuit QEC readiness
"""Tests for the no-QPU multi-circuit QEC readiness package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_multicircuit_qec_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("generate_multicircuit_qec_readiness", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load multi-circuit QEC readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weighted_decoder_matrix_is_symmetric_with_zero_diagonal() -> None:
    module = _load_module()

    matrix = module.weighted_decoder_matrix(3)

    assert matrix.shape == (9, 9)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] > matrix[0, 8]


def test_binomial_ci_contains_observed_rate() -> None:
    module = _load_module()

    low, high = module._binomial_ci(10, 100)

    assert 0.0 <= low <= 0.1 <= high <= 1.0


def test_resource_rows_include_encoded_and_unencoded_comparators() -> None:
    module = _load_module()

    rows = module.resource_rows(module.default_cases())

    methods = {row["method"] for row in rows}
    assert methods == {"unencoded_physical", "distance3_surface_code_offline"}
    assert all(int(row["transpiled_depth"]) >= 0 for row in rows)
    assert all(int(row["raw_qubits"]) >= 4 for row in rows)


def test_summary_and_manifest_use_precise_offline_qec_boundary(tmp_path: Path) -> None:
    module = _load_module()
    decoder_rows = []
    for spec in module.default_cases():
        noise = module.default_noise_models()[0]
        for decoder in (
            "unencoded_physical",
            "standard_mwpm",
            "physics_aware_mwpm",
            "physics_feature_disabled",
        ):
            decoder_rows.append(
                {
                    **spec.to_dict(),
                    **noise.to_dict(),
                    "decoder": decoder,
                    "seed": 0,
                    "trials": 100,
                    "failures": 0,
                    "logical_failure_rate": 0.0,
                    "logical_failure_ci_low": 0.0,
                    "logical_failure_ci_high": 0.0,
                    "syndrome_rate": 0.0,
                    "retained_fraction": 1.0,
                    "decoder_runtime_ms": 1.0,
                }
            )
    resource_rows = module.resource_rows(module.default_cases())

    summary = module.build_summary(decoder_rows, resource_rows)
    _, _, _, md_path = module.write_outputs(
        decoder_rows,
        resource_rows,
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert (
        "distance-3 surface-code offline logical-failure comparison"
        in summary["claim_boundary"]["supported"]
    )
    assert "toy" not in "\n".join(summary["claim_boundary"]["supported"]).lower()
    assert "toy" not in md_path.read_text(encoding="utf-8").lower()


def test_summary_blocks_when_physics_aware_does_not_beat_ablation() -> None:
    module = _load_module()
    decoder_rows = []
    for spec in module.default_cases():
        for noise in module.default_noise_models():
            for decoder in (
                "unencoded_physical",
                "standard_mwpm",
                "physics_aware_mwpm",
                "physics_feature_disabled",
            ):
                decoder_rows.append(
                    {
                        **spec.to_dict(),
                        **noise.to_dict(),
                        "decoder": decoder,
                        "seed": 0,
                        "trials": 100,
                        "failures": 10,
                        "logical_failure_rate": 0.10,
                        "logical_failure_ci_low": 0.05,
                        "logical_failure_ci_high": 0.15,
                        "syndrome_rate": 0.2,
                        "retained_fraction": 1.0,
                        "decoder_runtime_ms": 1.0,
                    }
                )

    summary = module.build_summary(decoder_rows, module.resource_rows(module.default_cases()))

    assert summary["schema"] == "scpn_phase3_multicircuit_qec_readiness_v1"
    assert summary["hardware_submission"] is False
    assert summary["qpu_minutes_spent"] == 0.0
    assert summary["ready_for_optional_hardware"] is False
    assert summary["readiness_decision"] == "blocked_physics_aware_decoder_did_not_beat_baselines"


def test_write_outputs_records_manifest_and_hashes(tmp_path: Path) -> None:
    module = _load_module()
    decoder_rows = []
    for spec in module.default_cases():
        noise = module.default_noise_models()[0]
        for decoder in (
            "unencoded_physical",
            "standard_mwpm",
            "physics_aware_mwpm",
            "physics_feature_disabled",
        ):
            decoder_rows.append(
                {
                    **spec.to_dict(),
                    **noise.to_dict(),
                    "decoder": decoder,
                    "seed": 0,
                    "trials": 100,
                    "failures": 0,
                    "logical_failure_rate": 0.0,
                    "logical_failure_ci_low": 0.0,
                    "logical_failure_ci_high": 0.0,
                    "syndrome_rate": 0.0,
                    "retained_fraction": 1.0,
                    "decoder_runtime_ms": 1.0,
                }
            )
    summary = module.build_summary(decoder_rows, module.resource_rows(module.default_cases()))

    json_path, decoder_path, resource_path, md_path = module.write_outputs(
        decoder_rows,
        module.resource_rows(module.default_cases()),
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert decoder_path.exists()
    assert resource_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "Hardware submission: `False`" in manifest
    assert "QPU minutes spent: `0.0`" in manifest
