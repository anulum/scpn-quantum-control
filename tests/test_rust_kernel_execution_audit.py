# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rust kernel execution-mode audit tests
"""Tests for Rust kernel SIMD/threading evidence inventory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_rust_kernel_execution = _load_tool_module(
    "audit_rust_kernel_execution_for_tests",
    "audit_rust_kernel_execution.py",
)

scan_crate = _audit_rust_kernel_execution.scan_crate
audit_to_json = _audit_rust_kernel_execution.audit_to_json
main = _audit_rust_kernel_execution.main


def _write_rust_fixture(tmp_path: Path, lib_rs: str, module_rs: str) -> Path:
    crate = tmp_path / "crate"
    src = crate / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(lib_rs, encoding="utf-8")
    (src / "module.rs").write_text(module_rs, encoding="utf-8")
    return crate


def test_kernel_execution_audit_classifies_rayon_threaded_fixture(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="\n".join(
            [
                "mod module;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::parallel_sum, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
        module_rs="\n".join(
            [
                "use rayon::prelude::*;",
                "#[pyfunction]",
                "pub fn parallel_sum(values: Vec<f64>) -> PyResult<f64> {",
                "    Ok(values.par_iter().sum())",
                "}",
            ]
        ),
    )

    audit = scan_crate(crate)

    assert audit.status == "pass"
    assert audit.pyfunction_count == 1
    assert audit.rayon_threaded_count == 1
    assert audit.explicit_simd_count == 0
    assert audit.kernel_records[0].execution_mode == "rayon_threaded"
    assert audit.kernel_records[0].performance_claim_eligible is False


def test_kernel_execution_audit_classifies_explicit_simd_fixture(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="\n".join(
            [
                "mod module;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::simd_sum, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
        module_rs="\n".join(
            [
                "use std::simd::Simd;",
                "#[pyfunction]",
                "pub fn simd_sum(values: Vec<f64>) -> PyResult<f64> {",
                "    let _lane = Simd::<f64, 4>::splat(0.0);",
                "    Ok(values.iter().sum())",
                "}",
            ]
        ),
    )

    audit = scan_crate(crate)

    assert audit.explicit_simd_count == 1
    assert audit.kernel_records[0].execution_mode == "explicit_simd"
    assert "std::simd::Simd" in audit.kernel_records[0].evidence_tokens


def test_kernel_execution_json_output_is_deterministic(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="\n".join(
            [
                "mod module;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::scalar, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
        module_rs="#[pyfunction]\npub fn scalar() -> PyResult<f64> { Ok(1.0) }\n",
    )

    decoded = json.loads(audit_to_json(scan_crate(crate)))

    assert set(decoded) == {
        "claim_boundary",
        "crate_root",
        "explicit_simd_count",
        "kernel_records",
        "ndarray_dot_count",
        "performance_claim_eligible_count",
        "pyfunction_count",
        "rayon_threaded_count",
        "scalar_or_unknown_count",
        "schema",
        "status",
    }
    assert decoded["schema"] == "scpn-rust-kernel-execution-audit/v1"


def test_kernel_execution_cli_writes_json(tmp_path: Path, capsys: Any) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="\n".join(
            [
                "mod module;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::scalar, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
        module_rs="#[pyfunction]\npub fn scalar() -> PyResult<f64> { Ok(1.0) }\n",
    )
    output = tmp_path / "audit.json"

    assert main(["--crate-root", str(crate), "--json", "--output", str(output)]) == 0
    assert "Rust kernel execution audit: pass" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_live_rust_crate_records_threading_without_simd_promotion() -> None:
    crate = Path(__file__).resolve().parents[1] / "scpn_quantum_engine"

    audit = scan_crate(crate)

    assert audit.status == "pass"
    # 137 original kernels plus the Kuramoto order-parameter gradient and Hessian, the
    # mean-phase value, gradient and Hessian, and the Daido order-parameter value and
    # gradient PyO3 exports.
    assert audit.pyfunction_count == 144
    assert audit.rayon_threaded_count > 0
    assert audit.explicit_simd_count == 0
    assert audit.performance_claim_eligible_count == 0
    assert any(
        record.file == "src/pec.rs"
        and record.symbol == "pec_sample_parallel"
        and record.execution_mode == "rayon_threaded"
        for record in audit.kernel_records
    )
