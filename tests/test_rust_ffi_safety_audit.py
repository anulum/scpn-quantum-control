# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rust FFI safety audit tests
"""Tests for the Rust FFI safety inventory gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_rust_ffi_safety = _load_tool_module(
    "audit_rust_ffi_safety_for_tests",
    "audit_rust_ffi_safety.py",
)

scan_crate = _audit_rust_ffi_safety.scan_crate
audit_to_json = _audit_rust_ffi_safety.audit_to_json
main = _audit_rust_ffi_safety.main


def _write_rust_fixture(tmp_path: Path, lib_rs: str, module_rs: str) -> Path:
    crate = tmp_path / "crate"
    src = crate / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(lib_rs, encoding="utf-8")
    (src / "module.rs").write_text(module_rs, encoding="utf-8")
    return crate


def test_rust_ffi_safety_audit_inventory_safe_pyo3_fixture(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="\n".join(
            [
                "mod module;",
                "#[pymodule]",
                "fn scpn_quantum_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::safe_kernel, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
        module_rs="\n".join(
            [
                "#[pyfunction]",
                "pub fn safe_kernel(value: f64) -> PyResult<f64> {",
                "    Ok(value)",
                "}",
            ]
        ),
    )

    audit = scan_crate(crate)

    assert audit.status == "pass"
    assert audit.unsafe_occurrence_count == 0
    assert audit.pyfunction_count == 1
    assert audit.pymodule_count == 1
    assert audit.unregistered_pyfunction_count == 0
    assert any(
        boundary.symbol == "safe_kernel" and boundary.registered
        for boundary in audit.pyo3_boundaries
    )


def test_rust_ffi_safety_audit_fails_on_unsafe_token(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="#[pymodule]\nfn m(_m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }\n",
        module_rs="\n".join(
            [
                "#[pyfunction]",
                "pub fn unchecked(values: &[f64]) -> PyResult<f64> {",
                "    let first = unsafe { *values.get_unchecked(0) };",
                "    Ok(first)",
                "}",
            ]
        ),
    )

    audit = scan_crate(crate)

    assert audit.status == "fail"
    assert audit.unsafe_occurrence_count == 1
    assert audit.unsafe_occurrences[0].occurrence_kind == "block"
    assert audit.unsafe_occurrences[0].symbol == "unchecked"


def test_rust_ffi_safety_json_output_is_deterministic(tmp_path: Path) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="#[pymodule]\nfn m(_m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }\n",
        module_rs="#[pyfunction]\npub fn safe_kernel() -> PyResult<()> { Ok(()) }\n",
    )

    decoded = json.loads(audit_to_json(scan_crate(crate)))

    assert set(decoded) == {
        "claim_boundary",
        "crate_root",
        "extern_c_count",
        "pymodule_count",
        "pyfunction_count",
        "pyo3_boundaries",
        "schema",
        "status",
        "unregistered_pyfunction_count",
        "unsafe_occurrence_count",
        "unsafe_occurrences",
    }
    assert decoded["schema"] == "scpn-rust-ffi-safety-audit/v1"


def test_rust_ffi_safety_cli_writes_json_and_returns_status(
    tmp_path: Path, capsys: object
) -> None:
    crate = _write_rust_fixture(
        tmp_path,
        lib_rs="#[pymodule]\nfn m(_m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }\n",
        module_rs="pub fn safe_inner() -> f64 { 1.0 }\n",
    )
    output = tmp_path / "audit.json"

    assert main(["--crate-root", str(crate), "--json", "--output", str(output)]) == 0
    assert "Rust FFI safety audit: pass" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_live_rust_crate_has_no_unsafe_and_inventories_pyo3_boundary() -> None:
    crate = Path(__file__).resolve().parents[1] / "scpn_quantum_engine"

    audit = scan_crate(crate)

    assert audit.status == "pass"
    assert audit.unsafe_occurrence_count == 0
    assert audit.pyfunction_count > 100
    assert audit.pymodule_count == 1
    assert audit.unregistered_pyfunction_count == 0
    assert any(
        boundary.file == "src/compiler_ad.rs"
        and boundary.symbol == "matrix_2x2_determinant_value"
        and boundary.registered
        for boundary in audit.pyo3_boundaries
    )
