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
pyo3_member_crate_roots = _audit_rust_kernel_execution.pyo3_member_crate_roots


def _write_crate_with_manifest(root: Path, manifest: str, lib_rs: str = "") -> Path:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
    (src / "lib.rs").write_text(lib_rs, encoding="utf-8")
    return root


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
    # The 177 count spans the engine's own src/ plus its in-tree program-AD replay member crate:
    # after that extraction four PyO3 wrappers (effect-IR forward, value-and-gradient, metadata
    # summary, and the registry metadata mirror) live in program_ad_replay/src and are audited as
    # part of the same shipped Python extension alongside the polyglot forward-integrator kernels.
    assert audit.pyfunction_count == 177
    assert audit.rayon_threaded_count > 0
    assert audit.explicit_simd_count == 0
    assert audit.performance_claim_eligible_count == 0
    assert any(
        record.file == "src/pec.rs"
        and record.symbol == "pec_sample_parallel"
        and record.execution_mode == "rayon_threaded"
        for record in audit.kernel_records
    )
    # the relocated program-AD wrappers are audited under the member crate's real path
    assert any(record.file.startswith("program_ad_replay/src/") for record in audit.kernel_records)


def test_pyo3_member_crate_roots_empty_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    assert pyo3_member_crate_roots(tmp_path) == ()


def test_pyo3_member_crate_roots_finds_nested_pyo3_path_dependency(tmp_path: Path) -> None:
    crate = tmp_path / "engine"
    member = crate / "member"
    _write_crate_with_manifest(member, '[package]\nname = "member"\n')
    _write_crate_with_manifest(
        crate,
        '[dependencies]\nmember = { path = "member", features = ["pyo3"] }\n',
    )

    assert pyo3_member_crate_roots(crate) == (member.resolve(),)


def test_pyo3_member_crate_roots_skips_string_pathless_featureless_and_external(
    tmp_path: Path,
) -> None:
    crate = tmp_path / "engine"
    outside = tmp_path / "outside"
    _write_crate_with_manifest(outside, '[package]\nname = "outside"\n')
    plain = crate / "plain"
    _write_crate_with_manifest(plain, '[package]\nname = "plain"\n')
    _write_crate_with_manifest(
        crate,
        "[dependencies]\n"
        'pyo3 = "0.29"\n'  # string spec, not a table
        'versioned = { version = "1.0", features = ["pyo3"] }\n'  # no path
        'plain = { path = "plain" }\n'  # path, no pyo3 feature
        'plain_other = { path = "plain", features = ["serde"] }\n'  # path, wrong feature
        'outside = { path = "../outside", features = ["pyo3"] }\n',  # pyo3 but not nested
    )

    assert pyo3_member_crate_roots(crate) == ()


def test_pyo3_member_crate_roots_dedupes_repeated_members(tmp_path: Path) -> None:
    crate = tmp_path / "engine"
    member = crate / "member"
    _write_crate_with_manifest(member, '[package]\nname = "member"\n')
    _write_crate_with_manifest(
        crate,
        "[dependencies]\n"
        'member = { path = "member", features = ["pyo3"] }\n'
        'member_alias = { path = "member", features = ["pyo3"] }\n',
    )

    assert pyo3_member_crate_roots(crate) == (member.resolve(),)


def test_scan_crate_includes_nested_pyo3_member_crate(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    member = engine / "member"
    _write_crate_with_manifest(
        member,
        '[package]\nname = "member"\n',
        lib_rs="\n".join(
            [
                "mod kernel;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(kernel::member_kernel, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
    )
    (member / "src" / "kernel.rs").write_text(
        "#[pyfunction]\npub fn member_kernel() -> PyResult<f64> { Ok(2.0) }\n",
        encoding="utf-8",
    )
    _write_crate_with_manifest(
        engine,
        '[dependencies]\nmember = { path = "member", features = ["pyo3"] }\n',
        lib_rs="\n".join(
            [
                "mod module;",
                "fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {",
                "    m.add_function(wrap_pyfunction!(module::engine_kernel, m)?)?;",
                "    Ok(())",
                "}",
            ]
        ),
    )
    (engine / "src" / "module.rs").write_text(
        "#[pyfunction]\npub fn engine_kernel() -> PyResult<f64> { Ok(1.0) }\n",
        encoding="utf-8",
    )

    audit = scan_crate(engine)

    assert audit.status == "pass"
    assert audit.pyfunction_count == 2
    assert {record.file for record in audit.kernel_records} == {
        "src/module.rs",
        "member/src/kernel.rs",
    }
