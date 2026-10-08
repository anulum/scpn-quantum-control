# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum kuramoto split audit tests
# scpn-quantum-control -- S6 split audit tests
"""Tests for the S6 quantum-kuramoto split audit."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_quantum_kuramoto_split import (
    audit_module_source,
    build_split_audit,
    write_split_audit,
)


def test_split_audit_has_rows_and_status_counts() -> None:
    """Build the actual source audit with reusable and research rows."""
    payload = build_split_audit()

    assert payload["schema"] == "s6_quantum_kuramoto_split_audit_v1"
    assert payload["rows"]
    statuses = payload["statuses"]
    assert isinstance(statuses, dict)
    assert statuses["reusable"] > 0
    assert statuses["scpn_specific"] > 0


def test_split_audit_keeps_package_publish_blocked() -> None:
    """Keep publication closed while the boundary is only audited."""
    payload = build_split_audit()
    boundary = payload["acceptance_boundary"]

    assert isinstance(boundary, dict)
    assert boundary["safe_to_publish_package_now"] is False
    assert "no package skeleton" in str(boundary["reason"])


def test_split_audit_classifies_core_and_scpn_specific_modules() -> None:
    """Classify the current physical Kuramoto and SSGF sources."""
    payload = build_split_audit()
    rows_payload = payload["rows"]
    assert isinstance(rows_payload, list)
    rows = {row["module"]: row for row in rows_payload}

    xy = rows["scpn_quantum_control.phase.xy_kuramoto"]
    assert xy["status"] == "needs_review"
    assert "imports_non_foundation_scpn_module" in xy["reasons"]
    assert "core_kuramoto_candidate" in xy["reasons"]
    assert rows["scpn_quantum_control.bridge.ssgf_adapter"]["status"] == "scpn_specific"


def test_split_audit_records_real_accelerator_alias_and_foundation() -> None:
    """Bind the historical import address to its real provider and bytes."""
    payload = build_split_audit()
    rows = payload["rows"]
    assert isinstance(rows, list)
    by_module = {row["module"]: row for row in rows}
    requested = "scpn_quantum_control.accel.rust_import"
    provider = importlib.import_module(requested)
    assert provider is importlib.import_module("oscillatools.accel.rust_import")
    assert provider.__file__ is not None
    source = Path(provider.__file__)
    row = by_module[requested]
    assert row["canonical_module"] == provider.__name__
    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row["path"] == str(source.relative_to(Path(__file__).resolve().parents[1]))
    assert "verified_compatibility_alias" in row["reasons"]
    assert "scpn_quantum_control._rust_accel" in by_module
    assert payload["candidate_root_modules"] == ["_rust_accel"]


@pytest.mark.parametrize("leaf", ["networked_kuramoto", "dispatcher"])
def test_split_audit_resolves_real_provider_relative_imports(leaf: str) -> None:
    """Resolve provider imports inside the actual accelerator package."""
    canonical = f"oscillatools.accel.{leaf}"
    provider = importlib.import_module(canonical)
    requested = f"scpn_quantum_control.accel.{leaf}"
    assert importlib.import_module(requested) is provider
    payload = build_split_audit(compatibility_aliases={requested: canonical})
    rows = payload["rows"]
    assert isinstance(rows, list)
    row = next(row for row in rows if row["module"] == requested)
    assert row["canonical_module"] == canonical
    assert all(
        not name.startswith("scpn_quantum_control.accel") for name in row["internal_imports"]
    )
    if leaf == "networked_kuramoto":
        assert "oscillatools" in row["external_import_roots"]


def test_split_audit_records_external_physical_provider() -> None:
    """Record a genuine external package without inventing a local path."""
    payload = build_split_audit(compatibility_aliases={"json": "json"})
    rows = payload["rows"]
    assert isinstance(rows, list)
    row = next(row for row in rows if row["module"] == "json")
    provider = importlib.import_module("json")
    assert provider.__file__ is not None
    assert row["path"] == "external:json"
    assert "json" in row["external_import_roots"]
    assert row["source_sha256"] == hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "requested,canonical",
    [
        ("scpn_quantum_control.accel.rust_import", "wrong_owner.rust_import"),
        ("sys", "sys"),
    ],
)
def test_split_audit_rejects_unverified_alias_source(requested: str, canonical: str) -> None:
    """Reject wrong provider identity and modules without physical source."""
    with pytest.raises(ValueError, match="no verified canonical source"):
        build_split_audit(compatibility_aliases={requested: canonical})


@pytest.fixture
def offline_source_root(tmp_path: Path) -> Path:
    """Create an offline AST input with the actual accelerator foundation bytes."""
    root = tmp_path / "snapshot"
    phase = root / "phase"
    phase.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "src/scpn_quantum_control/_rust_accel.py"
    (root / "_rust_accel.py").write_bytes(original.read_bytes())
    (phase / "imports.py").write_text(
        "import scpn_quantum_control.analysis\n"
        "import numpy\n"
        "from scpn_quantum_control.analysis import finite_size_scaling\n"
        "from numpy.linalg import norm\n"
        "from . import xy_kuramoto\n",
        encoding="utf-8",
    )
    cache = phase / "__pycache__"
    cache.mkdir()
    (cache / "invalid.py").write_text("not Python source!", encoding="utf-8")
    return root


def test_offline_split_audit_retains_actual_import_edges(offline_source_root: Path) -> None:
    """Audit real input files without executing their import statements."""
    payload = build_split_audit(source_root=offline_source_root, compatibility_aliases={})
    rows = payload["rows"]
    assert isinstance(rows, list)
    by_module = {row["module"]: row for row in rows}
    assert set(by_module) == {
        "scpn_quantum_control.phase.imports",
        "scpn_quantum_control._rust_accel",
    }
    row = by_module["scpn_quantum_control.phase.imports"]
    assert row["internal_imports"] == (
        "scpn_quantum_control.analysis",
        "scpn_quantum_control.phase",
    )
    assert row["external_import_roots"] == ("numpy",)
    assert row["status"] == "needs_review"
    assert (
        row["source_sha256"]
        == hashlib.sha256((offline_source_root / "phase/imports.py").read_bytes()).hexdigest()
    )
    assert payload["source_root"] == str(offline_source_root)


def test_source_audit_rejects_import_outside_declared_package(tmp_path: Path) -> None:
    """Fail closed instead of silently losing an unresolved relative edge."""
    source = tmp_path / "invalid_context.py"
    source.write_text("from ... import outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relative import leaves declared package"):
        audit_module_source("scpn_quantum_control.phase.snapshot", source)


def test_source_audit_rejects_invalid_python(tmp_path: Path) -> None:
    """Reject a physical input that cannot be parsed as Python."""
    source = tmp_path / "invalid.py"
    source.write_text("from .", encoding="utf-8")
    with pytest.raises(SyntaxError):
        audit_module_source("scpn_quantum_control.phase.snapshot", source)


@pytest.mark.parametrize("field,value", [("statuses", []), ("rows", {})])
def test_split_export_rejects_unrenderable_audit(
    tmp_path: Path, offline_source_root: Path, field: str, value: object
) -> None:
    """Validate stored input before creating either output artifact."""
    payload = build_split_audit(source_root=offline_source_root, compatibility_aliases={})
    payload[field] = value
    output = tmp_path / "output.json"
    document = tmp_path / "output.md"
    with pytest.raises(TypeError, match=f"{field} must be"):
        write_split_audit(payload, json_path=output, doc_path=document)
    assert not output.exists()
    assert not document.exists()


def test_split_export_binds_json_and_markdown_bytes(
    tmp_path: Path, offline_source_root: Path
) -> None:
    """Export an offline audit through the same public writer as the CLI."""
    payload = build_split_audit(source_root=offline_source_root, compatibility_aliases={})
    output = tmp_path / "output.json"
    document = tmp_path / "output.md"
    json_digest, doc_digest = write_split_audit(payload, json_path=output, doc_path=document)
    assert json_digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert doc_digest == hashlib.sha256(document.read_bytes()).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(payload))
    assert "scpn_quantum_control.phase.imports" in document.read_text(encoding="utf-8")


def test_offline_source_root_is_exercised_through_cli(
    tmp_path: Path, offline_source_root: Path
) -> None:
    """Run the actual audit CLI on stored source inputs and real aliases."""
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "cli"
    document = output / "audit.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/audit_quantum_kuramoto_split.py"),
            "--source-root",
            str(offline_source_root),
            "--out-dir",
            str(output),
            "--doc-path",
            str(document),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(
        (output / "quantum_kuramoto_split_audit_2026-05-07.json").read_text(encoding="utf-8")
    )
    assert payload["source_root"] == str(offline_source_root)
    assert len(payload["rows"]) == 3
    assert payload["acceptance_boundary"]["safe_to_publish_package_now"] is False
    assert document.is_file()
