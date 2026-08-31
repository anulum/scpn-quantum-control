# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — descriptive production naming audit tests
"""Exercise the repository policy for descriptive production names."""

from __future__ import annotations

from pathlib import Path

from tools.audit_descriptive_production_naming import (
    audit_paths,
    audit_repository,
    baseline_payload,
    finding_fingerprint,
    load_baseline,
    unexpected_findings,
)


def _write(path: Path, text: str) -> None:
    """Create one UTF-8 audit fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_repository_has_only_descriptive_production_names() -> None:
    """The live tree must not add unregistered internal-code leakage."""
    root = Path.cwd()
    findings = audit_repository(root)
    baseline = load_baseline(root / "tools/descriptive_production_naming_baseline.json")
    assert unexpected_findings(findings, baseline) == ()


def test_python_identifiers_descriptions_and_machine_names_fail(tmp_path: Path) -> None:
    """Python-facing names must describe their domain role."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(
        source,
        "\n".join(
            [
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                "# Commercial license available",
                "# copyright",
                "# copyright",
                "# ORCID",
                "# Contact",
                "# Product heading (BL-19)",
                '"""Product module for BL-19."""',
                "BL19_POINTER = 'bl19_payload'",
            ]
        ),
    )
    findings = audit_paths(tmp_path, ("src/package/surface.py",))
    assert {finding.kind for finding in findings} == {
        "module description",
        "module heading",
        "Python identifier",
        "machine-facing string",
    }


def test_paths_json_workflows_and_other_languages_fail(tmp_path: Path) -> None:
    """Task codes must not name artefacts, payload fields, CI, or polyglot symbols."""
    coded_path = "data/results/bl19_evidence.json"
    _write(tmp_path / coded_path, '{"bl19_key": "safe"}')
    workflow = ".github/workflows/checks.yml"
    _write(tmp_path / workflow, "jobs:\n  lint:\n    name: Product checks (ST-12)\n")
    rust = "scpn_quantum_engine/src/lib.rs"
    _write(tmp_path / rust, "fn bl19_runner() {}\n")
    public_doc = "docs/product.md"
    _write(tmp_path / public_doc, "# Product surface (BL-19)\n")
    findings = audit_paths(tmp_path, (coded_path, workflow, rust, public_doc))
    assert {finding.kind for finding in findings} == {
        "JSON machine name",
        "documentation heading",
        "source identifier",
        "tracked path",
        "workflow name",
    }


def test_descriptive_names_and_human_traceability_prose_pass(tmp_path: Path) -> None:
    """Internal comments may retain traceability outside runtime surfaces."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(
        source,
        "\n".join(
            [
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                "# Commercial license available",
                "# copyright",
                "# copyright",
                "# ORCID",
                "# Contact",
                "# Hardware safety policy",
                '"""Fail-closed hardware execution policy."""',
                "# Historical work item BL-19 established this boundary.",
                "HARDWARE_SAFETY_POINTER = 'hardware_safety_policy'",
            ]
        ),
    )
    assert audit_paths(tmp_path, ("src/package/surface.py",)) == ()


def test_python_docstrings_and_runtime_messages_fail(tmp_path: Path) -> None:
    """Public documentation and runtime errors must use domain language."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(
        source,
        "\n".join(
            [
                '"""Descriptive module."""',
                "def validate() -> None:",
                '    """Validate the BL-19 contract."""',
                '    raise ValueError("BL-19 input is invalid")',
            ]
        ),
    )

    findings = audit_paths(tmp_path, ("src/package/surface.py",))

    assert {finding.kind for finding in findings} == {
        "production docstring",
        "runtime or user-facing string",
    }


def test_public_documentation_body_and_json_prose_fail(tmp_path: Path) -> None:
    """Catch internal codes outside headings and identifier-like JSON values."""
    public_doc = "docs/product.md"
    _write(tmp_path / public_doc, "# Product\n\nImplements the BL-19 workflow.\n")
    evidence = "data/product.json"
    _write(tmp_path / evidence, '{"summary": "Evidence for BL-19."}')

    findings = audit_paths(tmp_path, (public_doc, evidence))

    assert {finding.kind for finding in findings} == {
        "JSON machine name",
        "public documentation text",
    }


def test_counted_baseline_allows_removal_but_rejects_duplicates(tmp_path: Path) -> None:
    """Ratchet known debt downward while rejecting a duplicated violation."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(source, 'ERROR = "BL-19 input is invalid"\n')
    finding = audit_paths(tmp_path, ("src/package/surface.py",))[0]
    payload = baseline_payload((finding,))
    counts = payload["known_finding_counts"]
    assert isinstance(counts, dict)

    assert unexpected_findings((), counts) == ()
    assert unexpected_findings((finding,), counts) == ()
    assert unexpected_findings((finding, finding), counts) == (finding,)
    assert finding_fingerprint(finding) in counts
