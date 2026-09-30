# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — descriptive production naming audit tests
"""Exercise the repository policy for descriptive production names."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_calendar_negative_value_does_not_exempt_campaign_names(tmp_path: Path) -> None:
    """Preserve the ISO week-date refusal fixture without exempting its test owner."""
    path = "tests/test_provider_route_catalogue.py"
    _write(tmp_path / path, 'VALUES = ["2026-W36-6", "post_w7_campaign"]\n')
    findings = audit_paths(tmp_path, (path,))
    assert [finding.value for finding in findings] == ["post_w7_campaign"]
    other = "tests/test_other_calendar.py"
    _write(tmp_path / other, 'VALUE = "2026-W36-6"\n')
    assert [finding.value for finding in audit_paths(tmp_path, (other,))] == ["2026-W36-6"]


def test_python_identifiers_descriptions_and_machine_names_fail(tmp_path: Path) -> None:
    """Python-facing names must describe their domain role."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(
        source,
        "\n".join(
            [
                # REUSE-IgnoreStart
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                # REUSE-IgnoreEnd
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
        "source text",
        "tracked path",
        "workflow name",
    }


def test_internal_traceability_comment_fails_on_public_source(tmp_path: Path) -> None:
    """Internal traceability belongs in coordination records, not source comments."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(
        source,
        "\n".join(
            [
                # REUSE-IgnoreStart
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                # REUSE-IgnoreEnd
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
    findings = audit_paths(tmp_path, ("src/package/surface.py",))
    assert [(finding.kind, finding.line) for finding in findings] == [("source comment", 9)]


def test_root_docs_notebooks_tests_and_hyphenated_polyglot_text_fail(
    tmp_path: Path,
) -> None:
    """The audit covers every public surface previously missed by the scanner."""
    _write(tmp_path / "ROADMAP.md", "# Product\n\nCompleted BL-19.\n")
    _write(
        tmp_path / "notebooks" / "study.ipynb",
        '{"cells": [{"cell_type": "markdown", "source": ["BL-20 study"]}]}',
    )
    _write(tmp_path / "tests" / "test_surface.py", '"""Validate BL-21."""\n')
    _write(tmp_path / "studio-web" / "src" / "panel.tsx", "// Product panel (ST-12)\n")

    findings = audit_paths(
        tmp_path,
        (
            "ROADMAP.md",
            "notebooks/study.ipynb",
            "tests/test_surface.py",
            "studio-web/src/panel.tsx",
        ),
    )

    assert {finding.kind for finding in findings} == {
        "JSON machine name",
        "module description",
        "public documentation text",
        "source text",
    }


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


def test_letter_suffixed_work_item_code_fails(tmp_path: Path) -> None:
    """A letter suffix must not hide an internal code from the audit."""
    source = tmp_path / "src" / "package" / "surface.py"
    _write(source, '"""Calibration capture formerly tracked as AUD-4b."""\n')

    findings = audit_paths(tmp_path, ("src/package/surface.py",))

    assert [(finding.kind, finding.value) for finding in findings] == [
        ("module description", "Calibration capture formerly tracked as AUD-4b."),
    ]


def test_decimal_work_item_and_campaign_stage_shorthand_fail(tmp_path: Path) -> None:
    """Decimal queue codes and abbreviated campaign stages stay internal."""
    public_doc = "docs/campaign.md"
    _write(
        tmp_path / public_doc,
        "# Calibration campaign\n\nQWC-5.3 ran in W7 before post-W7 review.\n",
    )
    evidence = "data/calibration_counts_w7_2026-09-04.json"
    _write(tmp_path / evidence, '{"status": "post_w7_pre_w8_sensitivity"}')

    findings = audit_paths(tmp_path, (public_doc, evidence))

    assert {finding.kind for finding in findings} == {
        "JSON machine name",
        "public documentation text",
        "tracked path",
    }


def test_campaign_stage_shorthand_fails_across_code_and_workflows(tmp_path: Path) -> None:
    """The stage-name guard covers Python, polyglot code, JSON, and CI names."""
    python = "src/package/surface.py"
    _write(tmp_path / python, "def analyse_w7_results() -> None:\n    pass\n")
    rust = "scpn_quantum_engine/src/lib.rs"
    _write(tmp_path / rust, "fn analyse_w7_results() {}\n")
    workflow = ".github/workflows/checks.yml"
    _write(tmp_path / workflow, "jobs:\n  analyse_w7:\n    name: Analyse W7 results\n")
    evidence = "data/result.json"
    _write(tmp_path / evidence, '{"post_w7_status": "complete"}')

    findings = audit_paths(tmp_path, (python, rust, workflow, evidence))

    assert {finding.kind for finding in findings} == {
        "JSON machine name",
        "Python identifier",
        "source identifier",
        "workflow job ID",
        "workflow name",
    }


def test_scientific_weight_identifiers_are_permitted(tmp_path: Path) -> None:
    """Lowercase mathematical weight names are not campaign-stage labels."""
    source = tmp_path / "src" / "package" / "weights.py"
    _write(source, "def encode(w1: float) -> float:\n    return w1\n")

    assert audit_paths(tmp_path, ("src/package/weights.py",)) == ()


def test_exact_stale_contract_fixture_does_not_hide_other_codes(tmp_path: Path) -> None:
    """Allow only the exact obsolete values rejected by the binding-spec test."""
    source = tmp_path / "tests" / "test_binding_spec.py"
    _write(source, 'STALE = ("ws_0", "ws_1", "ws_2")\nOTHER = "BL-19"\n')

    findings = audit_paths(tmp_path, ("tests/test_binding_spec.py",))

    assert [(finding.kind, finding.value) for finding in findings] == [
        ("machine-facing string", "BL-19"),
    ]


def test_scientific_thermal_energy_symbol_is_permitted(tmp_path: Path) -> None:
    """Do not mistake thermal energy at body temperature for a task code."""
    notebook = tmp_path / "notebooks" / "biochemical_validation.ipynb"
    _write(
        notebook,
        '{"cells": [{"cell_type": "code", "source": ["kT_37C = 1.38e-23 * 310"]}]}',
    )

    assert audit_paths(tmp_path, ("notebooks/biochemical_validation.ipynb",)) == ()


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


def test_opaque_embedded_payload_is_not_misclassified_as_naming_debt(
    tmp_path: Path,
) -> None:
    """Do not interpret incidental tokens inside long encoded payloads as names."""
    evidence = "data/product.json"
    _write(
        tmp_path / evidence,
        '{"encoded": "' + "A" * 600 + "W7" + "A" * 3496 + "BL-19" + '"}',
    )

    assert audit_paths(tmp_path, (evidence,)) == ()


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


def test_split_owner_and_programme_codes_are_rejected(tmp_path: Path) -> None:
    """Audit task codes in real source, dataset values and tracked path names."""
    codes = ("QSP-02", "QD2", "QS5", "CORE-E05", "F06", "D001", "D999")
    for index, code in enumerate(codes):
        source = f"src/package/surface_{index}.py"
        data = f"data/surface_{index}.json"
        path = f"data/{code}.json"
        _write(tmp_path / source, f'VALUE = "{code}"\n')
        _write(tmp_path / data, '{"owner": "' + code + '"}')
        _write(tmp_path / path, "{}")
        findings = audit_paths(tmp_path, (source, data, path))
        assert {(item.path, item.value) for item in findings} == {
            (source, code),
            (data, code),
            (path, path),
        }


def test_scientific_standards_and_lint_codes_remain_valid(tmp_path: Path) -> None:
    """Scientific names and documented docstring rule identifiers are public terms."""
    source = "tools/documentation.py"
    _write(
        tmp_path / source,
        'VALUES = ("SHA-256", "QAOA", "D413", "D417", "D420", "D421", "D100", "f32", "f64", "_f64", "f21", "iqm_dla_core_n4_d10_even", "d800", "CORE")\n',
    )
    assert audit_paths(tmp_path, (source,)) == ()


def test_exact_obsolete_removal_owner_does_not_exempt_new_aliases(tmp_path: Path) -> None:
    """Keep only the stale contract refusal literal in the guard test owner."""
    source = "tests/test_split_boundary_guard.py"
    _write(tmp_path / source, 'STALE = "QSP-08"\nNEW = "QSP-07"\n')
    assert [item.value for item in audit_paths(tmp_path, (source,))] == ["QSP-07"]


def _naming_cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the public audit command against a real repository."""
    tool = Path(__file__).resolve().parents[1] / "tools/audit_descriptive_production_naming.py"
    return subprocess.run(
        [sys.executable, str(tool), "--repo", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_writes_a_baseline_and_rejects_growth(tmp_path: Path) -> None:
    """Exercise baseline creation, acceptance and new source debt through the CLI."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path / "tools/descriptive_production_naming_baseline.json", "{}")
    _write(tmp_path / "src/package/surface.py", 'OWNER = "QSP-02"\n')
    written = _naming_cli(tmp_path, "--write-baseline")
    assert written.returncode == 0, written.stderr
    assert "wrote 1 known findings" in written.stdout
    assert _naming_cli(tmp_path).returncode == 0
    _write(tmp_path / "src/package/another.py", 'OWNER = "QD2"\n')
    rejected = _naming_cli(tmp_path)
    assert rejected.returncode == 1
    assert "another.py:1: machine-facing string: QD2" in rejected.stdout
    assert "1 new finding(s)" in rejected.stdout


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"schema":"old"}',
        json.dumps(
            {
                "schema": "scpn_qc.descriptive_production_naming_baseline.v1",
                "known_finding_counts": [],
            }
        ),
        json.dumps(
            {
                "schema": "scpn_qc.descriptive_production_naming_baseline.v1",
                "known_finding_counts": {"invalid": 1},
            }
        ),
    ],
)
def test_cli_refuses_invalid_counted_baselines(tmp_path: Path, payload: str) -> None:
    """Malformed baseline schemas and invalid debt entries fail the public command."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path / "tools/descriptive_production_naming_baseline.json", payload)
    rejected = _naming_cli(tmp_path)
    assert rejected.returncode == 1
    assert "baseline is invalid" in rejected.stdout


def test_comment_only_source_and_workflow_steps_are_audited(tmp_path: Path) -> None:
    """Parse empty source bodies and inspect codes inside workflow commands."""
    source = "src/package/empty.py"
    workflow = ".github/workflows/checks.yml"
    _write(tmp_path / source, "# QSP-02\n")
    _write(tmp_path / workflow, "jobs:\n  check:\n    steps:\n      - run: echo QSP-02\n")
    findings = audit_paths(tmp_path, (source, workflow))
    assert {item.kind for item in findings} == {"source comment", "workflow text"}


def test_deleted_tracked_coded_path_still_fails(tmp_path: Path) -> None:
    """Removing a tracked file does not excuse an internal-code path in the index."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = "src/package/QSP-02.py"
    _write(tmp_path / source, "VALUE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", source], check=True)
    (tmp_path / source).unlink()
    findings = audit_repository(tmp_path)
    assert [(item.path, item.kind) for item in findings] == [(source, "tracked path")]
