# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM public artefact exposure audit tests
"""Tests for the IBM public artefact exposure audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_AUDIT_TOOL = ROOT / "tools" / "audit_ibm_public_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("audit_ibm_public_artifacts", _AUDIT_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PRIVATE_ONLY = _MODULE.PRIVATE_ONLY
PUBLIC_HASHED = _MODULE.PUBLIC_HASHED
PUBLIC_PROVENANCE = _MODULE.PUBLIC_PROVENANCE
RAW_COUNTS_REVIEW = _MODULE.RAW_COUNTS_REVIEW
candidate_files = _MODULE.candidate_files
findings_to_json = _MODULE.findings_to_json
format_findings = _MODULE.format_findings
has_private_only_findings = _MODULE.has_private_only_findings
main = _MODULE.main
scan_files = _MODULE.scan_files
scan_text = _MODULE.scan_text

RAW_EXAMPLE_ID = "d" + "6h3e2f3o3rs73caglmg"
RAW_FIM_ID = "d" + "7t53ofljm6s73bc6bj0"


def test_ibm_audit_classifies_raw_job_id_values_as_private_only() -> None:
    findings = scan_text(
        Path("results/example.json"),
        f'{{"backend": "ibm_fez", "job_id": "{RAW_EXAMPLE_ID}", "counts": {{"0": 10}}}}',
    )

    by_label = {finding.label: finding for finding in findings}

    assert by_label["job_id"].classification == PRIVATE_ONLY
    assert by_label["job_id"].severity == "high"
    assert by_label["counts"].classification == RAW_COUNTS_REVIEW
    assert by_label["ibm_fez"].classification == PUBLIC_PROVENANCE


def test_ibm_audit_classifies_nonliteral_job_id_fields_as_hashable_identifier() -> None:
    findings = scan_text(
        Path("docs/hardware_status_ledger.md"),
        "Backend: ibm_kingston\nJob ID: `<private-hash:abc123>`",
    )

    assert any(finding.classification == PUBLIC_HASHED for finding in findings)
    assert any(finding.classification == PUBLIC_PROVENANCE for finding in findings)
    assert not has_private_only_findings(findings)


def test_ibm_audit_flags_runtime_instance_mapping_in_ibm_context() -> None:
    findings = scan_text(
        Path("notebooks/hardware.ipynb"),
        'QiskitRuntimeService(channel="ibm_cloud", instance="crn:v1:bluemix:public:quantum-computing:us-east:a/account::")',
    )

    assert any(
        finding.label == "instance" and finding.classification == PRIVATE_ONLY
        for finding in findings
    )


def test_ibm_audit_accepts_env_var_runtime_instance_mapping() -> None:
    findings = scan_text(
        Path("notebooks/hardware.ipynb"),
        'QiskitRuntimeService(channel="ibm_cloud", instance=ibm_crn, **{"token": ibm_token})',
    )

    assert not has_private_only_findings(findings)


def test_ibm_audit_does_not_flag_generic_project_fields_without_ibm_context() -> None:
    findings = scan_text(
        Path("docs/project_notes.json"),
        '{"project": "documentation cleanup", "group": "maintainers"}',
    )

    assert not findings


def test_ibm_audit_flags_retrieval_manifests_as_private_only() -> None:
    findings = scan_text(
        Path("data/example_manifest.json"),
        '{"retrieval_manifest": "data/phase/retrieval_manifest.json", "backend": "ibm_kingston"}',
    )

    assert any(finding.label == "retrieval_manifest" for finding in findings)
    assert has_private_only_findings(findings)


def test_ibm_audit_flags_raw_ibm_ids_outside_job_fields() -> None:
    findings = scan_text(
        Path("docs/publication.md"),
        f"The public paper cited {RAW_FIM_ID} before sanitisation.",
    )

    assert any(finding.label == "raw_ibm_job_id" for finding in findings)
    assert has_private_only_findings(findings)


def test_ibm_audit_flags_raw_ibm_ids_in_public_paths(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    raw_path = result_dir / f"{RAW_FIM_ID}.json"
    raw_path.write_text('{"backend": "ibm_kingston"}\n', encoding="utf-8")

    findings = scan_files(tmp_path, (Path("results") / f"{RAW_FIM_ID}.json",))

    assert any(finding.label == "raw_ibm_job_id_path" for finding in findings)
    assert has_private_only_findings(findings)


def test_ibm_audit_flags_extended_operational_metadata() -> None:
    findings = scan_text(
        Path("data/scpn_fim_hamiltonian/example.json"),
        (
            '{"credential_source": "vault", "vault_token_kind": "ibm_cloud", '
            '"submitted_at_utc": "2026-05-05T20:24:00Z", "pending_jobs": 3}'
        ),
    )

    labels = {finding.label for finding in findings}
    assert {"credential_source", "vault_token_kind", "submitted_at_utc", "pending_jobs"} <= labels
    assert has_private_only_findings(findings)


def test_ibm_audit_candidate_files_exclude_internal_paths_by_default(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    private_docs = tmp_path / "docs" / "internal"
    private_docs.mkdir(parents=True)
    (tmp_path / "README.md").write_text("ibm_kingston\n", encoding="utf-8")
    (private_docs / "private.md").write_text(
        f'"job_id": "{RAW_EXAMPLE_ID}"\n',
        encoding="utf-8",
    )

    files = candidate_files(tmp_path, tracked_only=False)

    assert Path("README.md") in files
    assert Path("docs") / "internal" / "private.md" not in files


def test_ibm_audit_candidate_files_default_to_public_artifact_scope(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "runner.py").write_text('"job_id": "fixture"\n', encoding="utf-8")
    (tmp_path / "tests" / "test_runner.py").write_text('"job_id": "fixture"\n', encoding="utf-8")
    (tmp_path / "docs" / "hardware.md").write_text("Backend: ibm_kingston\n", encoding="utf-8")

    files = candidate_files(tmp_path, tracked_only=False)

    assert Path("docs/hardware.md") in files
    assert Path("src/runner.py") not in files
    assert Path("tests/test_runner.py") not in files


def test_ibm_audit_candidate_files_can_scan_all_repo_text(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runner.py").write_text('"job_id": "fixture"\n', encoding="utf-8")

    files = candidate_files(tmp_path, tracked_only=False, public_artifacts_only=False)

    assert Path("src/runner.py") in files


def test_ibm_audit_candidate_files_can_include_internal_paths(tmp_path: Path) -> None:
    private_docs = tmp_path / "docs" / "internal"
    private_docs.mkdir(parents=True)
    (private_docs / "private.md").write_text(
        f'"job_id": "{RAW_EXAMPLE_ID}"\n',
        encoding="utf-8",
    )

    files = candidate_files(tmp_path, include_internal=True, tracked_only=False)

    assert Path("docs") / "internal" / "private.md" in files


def test_ibm_audit_json_output_is_deterministic() -> None:
    findings = scan_text(
        Path("results/example.json"),
        f'{{"job_id": "{RAW_EXAMPLE_ID}", "backend": "ibm_fez"}}',
    )

    decoded = json.loads(findings_to_json(findings))

    assert decoded[0]["path"] == "results/example.json"
    assert sorted(decoded[0]) == [
        "classification",
        "label",
        "line",
        "path",
        "reason",
        "severity",
        "snippet",
    ]


def test_ibm_audit_text_summary_reports_counts() -> None:
    findings = scan_text(
        Path("results/example.json"),
        f'{{"job_id": "{RAW_EXAMPLE_ID}", "backend": "ibm_fez"}}',
    )

    summary = format_findings(findings)

    assert "private_only_mapping: 1" in summary
    assert "public_provenance: 1" in summary


def test_ibm_audit_text_summary_can_limit_displayed_findings() -> None:
    raw_ids = ["d" + f"6h3e2f3o3rs73cagl{i:03d}" for i in range(3)]
    findings = scan_text(
        Path("results/example.json"),
        "\n".join(f'{{"job_id": "{raw_id}"}}' for raw_id in raw_ids),
    )

    summary = format_findings(findings, max_findings=1)

    assert "private_only_mapping: 3" in summary
    assert "2 additional findings omitted" in summary


def test_ibm_audit_cli_fail_on_private_returns_nonzero(tmp_path: Path, capsys: object) -> None:
    fixture = tmp_path / "public.json"
    fixture.write_text(
        f'{{"backend": "ibm_fez", "job_id": "{RAW_EXAMPLE_ID}"}}',
        encoding="utf-8",
    )

    assert (
        main(["--input", str(fixture), "--project-root", str(tmp_path), "--fail-on-private"]) == 1
    )
    output = capsys.readouterr().out
    assert "private_only_mapping: 1" in output


def test_ibm_audit_cli_json_mode_returns_zero_without_private_findings(
    tmp_path: Path, capsys: object
) -> None:
    fixture = tmp_path / "public.md"
    fixture.write_text("Backend: ibm_kingston\n", encoding="utf-8")

    assert main(["--input", str(fixture), "--project-root", str(tmp_path), "--json"]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["classification"] == PUBLIC_PROVENANCE
