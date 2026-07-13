# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — coverage-debt register audit tests
"""Tests for the deterministic coverage-debt register audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import audit_coverage_debt as audit


@dataclass(frozen=True)
class RepoFixture:
    """Paths and parsed policy for one isolated coverage-debt repository."""

    root: Path
    policy_path: Path
    audit_path: Path
    register_path: Path
    policy: audit.CoverageDebtPolicy


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON and create its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit_row(
    path: str,
    *,
    covered: int | None,
    valid: int | None,
) -> dict[str, object]:
    """Build one coverage-gap JSON row with consistent counts."""
    if covered is None or valid is None:
        return {
            "path": path,
            "line_percent": None,
            "covered_lines": None,
            "valid_lines": None,
            "missing_lines": None,
            "status": "missing_from_report",
            "justification": None,
        }
    return {
        "path": path,
        "line_percent": 100.0 * covered / valid,
        "covered_lines": covered,
        "valid_lines": valid,
        "missing_lines": valid - covered,
        "status": "ok",
        "justification": None,
    }


def _policy_payload(artifact_sha256: str) -> dict[str, object]:
    """Return the valid policy used by isolated tests."""
    return {
        "schema_version": 1,
        "source_root": "src/pkg",
        "register_path": "data/coverage/debt.json",
        "debt_threshold_percent": 100.0,
        "justified_exclusions_path": "docs/exclusions.json",
        "claim_ledger_path": "data/claim_ledger.json",
        "claim_surface_key": "implementation_surface",
        "high_line_debt_minimum": 50,
        "baseline": {
            "artifact_name": "coverage-gap-audit-3.12",
            "artifact_sha256": artifact_sha256,
            "origin_commit": "a" * 40,
            "python_version": "3.12",
            "remote_ci_run": 123,
            "source_url": "https://example.test/actions/runs/123",
        },
        "baseline_invalidated_paths": [
            {"path": "src/pkg/changed.py", "reason": "implementation moved"}
        ],
        "runtime_hot_paths": [{"path": "src/pkg/hot.py", "reason": "runtime dispatch"}],
        "current_artifact_rule": "No new debt and no measured regression.",
    }


def _build_repo(tmp_path: Path) -> RepoFixture:
    """Create a complete isolated repository and its tracked register."""
    root = tmp_path / "repo"
    source = root / "src" / "pkg"
    source.mkdir(parents=True)
    paths = (
        "claim.py",
        "hot.py",
        "changed.py",
        "high.py",
        "standard.py",
        "excluded.py",
        "covered.py",
        "new.py",
    )
    for filename in paths:
        (source / filename).write_text('"""Fixture module."""\n', encoding="utf-8")
    rows = [
        _audit_row("src/pkg/claim.py", covered=8, valid=10),
        _audit_row("src/pkg/hot.py", covered=9, valid=10),
        _audit_row("src/pkg/changed.py", covered=9, valid=10),
        _audit_row("src/pkg/high.py", covered=0, valid=50),
        _audit_row("src/pkg/standard.py", covered=9, valid=10),
        _audit_row("src/pkg/excluded.py", covered=None, valid=None),
        _audit_row("src/pkg/covered.py", covered=10, valid=10),
    ]
    audit_path = root / "baseline.json"
    _write_json(audit_path, rows)
    artifact_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    policy_path = root / "tools" / "policy.json"
    _write_json(policy_path, _policy_payload(artifact_sha))
    _write_json(
        root / "docs" / "exclusions.json",
        {
            "exclusions": [
                {
                    "path_glob": "src/pkg/excluded*.py",
                    "reason": "optional runtime",
                    "exercised_by": "optional lane",
                }
            ]
        },
    )
    _write_json(
        root / "data" / "claim_ledger.json",
        {
            "claims": [
                {
                    "claim_id": "public-claim",
                    "implementation_surface": [
                        "src/pkg/claim.py",
                        "docs/not-source.md",
                    ],
                }
            ]
        },
    )
    policy = audit.load_policy(policy_path)
    register = audit.generate_register(
        project_root=root,
        policy_path=policy_path,
        policy=policy,
        audit_path=audit_path,
        baseline_mode=True,
    )
    register_path = root / policy.register_path
    register_path.parent.mkdir(parents=True)
    register_path.write_text(audit.render_register(register), encoding="utf-8")
    return RepoFixture(root, policy_path, audit_path, register_path, policy)


def test_parse_policy_exposes_complete_contract(tmp_path: Path) -> None:
    """Parse every policy surface used by generation and enforcement."""
    fixture = _build_repo(tmp_path)
    policy = fixture.policy

    assert policy.schema_version == 1
    assert policy.source_root == "src/pkg"
    assert policy.register_path == "data/coverage/debt.json"
    assert policy.debt_threshold_percent == 100.0
    assert policy.high_line_debt_minimum == 50
    assert policy.baseline.remote_ci_run == 123
    assert policy.baseline_invalidated_paths[0].path.endswith("changed.py")
    assert policy.runtime_hot_paths[0].reason == "runtime dispatch"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: [], "policy must be an object"),
        (lambda row: {**row, "schema_version": True}, "schema_version must be an integer"),
        (lambda row: {**row, "schema_version": 2}, "schema_version must be 1"),
        (
            lambda row: {**row, "debt_threshold_percent": "100"},
            "debt_threshold_percent must be a number",
        ),
        (
            lambda row: {**row, "debt_threshold_percent": math.inf},
            "debt_threshold_percent must be finite",
        ),
        (
            lambda row: {**row, "debt_threshold_percent": 99.0},
            "debt_threshold_percent must remain 100",
        ),
        (
            lambda row: {**row, "high_line_debt_minimum": 0},
            "high_line_debt_minimum must be positive",
        ),
        (lambda row: {**row, "source_root": "../src"}, "must be repository-relative"),
        (lambda row: {**row, "register_path": "debt.txt"}, "must name a .json file"),
        (
            lambda row: {**row, "baseline_invalidated_paths": {}},
            "baseline_invalidated_paths must be an array",
        ),
        (
            lambda row: {
                **row,
                "runtime_hot_paths": [
                    {"path": "src/pkg/hot.py", "reason": "first"},
                    {"path": "src/pkg/hot.py", "reason": "second"},
                ],
            },
            "contains duplicate path",
        ),
    ],
)
def test_parse_policy_rejects_invalid_root_fields(mutate: object, message: str) -> None:
    """Reject malformed policy root fields and path-priority arrays."""
    valid = _policy_payload("1" * 64)
    assert callable(mutate)
    payload = mutate(valid)
    with pytest.raises(ValueError, match=message):
        audit.parse_policy(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_sha256", "bad", "must be a lowercase SHA-256"),
        ("origin_commit", "ABC", "must be a full lowercase Git SHA"),
        ("remote_ci_run", True, "must be an integer"),
        ("remote_ci_run", 0, "must be positive"),
        ("source_url", "http://example.test", "must use https"),
        ("artifact_name", "", "must be a non-empty string"),
    ],
)
def test_parse_policy_rejects_invalid_baseline_fields(
    field: str, value: object, message: str
) -> None:
    """Reject unverifiable remote baseline provenance."""
    payload = _policy_payload("1" * 64)
    baseline_value = payload["baseline"]
    assert isinstance(baseline_value, dict)
    baseline = dict(baseline_value)
    baseline[field] = value
    payload["baseline"] = baseline

    with pytest.raises(ValueError, match=message):
        audit.parse_policy(payload)


def test_parse_coverage_audit_accepts_measured_and_missing_rows() -> None:
    """Parse consistent measured and missing coverage rows."""
    rows = audit.parse_coverage_audit(
        [
            _audit_row("src/pkg/a.py", covered=9, valid=10),
            _audit_row("src/pkg/b.py", covered=None, valid=None),
        ]
    )

    assert rows[0].missing_lines == 1
    assert rows[1].line_percent is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "coverage_audit must be an array"),
        ([], "coverage_audit must not be empty"),
        ([{"path": "src/pkg/a.txt"}], "must name a .py file"),
        (
            [
                _audit_row("src/pkg/a.py", covered=1, valid=1),
                _audit_row("src/pkg/a.py", covered=1, valid=1),
            ],
            "duplicate path",
        ),
        (
            [
                {
                    **_audit_row("src/pkg/a.py", covered=None, valid=None),
                    "covered_lines": 0,
                }
            ],
            "must omit all counts",
        ),
        (
            [
                {
                    **_audit_row("src/pkg/a.py", covered=1, valid=1),
                    "line_percent": 101.0,
                }
            ],
            "must be between 0 and 100",
        ),
        (
            [
                {
                    **_audit_row("src/pkg/a.py", covered=1, valid=1),
                    "missing_lines": None,
                }
            ],
            "requires all line counts",
        ),
        (
            [
                {
                    **_audit_row("src/pkg/a.py", covered=1, valid=1),
                    "missing_lines": 1,
                }
            ],
            "inconsistent line counts",
        ),
        (
            [
                {
                    **_audit_row("src/pkg/a.py", covered=1, valid=2),
                    "line_percent": 90.0,
                }
            ],
            "does not match its line counts",
        ),
    ],
)
def test_parse_coverage_audit_rejects_malformed_rows(payload: object, message: str) -> None:
    """Fail closed for malformed coverage evidence."""
    with pytest.raises(ValueError, match=message):
        audit.parse_coverage_audit(payload)


def test_parse_coverage_audit_rejects_negative_counts() -> None:
    """Reject negative optional line counts before consistency checks."""
    row = _audit_row("src/pkg/a.py", covered=1, valid=2)
    row["covered_lines"] = -1

    with pytest.raises(ValueError, match="covered_lines must be non-negative"):
        audit.parse_coverage_audit([row])


def test_generate_register_prioritises_claim_hot_unmeasured_and_large_debt(
    tmp_path: Path,
) -> None:
    """Generate all five priority classes in deterministic order."""
    fixture = _build_repo(tmp_path)
    generated = audit.generate_register(
        project_root=fixture.root,
        policy_path=fixture.policy_path,
        policy=fixture.policy,
        audit_path=fixture.audit_path,
        baseline_mode=True,
    )

    assert [entry.priority for entry in generated.entries] == [
        "P0_claim_bearing",
        "P1_runtime_hot_path",
        "P2_unmeasured",
        "P2_unmeasured",
        "P3_high_line_debt",
        "P4_standard",
    ]
    by_name = {Path(entry.path).name: entry for entry in generated.entries}
    assert by_name["claim.py"].claim_ids == ("public-claim",)
    assert by_name["changed.py"].status == "source_changed_since_baseline"
    assert by_name["new.py"].status == "unmeasured_since_baseline"
    assert "excluded.py" not in by_name
    assert "covered.py" not in by_name
    assert generated.payload["summary"] == {
        "debt_file_count": 6,
        "known_missing_line_count": 54,
        "unmeasured_debt_file_count": 2,
        "fully_covered_file_count": 1,
        "justified_exclusion_file_count": 1,
        "measured_non_excluded_file_count": 5,
        "priority_counts": {
            "P0_claim_bearing": 1,
            "P1_runtime_hot_path": 1,
            "P2_unmeasured": 2,
            "P3_high_line_debt": 1,
            "P4_standard": 1,
        },
        "status_counts": {
            "below_target": 4,
            "source_changed_since_baseline": 1,
            "unmeasured_since_baseline": 1,
        },
    }


def test_generate_register_requires_exact_baseline_digest(tmp_path: Path) -> None:
    """Reject an artifact that is not the policy-pinned remote baseline."""
    fixture = _build_repo(tmp_path)
    fixture.audit_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        audit.generate_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            audit_path=fixture.audit_path,
            baseline_mode=True,
        )


def test_generate_register_requires_configured_paths_to_exist(tmp_path: Path) -> None:
    """Reject stale configured runtime or invalidation paths."""
    fixture = _build_repo(tmp_path)
    (fixture.root / "src" / "pkg" / "hot.py").unlink()

    with pytest.raises(ValueError, match="configured coverage-debt path does not exist"):
        audit.generate_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            audit_path=fixture.audit_path,
            baseline_mode=True,
        )


def test_generate_register_requires_live_source_root(tmp_path: Path) -> None:
    """Reject policies whose package source root does not exist."""
    fixture = _build_repo(tmp_path)
    payload = _policy_payload(fixture.policy.baseline.artifact_sha256)
    payload["source_root"] = "src/missing"
    policy = audit.parse_policy(payload)

    with pytest.raises(ValueError, match="source root does not exist"):
        audit.generate_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=policy,
            audit_path=fixture.audit_path,
            baseline_mode=False,
        )


def test_generate_register_rejects_exclusion_without_path(tmp_path: Path) -> None:
    """Reject a justified-exclusion row with no exact path or glob."""
    fixture = _build_repo(tmp_path)
    _write_json(
        fixture.root / "docs" / "exclusions.json",
        {"exclusions": [{"reason": "missing selector"}]},
    )

    with pytest.raises(ValueError, match="requires path or path_glob"):
        audit.generate_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            audit_path=fixture.audit_path,
            baseline_mode=False,
        )


def test_generate_register_rejects_non_text_claim_surface(tmp_path: Path) -> None:
    """Reject non-text public claim-ledger implementation surfaces."""
    fixture = _build_repo(tmp_path)
    _write_json(
        fixture.root / "data" / "claim_ledger.json",
        {"claims": [{"claim_id": "claim", "implementation_surface": [1]}]},
    )

    with pytest.raises(ValueError, match=r"implementation_surface\[0\] must be text"):
        audit.generate_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            audit_path=fixture.audit_path,
            baseline_mode=False,
        )


def test_current_generation_classifies_absent_rows_as_missing(tmp_path: Path) -> None:
    """Classify paths absent from fresh evidence as report gaps."""
    fixture = _build_repo(tmp_path)
    generated = audit.generate_register(
        project_root=fixture.root,
        policy_path=fixture.policy_path,
        policy=fixture.policy,
        audit_path=fixture.audit_path,
        baseline_mode=False,
    )
    by_name = {Path(entry.path).name: entry for entry in generated.entries}

    assert by_name["new.py"].status == "missing_from_report"
    assert by_name["changed.py"].status == "below_target"


def test_current_generation_classifies_null_measurement_as_missing(tmp_path: Path) -> None:
    """Classify an explicit null coverage row as missing from the report."""
    fixture = _build_repo(tmp_path)
    rows = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
    standard = next(row for row in rows if row["path"].endswith("standard.py"))
    standard.update(_audit_row("src/pkg/standard.py", covered=None, valid=None))
    current_path = fixture.root / "current.json"
    _write_json(current_path, rows)

    generated = audit.generate_register(
        project_root=fixture.root,
        policy_path=fixture.policy_path,
        policy=fixture.policy,
        audit_path=current_path,
        baseline_mode=False,
    )

    by_name = {Path(entry.path).name: entry for entry in generated.entries}
    assert by_name["standard.py"].status == "missing_from_report"


def test_audit_tracked_register_passes_clean_fixture(tmp_path: Path) -> None:
    """Accept a register whose policy and live inputs have not drifted."""
    fixture = _build_repo(tmp_path)

    assert (
        audit.audit_tracked_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            register_path=fixture.register_path,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate debt path"),
        ("priority", "priority is unsupported"),
        ("status", "status is unsupported"),
        ("claim", r"claim_ids\[0\] must be text"),
    ],
)
def test_audit_tracked_register_rejects_malformed_debt_rows(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Reject malformed tracked debt rows before drift comparison."""
    fixture = _build_repo(tmp_path)
    payload = json.loads(fixture.register_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        payload["debt"].append(dict(payload["debt"][0]))
    elif mutation == "priority":
        payload["debt"][0]["priority"] = "P9_unknown"
    elif mutation == "status":
        payload["debt"][0]["status"] = "unknown"
    else:
        payload["debt"][0]["claim_ids"] = [1]
    _write_json(fixture.register_path, payload)

    with pytest.raises(ValueError, match=message):
        audit.audit_tracked_register(
            project_root=fixture.root,
            policy_path=fixture.policy_path,
            policy=fixture.policy,
            register_path=fixture.register_path,
        )


def test_audit_tracked_register_reports_schema_drift(tmp_path: Path) -> None:
    """Report an unsupported tracked register schema version."""
    fixture = _build_repo(tmp_path)
    payload = json.loads(fixture.register_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    _write_json(fixture.register_path, payload)

    errors = audit.audit_tracked_register(
        project_root=fixture.root,
        policy_path=fixture.policy_path,
        policy=fixture.policy,
        register_path=fixture.register_path,
    )

    assert "register.schema_version must be 1" in errors


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("policy", "policy digest is stale"),
        ("claim", "claim_ledger_sha256 is stale"),
        ("exclusion", "justified_exclusions_sha256 is stale"),
        ("source", "source file count is stale"),
        ("order", "not in deterministic priority order"),
        ("claim_ids", "claim IDs are stale"),
        ("priority", "priority metadata is stale"),
        ("debt_count", "debt_file_count is inconsistent"),
        ("missing_count", "known_missing_line_count is inconsistent"),
    ],
)
def test_audit_tracked_register_detects_input_and_row_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Detect every tracked input and derived-row drift class."""
    fixture = _build_repo(tmp_path)
    payload = json.loads(fixture.register_path.read_text(encoding="utf-8"))
    if mutation == "policy":
        policy_payload = json.loads(fixture.policy_path.read_text(encoding="utf-8"))
        policy_payload["current_artifact_rule"] = "Changed governance text."
        _write_json(fixture.policy_path, policy_payload)
    elif mutation == "claim":
        (fixture.root / "data" / "claim_ledger.json").write_text(
            '{"claims": []}\n', encoding="utf-8"
        )
    elif mutation == "exclusion":
        (fixture.root / "docs" / "exclusions.json").write_text(
            '{"exclusions": []}\n', encoding="utf-8"
        )
    elif mutation == "source":
        (fixture.root / "src" / "pkg" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    elif mutation == "order":
        payload["debt"] = list(reversed(payload["debt"]))
        _write_json(fixture.register_path, payload)
    elif mutation == "claim_ids":
        payload["debt"][0]["claim_ids"] = []
        _write_json(fixture.register_path, payload)
    elif mutation == "priority":
        payload["debt"][0]["priority_reason"] = "stale"
        _write_json(fixture.register_path, payload)
    elif mutation == "debt_count":
        payload["summary"]["debt_file_count"] = 999
        _write_json(fixture.register_path, payload)
    else:
        payload["summary"]["known_missing_line_count"] = 999
        _write_json(fixture.register_path, payload)

    errors = audit.audit_tracked_register(
        project_root=fixture.root,
        policy_path=fixture.policy_path,
        policy=fixture.policy,
        register_path=fixture.register_path,
    )

    assert any(message in error for error in errors)


def test_compare_current_debt_accepts_improvements_and_unmeasured_rows() -> None:
    """Accept closed debt and the first measurement of an unmeasured row."""
    baseline = (
        audit.DebtEntry(
            "src/pkg/measured.py",
            "P4_standard",
            "reason",
            "below_target",
            80.0,
            8,
            10,
            2,
            (),
        ),
        audit.DebtEntry(
            "src/pkg/new.py",
            "P2_unmeasured",
            "reason",
            "unmeasured_since_baseline",
            None,
            None,
            None,
            None,
            (),
        ),
    )
    current = (
        audit.DebtEntry(
            "src/pkg/new.py",
            "P4_standard",
            "reason",
            "below_target",
            50.0,
            5,
            10,
            5,
            (),
        ),
    )

    assert audit.compare_current_debt(baseline, current) == ()


def test_compare_current_debt_rejects_new_paths_and_regression() -> None:
    """Reject unregistered debt and increased known missing lines."""
    baseline = (
        audit.DebtEntry(
            "src/pkg/measured.py",
            "P4_standard",
            "reason",
            "below_target",
            90.0,
            9,
            10,
            1,
            (),
        ),
    )
    current = (
        audit.DebtEntry(
            "src/pkg/measured.py",
            "P4_standard",
            "reason",
            "below_target",
            80.0,
            8,
            10,
            2,
            (),
        ),
        audit.DebtEntry(
            "src/pkg/new.py",
            "P4_standard",
            "reason",
            "below_target",
            90.0,
            9,
            10,
            1,
            (),
        ),
    )

    errors = audit.compare_current_debt(baseline, current)

    assert errors == (
        "coverage debt regressed: src/pkg/measured.py 1 -> 2 missing lines",
        "new unregistered coverage debt: src/pkg/new.py",
    )


def test_main_writes_audits_and_checks_current_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the three successful CLI modes."""
    fixture = _build_repo(tmp_path)
    common = [
        "--project-root",
        str(fixture.root),
        "--policy",
        str(fixture.policy_path),
    ]

    assert (
        audit.main([*common, "--coverage-audit", str(fixture.audit_path), "--write-register"]) == 0
    )
    assert audit.main(common) == 0
    assert (
        audit.main([*common, "--coverage-audit", str(fixture.audit_path), "--check-current"]) == 0
    )
    assert "coverage debt: 6 files" in capsys.readouterr().out


def test_main_reports_register_drift_and_missing_register(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return non-zero for a missing or stale tracked register."""
    fixture = _build_repo(tmp_path)
    common = [
        "--project-root",
        str(fixture.root),
        "--policy",
        str(fixture.policy_path),
    ]
    fixture.register_path.unlink()
    assert audit.main(common) == 1
    assert "register missing" in capsys.readouterr().out
    audit.main([*common, "--coverage-audit", str(fixture.audit_path), "--write-register"])
    (fixture.root / "src" / "pkg" / "extra.py").write_text("x = 1\n", encoding="utf-8")

    assert audit.main(common) == 1
    assert "Coverage-debt audit failed" in capsys.readouterr().out


def test_main_rejects_conflicting_or_incomplete_modes(tmp_path: Path) -> None:
    """Expose argparse errors for invalid operation combinations."""
    fixture = _build_repo(tmp_path)
    common = [
        "--project-root",
        str(fixture.root),
        "--policy",
        str(fixture.policy_path),
    ]
    with pytest.raises(SystemExit, match="2"):
        audit.main([*common, "--write-register", "--check-current"])
    with pytest.raises(SystemExit, match="2"):
        audit.main([*common, "--check-current"])
