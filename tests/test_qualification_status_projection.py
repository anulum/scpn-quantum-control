# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — qualification applicability tests
"""Exercise sealed metadata, actual checkout files and executable CI ownership.

The receipts are labelled unassessed contract vectors, not measured scientific
results. Each temporary checkout contains byte-identical production source,
test cohort and CI files from this repository. Negative vectors mutate the
real format/files, without replacing an SDK, runtime or production function.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest
from _qualification_receipt_vectors import (
    COHORT,
    POLICY,
    ROOT,
    SOURCE,
    WORKFLOW,
    build_receipt,
    seal,
)

from scpn_quantum_control.qualification_status_projection import (
    project_qualification_status,
)


def test_qualification_status_projection_01(tmp_path: Path) -> None:
    """A sealed falsification remains falsified despite passed engineering axes."""
    root, receipt, payload = build_receipt(tmp_path)
    payload["scientific_status"] = "falsified"
    digest = seal(receipt, payload)
    before = receipt.read_bytes()
    result = project_qualification_status(receipt, digest, repo_root=root)
    assert result.engineering_qualified
    assert result.scientific_status == "falsified"
    assert result.to_dict()["claim_class"] == "simulation"
    assert "not adjudicated or promoted" in str(result.to_dict()["claim_boundary"])
    assert receipt.read_bytes() == before


def test_qualification_status_projection_02(tmp_path: Path) -> None:
    """Source drift removes applicability and exact byte restoration recovers it."""
    root, receipt, payload = build_receipt(tmp_path)
    digest = seal(receipt, payload)
    source = root / SOURCE
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    result = project_qualification_status(receipt, digest, repo_root=root)
    assert result.source_exists and not result.source_current
    assert not result.engineering_qualified
    assert {status for _, status in result.axes} == {"stale"}
    source.write_bytes(original)
    assert project_qualification_status(receipt, digest, repo_root=root).engineering_qualified
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest


def test_qualification_status_projection_03(tmp_path: Path) -> None:
    """An actually absent module remains unavailable regardless of receipt passes."""
    root, receipt, payload = build_receipt(tmp_path)
    payload["runtime"] = {
        "module": "scpn_missing_qualification_runtime",
        "distribution": "scpn-missing-qualification-runtime",
        "version": "1",
    }
    digest = seal(receipt, payload)
    result = project_qualification_status(receipt, digest, repo_root=root)
    assert result.runtime_status == "unavailable"
    assert not result.engineering_qualified
    assert {status for _, status in result.axes} == {"unavailable"}


def test_qualification_status_projection_04(tmp_path: Path) -> None:
    """Duplicating the actual assurance category refuses qualification entirely."""
    root, receipt, payload = build_receipt(tmp_path)
    digest = seal(receipt, payload)
    policy_path = root / POLICY
    policy = json.loads(policy_path.read_text())
    owner = next(row for row in policy["categories"] if payload["ci_job"] in row["jobs"])
    policy["categories"].append(owner)
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="exactly one CI owner"):
        project_qualification_status(receipt, digest, repo_root=root)
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest


def test_unassessed_science_is_not_promoted(tmp_path: Path) -> None:
    """Engineering passes and digest identity leave science explicitly unassessed."""
    root, receipt, payload = build_receipt(tmp_path)
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    assert result.scientific_status == "unassessed"
    assert result.engineering_qualified
    assert result.to_dict()["axes"] == payload["axes"]
    assert result.ci_owner == f"{WORKFLOW}#resource-budget-gate-quality"


def test_replaced_projection_cannot_restore_a_missing_runtime_badge(tmp_path: Path) -> None:
    """A public immutable projection cannot grant qualification with missing runtime."""
    root, receipt, payload = build_receipt(tmp_path)
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    assert not replace(result, runtime_status="unavailable").engineering_qualified
    assert not replace(result, source_current=False).engineering_qualified
    assert not replace(result, source_exists=False).engineering_qualified
    assert not replace(result, axes=()).engineering_qualified


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "domain_qualification_receipt.v2"),
        ("domain_id", " "),
        ("category", 1),
        ("scientific_status", "green"),
        ("claim_class", "advantage"),
        ("axes", {"forward": "passed"}),
        ("axes", dict.fromkeys(("forward", "derivative", "composition", "backend"), "green")),
        ("source_hashes", {}),
        ("source_hashes", {SOURCE: "not-a-digest"}),
        ("source_hashes", {"../outside.py": "0" * 64}),
        ("test_cohort", []),
        ("test_cohort", [COHORT, COHORT]),
        ("test_cohort", ["tests/absent.py"]),
        ("test_cohort", [SOURCE]),
        ("runtime", {}),
        ("runtime", {"module": "sys", "distribution": "python", "version": ""}),
        ("runtime", {"module": "os", "distribution": "python", "version": "1"}),
        ("runtime", {"module": "os.path", "distribution": "python", "version": "1"}),
        ("ci_job", "unregistered-qualification-job"),
        ("ci_job", "ci-gate"),
    ],
)
def test_malformed_qualification_vector_preserves_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Malformed real-file inputs never obtain a badge or modify their receipt."""
    root, receipt, payload = build_receipt(tmp_path)
    payload[field] = value
    digest = seal(receipt, payload)
    with pytest.raises(ValueError):
        project_qualification_status(receipt, digest, repo_root=root)
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest


def test_runtime_version_drift_withholds_all_engineering_axes(tmp_path: Path) -> None:
    """A receipt for another Python version cannot qualify the current runtime."""
    root, receipt, payload = build_receipt(tmp_path)
    payload["runtime"] = {"module": "sys", "distribution": "python", "version": "0"}
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    assert result.runtime_status == "stale"
    assert not result.engineering_qualified
    assert {status for _, status in result.axes} == {"stale"}


def test_missing_production_source_withholds_qualification(tmp_path: Path) -> None:
    """A deleted bound production file cannot retain any recorded pass."""
    root, receipt, payload = build_receipt(tmp_path)
    digest = seal(receipt, payload)
    (root / SOURCE).unlink()
    result = project_qualification_status(receipt, digest, repo_root=root)
    assert not result.source_exists and not result.source_current
    assert not result.engineering_qualified
    assert {status for _, status in result.axes} == {"unavailable"}
    assert result.scientific_status == "unassessed"


@pytest.mark.parametrize("status", ("failed", "blocked", "unavailable"))
def test_failed_axis_is_independent_of_other_passes(tmp_path: Path, status: str) -> None:
    """A single withheld derivative axis neither erases nor upgrades other axes."""
    root, receipt, payload = build_receipt(tmp_path)
    axes = payload["axes"]
    assert isinstance(axes, dict)
    axes["derivative"] = status
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    assert not result.engineering_qualified
    assert dict(result.axes) == axes
    assert result.blockers == (f"derivative: {status}",)


@pytest.mark.parametrize("status", ("failed", "blocked"))
@pytest.mark.parametrize("drift", ("source-stale", "source-missing", "runtime-missing"))
def test_recorded_failure_is_not_softened_by_staleness(
    tmp_path: Path,
    status: str,
    drift: str,
) -> None:
    """A recorded failure or block stays visible when applicability is also lost."""
    root, receipt, payload = build_receipt(tmp_path)
    axes = payload["axes"]
    assert isinstance(axes, dict)
    axes["backend"] = status
    if drift == "runtime-missing":
        payload["runtime"] = {
            "module": "scpn_missing_qualification_runtime",
            "distribution": "scpn-missing-qualification-runtime",
            "version": "1",
        }
    digest = seal(receipt, payload)
    source = root / SOURCE
    if drift == "source-stale":
        source.write_bytes(source.read_bytes() + b"\n")
    elif drift == "source-missing":
        source.unlink()
    result = project_qualification_status(receipt, digest, repo_root=root)
    withdrawn = "stale" if drift == "source-stale" else "unavailable"
    assert dict(result.axes) == {
        "forward": withdrawn,
        "derivative": withdrawn,
        "composition": withdrawn,
        "backend": status,
    }
    assert f"backend: {status}" in result.blockers
    assert not result.engineering_qualified


@pytest.mark.parametrize("distribution", ("numpy", "scpn-absent-qualification-distribution"))
def test_actual_installed_module_and_distribution_metadata(
    tmp_path: Path,
    distribution: str,
) -> None:
    """Real NumPy availability requires its own installed distribution metadata."""
    root, receipt, payload = build_receipt(tmp_path)
    payload["runtime"] = {
        "module": "numpy",
        "distribution": distribution,
        "version": importlib.metadata.version("numpy"),
    }
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    expected = "available" if distribution == "numpy" else "unavailable"
    assert result.runtime_status == expected
    assert result.engineering_qualified == (expected == "available")


def test_distribution_without_file_inventory_is_unavailable(tmp_path: Path) -> None:
    """An actual installed NumPy metadata copy without RECORD cannot certify ownership."""
    root, receipt, payload = build_receipt(tmp_path)
    installed = importlib.metadata.distribution("numpy")
    metadata = installed.read_text("METADATA")
    assert metadata is not None
    package_path = tmp_path / "incomplete_installation"
    dist_info = package_path / f"numpy-{installed.version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(metadata)
    payload["runtime"] = {"module": "numpy", "distribution": "numpy", "version": installed.version}
    digest = seal(receipt, payload)
    # Alter the real package discovery environment, without replacing an SDK or reader.
    sys.path.insert(0, str(package_path))
    try:
        result = project_qualification_status(receipt, digest, repo_root=root)
    finally:
        sys.path.remove(str(package_path))
    assert result.runtime_status == "unavailable"
    assert not result.engineering_qualified


@pytest.mark.parametrize("missing", (COHORT, WORKFLOW, POLICY))
def test_receipt_requires_source_bound_cohort_and_ci_files(
    tmp_path: Path,
    missing: str,
) -> None:
    """A readable but unbound cohort or ownership declaration is not fresh evidence."""
    root, receipt, payload = build_receipt(tmp_path)
    sources = payload["source_hashes"]
    assert isinstance(sources, dict)
    del sources[missing]
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="source-bound|source-bind"):
        project_qualification_status(receipt, digest, repo_root=root)


@pytest.mark.parametrize("value", ([], 1, None))
def test_receipt_objects_require_the_real_schema(tmp_path: Path, value: object) -> None:
    """Non-object receipt bytes cannot be interpreted as a qualification record."""
    root, receipt, _ = build_receipt(tmp_path)
    receipt.write_text(json.dumps(value))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="string-keyed object"):
        project_qualification_status(receipt, digest, repo_root=root)


def test_large_receipt_is_refused_before_decode(tmp_path: Path) -> None:
    """An oversized actual file is refused without rewriting its bytes."""
    root, receipt, _ = build_receipt(tmp_path)
    receipt.write_bytes(b" " * 1_048_577)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        project_qualification_status(receipt, digest, repo_root=root)
    assert receipt.stat().st_size == 1_048_577


@pytest.mark.parametrize("reference", ("../outside.py", "/tmp/outside.py"))
def test_qualification_source_reference_cannot_escape_checkout(
    tmp_path: Path,
    reference: str,
) -> None:
    """A bound real source does not legitimise another reference outside its checkout."""
    root, receipt, payload = build_receipt(tmp_path)
    sources = payload["source_hashes"]
    assert isinstance(sources, dict)
    sources[reference] = "0" * 64
    with pytest.raises(ValueError, match="repository-relative"):
        project_qualification_status(receipt, seal(receipt, payload), repo_root=root)


@pytest.mark.parametrize("jobs", (None, [1]))
def test_malformed_ci_category_jobs_are_refused(tmp_path: Path, jobs: object) -> None:
    """Malformed policy ownership lists cannot be silently ignored."""
    root, receipt, payload = build_receipt(tmp_path)
    policy = root / POLICY
    data = json.loads(policy.read_text())
    data["categories"][0]["jobs"] = jobs
    policy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CI jobs must be a string list"):
        project_qualification_status(receipt, seal(receipt, payload), repo_root=root)


def test_ci_registered_job_must_exist_in_its_workflow(tmp_path: Path) -> None:
    """A real declaration with an absent executable job cannot qualify its cohort."""
    root, receipt, payload = build_receipt(tmp_path)
    path = root / WORKFLOW
    path.write_text(
        path.read_text().replace(
            "  resource-budget-gate-quality:",
            "  resource-budget-gate-quality-retired:",
        )
    )
    with pytest.raises(ValueError, match="does not execute qualification job"):
        project_qualification_status(receipt, seal(receipt, payload), repo_root=root)


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", 2),
        ("schema_version", True),
        ("categories", {}),
    ),
)
def test_unknown_ci_policy_cannot_qualify(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """A changed policy schema cannot inherit a former executable owner."""
    root, receipt, payload = build_receipt(tmp_path)
    policy = root / POLICY
    data = json.loads(policy.read_text())
    data[field] = value
    policy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown CI ownership policy"):
        project_qualification_status(receipt, seal(receipt, payload), repo_root=root)


def test_receipt_identity_is_required_before_parsing(tmp_path: Path) -> None:
    """A malformed expected digest and a substituted file both fail closed."""
    root, receipt, payload = build_receipt(tmp_path)
    seal(receipt, payload)
    for digest in ("", "0" * 64):
        with pytest.raises(ValueError, match="SHA-256|digest mismatch"):
            project_qualification_status(receipt, digest, repo_root=root)


def test_release_profile_without_receipts_excludes_every_category() -> None:
    """Existing implementation/coverage cannot replace missing runtime receipts."""
    from scpn_quantum_control import build_differentiable_release_profile

    profile = build_differentiable_release_profile()
    assert not profile["baseline_release_ready"]
    assert not profile["included_engineering_domains"]
    assert len(profile["excluded_capabilities"]) == len(profile["support_rows"])
    assert all(row["qualification_status"] == "unavailable" for row in profile["support_rows"])


def test_release_profile_cannot_mix_a_foreign_checkout_with_canonical_ledger(
    tmp_path: Path,
) -> None:
    """A selected checkout must supply its own ledger rather than borrow another tree's claims."""
    from scpn_quantum_control import build_differentiable_release_profile

    with pytest.raises(FileNotFoundError):
        build_differentiable_release_profile(repo_root=tmp_path)


def test_release_profile_refuses_duplicate_or_unknown_domains(tmp_path: Path) -> None:
    """No duplicated identity or invented baseline category reaches Studio rows."""
    from scpn_quantum_control import build_differentiable_release_profile

    _, receipt, payload = build_receipt(tmp_path)
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="unique identities"):
        build_differentiable_release_profile(((receipt, digest), (receipt, digest)))
    payload["category"] = "unregistered-baseline-category"
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="unknown baseline category"):
        build_differentiable_release_profile(((receipt, digest),))


def test_type_checked_test_is_not_a_runtime_qualification_cohort(tmp_path: Path) -> None:
    """Naming a file only in lint/types cannot qualify an execution domain."""
    root, receipt, payload = build_receipt(tmp_path)
    name = "tests/test_resource_budget_gate_quality_gate.py"
    copyfile(ROOT / name, root / name)
    payload["test_cohort"] = [name]
    sources = payload["source_hashes"]
    assert isinstance(sources, dict)
    sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    payload["source_hashes"] = sources
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="runtime cohort"):
        project_qualification_status(receipt, digest, repo_root=root)


def test_runtime_distribution_must_own_the_named_import(tmp_path: Path) -> None:
    """Installed NumPy metadata cannot certify an unrelated stdlib module."""
    root, receipt, payload = build_receipt(tmp_path)
    payload["runtime"] = {
        "module": "json",
        "distribution": "numpy",
        "version": importlib.metadata.version("numpy"),
    }
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="does not own"):
        project_qualification_status(receipt, digest, repo_root=root)


def test_duplicate_scientific_verdict_keys_fail_closed(tmp_path: Path) -> None:
    """A byte-valid receipt cannot overwrite falsification using duplicate keys."""
    root, receipt, payload = build_receipt(tmp_path)
    raw = (
        json.dumps(payload)
        .replace(
            '"scientific_status": "unassessed"',
            '"scientific_status": "falsified", "scientific_status": "supported"',
        )
        .encode()
    )
    receipt.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        project_qualification_status(receipt, hashlib.sha256(raw).hexdigest(), repo_root=root)
    assert receipt.read_bytes() == raw


@pytest.mark.parametrize(
    "command",
    (
        f"python -m pytest tests/test_resource_budget_gate_quality_gate.py\n          echo {COHORT}",
        f"python -m pytest {COHORT}::test_budget_ledger_releases_on_scope_exit",
        f"python -m pytest {COHORT} --collect-only",
        f"python -m pytest {COHORT} -k nonexistent_qualification_case",
        f"python -m pytest {COHORT} -m qualification_subset",
        f"python -m coverage run {COHORT}",
    ),
)
def test_partial_or_echoed_cohort_does_not_qualify(
    tmp_path: Path,
    command: str,
) -> None:
    """A literal shell block or single test selector cannot stand for a full cohort."""
    root, receipt, payload = build_receipt(tmp_path)
    workflow = root / WORKFLOW
    text = workflow.read_text()
    start = text.index("  resource-budget-gate-quality:")
    end = text.find("\n  ", start + 3)
    while end != -1 and text[end + 3 : end + 4] == " ":
        end = text.find("\n  ", end + 3)
    if end == -1:
        end = len(text)
    # The real job remains the owner; only its executable run step is replaced.
    replacement = (
        "  resource-budget-gate-quality:\n"
        "    runs-on: ubuntu-latest\n    steps:\n"
        f"      - run: |\n          {command}\n"
    )
    workflow.write_text(text[:start] + replacement + text[end:])
    sources = payload["source_hashes"]
    assert isinstance(sources, dict)
    sources[WORKFLOW] = hashlib.sha256(workflow.read_bytes()).hexdigest()
    digest = seal(receipt, payload)
    with pytest.raises(ValueError, match="runtime cohort"):
        project_qualification_status(receipt, digest, repo_root=root)


def test_receipt_growing_after_stat_is_refused(tmp_path: Path) -> None:
    """A real FIFO reports size 0 to stat yet delivers more than 1 MiB on read."""
    root, _, _ = build_receipt(tmp_path)
    fifo = tmp_path / "receipt.fifo"
    os.mkfifo(fifo)

    def feed() -> None:
        try:
            with fifo.open("wb") as stream:
                stream.write(b" " * 1_048_577)
        except BrokenPipeError:
            pass

    writer = threading.Thread(target=feed)
    writer.start()
    try:
        with pytest.raises(ValueError, match="exceeds 1 MiB"):
            project_qualification_status(fifo, "0" * 64, repo_root=root)
    finally:
        writer.join(timeout=10)
    assert not writer.is_alive()


def test_inline_run_command_qualifies_the_full_cohort(tmp_path: Path) -> None:
    """A single-line run step that executes the whole cohort file is accepted."""
    root, receipt, payload = build_receipt(tmp_path)
    workflow = root / WORKFLOW
    text = workflow.read_text()
    start = text.index("  resource-budget-gate-quality:")
    end = text.find("\n  ", start + 3)
    while end != -1 and text[end + 3 : end + 4] == " ":
        end = text.find("\n  ", end + 3)
    if end == -1:
        end = len(text)
    replacement = (
        "  resource-budget-gate-quality:\n"
        "    runs-on: ubuntu-latest\n    steps:\n"
        f"      - run: python -m pytest {COHORT} -q\n"
    )
    workflow.write_text(text[:start] + replacement + text[end:])
    sources = payload["source_hashes"]
    assert isinstance(sources, dict)
    sources[WORKFLOW] = hashlib.sha256(workflow.read_bytes()).hexdigest()
    result = project_qualification_status(receipt, seal(receipt, payload), repo_root=root)
    assert result.engineering_qualified
    assert result.test_cohort == (COHORT,)


def test_release_profile_keeps_unassessed_science_without_exclusion_reason(
    tmp_path: Path,
) -> None:
    """An unassessed verdict adds no falsification reason and is not promoted."""
    from scpn_quantum_control import build_differentiable_release_profile

    _, receipt, payload = build_receipt(tmp_path)
    digest = seal(receipt, payload)
    profile = build_differentiable_release_profile(((receipt, digest),), repo_root=ROOT)
    row = next(item for item in profile["support_rows"] if item["category"] == payload["category"])
    assert row["qualification_status"] == "qualified"
    assert row["domains"][0]["scientific_status"] == "unassessed"
    assert profile["included_engineering_domains"] == ["resource_admission_contract_vector"]
    excluded = next(
        item
        for item in profile["excluded_capabilities"]
        if item["category"] == payload["category"]
    )
    assert not any("falsified" in reason for reason in excluded["reasons"])
    assert not profile["baseline_release_ready"]


def test_release_profile_excludes_a_falsified_domain_without_demoting_engineering(
    tmp_path: Path,
) -> None:
    """A falsified verdict withholds release while its engineering axes stay qualified."""
    from scpn_quantum_control import build_differentiable_release_profile

    _, receipt, payload = build_receipt(tmp_path)
    payload["scientific_status"] = "falsified"
    digest = seal(receipt, payload)
    profile = build_differentiable_release_profile(((receipt, digest),), repo_root=ROOT)
    row = next(item for item in profile["support_rows"] if item["category"] == payload["category"])
    assert row["qualification_status"] == "qualified"
    assert row["domains"][0]["scientific_status"] == "falsified"
    excluded = next(
        item
        for item in profile["excluded_capabilities"]
        if item["category"] == payload["category"]
    )
    assert "resource_admission_contract_vector: scientific claim falsified" in excluded["reasons"]
    assert not profile["baseline_release_ready"]
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest
