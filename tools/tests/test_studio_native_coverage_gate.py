# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native coverage publication proof tests
"""Exercise publication checks with original collected counters and damaged proof copies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import coverage
import pytest

from tools.studio_browser_journey import loopback_url
from tools.studio_native_coverage_gate import (
    begin_native_qualification,
    check_native_coverage,
    finish_native_qualification,
)
from tools.studio_workspace_browser_coverage import verify_native_coverage


@pytest.fixture
def qualified_proof() -> tuple[Path, Path]:
    """Use a real complete collector qualification made before this test cohort.

    Returns
    -------
    tuple[Path, Path]
        Original repository and source-bound receipt with its original raw counters.

    """
    root = Path(__file__).resolve().parents[2]
    receipt = Path(os.environ["STUDIO_NATIVE_GATE_RECEIPT"])
    check_native_coverage(root, receipt)
    return root, receipt


def test_original_source_bound_counters_are_accepted(qualified_proof: tuple[Path, Path]) -> None:
    """Accept original unrounded complete counters without modifying any evidence.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Actual repository and genuine current-source qualification.

    """
    root, receipt = qualified_proof
    before = receipt.read_bytes()
    check_native_coverage(root, receipt)
    assert receipt.read_bytes() == before


def test_explicit_environment_receipt_is_accepted(
    qualified_proof: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real pre-push receipt selection through its public environment.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Genuine repository and complete measured proof.
    monkeypatch : pytest.MonkeyPatch
        Scope the public process configuration, without replacing Git or coverage.

    """
    root, receipt = qualified_proof
    monkeypatch.setenv("SCPN_STUDIO_NATIVE_COVERAGE_RECEIPT", str(receipt))
    check_native_coverage(root)


def test_missing_receipt_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a real filesystem directory with no configured publication proof.

    Parameters
    ----------
    tmp_path : Path
        Unconfigured real directory, never an invented coverage/source fixture.
    monkeypatch : pytest.MonkeyPatch
        Isolate public environment and Git configuration from the caller.

    """
    monkeypatch.delenv("SCPN_STUDIO_NATIVE_COVERAGE_RECEIPT", raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-global-config"))
    with pytest.raises(ValueError, match="qualification required"):
        check_native_coverage(tmp_path)


def test_qualification_lifecycle_uses_actual_collected_data(
    qualified_proof: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Check source binding and real coverage reporting through the public lifecycle.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Original source and actual collected coverage database.
    tmp_path : Path
        Owned output directory for this proof validation roundtrip.

    """
    root, receipt = qualified_proof
    data = Path(json.loads(receipt.read_text())["data"])
    output = tmp_path / "qualification"
    begin_native_qualification(root, output)
    with pytest.raises(FileExistsError):
        begin_native_qualification(root, output)
    completed = finish_native_qualification(root, output, data)
    check_native_coverage(root, completed)


def test_changed_source_binding_cannot_finish(
    qualified_proof: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Refuse a corrupted pre-execution source binding without editing original sources.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Actual source and original raw database.
    tmp_path : Path
        Owned evidence copy directory.

    """
    root, receipt = qualified_proof
    output = tmp_path / "qualification"
    begin_native_qualification(root, output)
    before = output / "source-before.json"
    value = json.loads(before.read_text())
    del value[next(iter(value))]
    before.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source inputs changed"):
        finish_native_qualification(root, output, Path(json.loads(receipt.read_text())["data"]))


@pytest.mark.parametrize(
    "changed_input",
    [
        ".github/actions/studio-provider-observation/action.yml",
        "data/studio/provider_observation_cases.json",
        ".github/actions/studio-method-comparison/action.yml",
        "data/studio/method_comparison_cases.json",
    ],
)
def test_provider_observation_inputs_are_bound_before_execution(
    tmp_path: Path, changed_input: str
) -> None:
    """Refuse changed original action/capture bytes before consuming any counter data.

    Parameters
    ----------
    tmp_path
        Owned complete copy of the actual selected source inputs, not invented counters.
    changed_input
        Original executable action or captured native observation corpus.

    """
    root = Path(__file__).resolve().parents[2]
    reference = tmp_path / "reference"
    begin_native_qualification(root, reference)
    inputs = json.loads((reference / "source-before.json").read_text())
    assert changed_input in inputs
    assert inputs[changed_input] == hashlib.sha256((root / changed_input).read_bytes()).hexdigest()
    copied = tmp_path / "original-inputs"
    for relative in inputs:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    output = tmp_path / "qualification"
    begin_native_qualification(copied, output)
    changed = copied / changed_input
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source inputs changed"):
        finish_native_qualification(copied, output, tmp_path / "absent-counter-data")
    assert not (output / "native-coverage-receipt.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "report",
        "data",
        "owner",
        "duplicate",
        "extra_owner",
        "totals",
        "statements",
        "branches",
        "missing_lines",
        "missing_branches",
        "line_list",
        "branch_list",
    ],
)
def test_damaged_qualification_cannot_authorize_push(
    qualified_proof: tuple[Path, Path],
    tmp_path: Path,
    damage: str,
) -> None:
    """Refuse explicit damage to copies of genuine proof, never fabricated native counters.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Actual source-bound complete qualification.
    tmp_path : Path
        Directory for explicitly damaged proof copies.
    damage : str
        Stale source, altered raw evidence, missing owner or incomplete report metadata.

    """
    root, receipt = qualified_proof
    value = json.loads(receipt.read_text())
    if damage == "source":
        value["source_sha256"] = {}
    elif damage in {"report", "data"}:
        copied = tmp_path / damage
        copied.write_bytes(Path(value[damage]).read_bytes() + b"declared corruption")
        value[damage] = str(copied)
    else:
        report = json.loads(Path(value["report"]).read_text())
        name = next(iter(report["files"]))
        owner = report["files"][name]
        if damage == "owner":
            report["files"] = {}
        elif damage == "duplicate":
            report["files"]["duplicate/" + name] = owner
        elif damage == "extra_owner":
            report["files"]["unrelated.py"] = owner
        elif damage == "totals":
            report["totals"]["num_statements"] = 0
        elif damage in {"statements", "branches"}:
            owner["summary"]["num_" + damage] = 0
        elif damage in {"missing_lines", "missing_branches"}:
            owner["summary"][damage] = 1
        elif damage == "line_list":
            owner["missing_lines"] = [1]
        else:
            owner["missing_branches"] = [[1, 2]]
        copied = tmp_path / "damaged-report.json"
        copied.write_text(json.dumps(report))
        value["report"] = str(copied)
        value["report_sha256"] = hashlib.sha256(copied.read_bytes()).hexdigest()
    damaged = tmp_path / "damaged-receipt.json"
    damaged.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        check_native_coverage(root, damaged)


@pytest.mark.parametrize("mode", ["foreign", "mixed", "lines", "partial"])
def test_original_report_refuses_unrelated_or_incomplete_recorded_data(
    qualified_proof: tuple[Path, Path], tmp_path: Path, mode: str
) -> None:
    """Reject actual mismatched measurements and accept the restored original pair.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Genuine complete native report and its original recorded data.
    tmp_path : Path
        Owned directory for actual independent measurements and receipt copies.
    mode : str
        Other-owner, mixed-owner, statement-only or partial native measurement.

    """
    root, receipt = qualified_proof
    original = receipt.read_bytes()
    value = json.loads(original)
    check_native_coverage(root, receipt)
    snapshot = tmp_path / "cohorts.json"
    if mode != "foreign":
        result = subprocess.run(
            [
                "node",
                "--experimental-strip-types",
                "--input-type=module",
                "-e",
                'import {readBrowserCohorts} from "./quality/browserCohorts.ts"; '
                "process.stdout.write(JSON.stringify(await readBrowserCohorts(process.cwd())));",
            ],
            cwd=root / "studio-web",
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=os.environ | {"NODE_OPTIONS": "--max-old-space-size=768"},
        )
        snapshot.write_text(result.stdout, encoding="utf-8")
    owners = []
    if mode in {"foreign", "mixed"}:
        owners.append(str(root / "tools/studio_native_coverage_gate.py"))
    if mode != "foreign":
        owners.append(str(root / "tools/studio_workspace_browser_coverage.py"))
    data = tmp_path / "independent.coverage"
    measured = coverage.Coverage(
        data_file=str(data), config_file=False, branch=mode != "lines", include=owners
    )
    measured.start()
    try:
        if mode in {"foreign", "mixed"}:
            begin_native_qualification(root, tmp_path / "independent-source-binding")
        if mode != "foreign":
            verify_native_coverage(
                Path(os.environ["STUDIO_NATIVE_COVERAGE_ARTIFACT"]),
                snapshot,
                root / "studio-web",
            )
    finally:
        measured.stop()
        measured.save()
    value["data"] = str(data)
    value["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    substituted = tmp_path / "substituted-receipt.json"
    substituted.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Native raw data"):
        check_native_coverage(root, substituted)
    substituted.write_bytes(original)
    check_native_coverage(root, substituted)
    assert receipt.read_bytes() == original
    assert hashlib.sha256(Path(value["report"]).read_bytes()).hexdigest() == value["report_sha256"]


def test_original_recording_remains_valid_for_an_identical_source_checkout(
    qualified_proof: tuple[Path, Path], tmp_path: Path
) -> None:
    """Keep genuine raw paths portable while comparing exact admitted source bytes.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Original complete recording, source map and native report.
    tmp_path : Path
        Separate checkout containing only unchanged admitted source files.

    """
    root, receipt = qualified_proof
    value = json.loads(receipt.read_text(encoding="utf-8"))
    relocated = tmp_path / "same-source-checkout"
    for name, digest in value["source_sha256"].items():
        destination = relocated / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    check_native_coverage(relocated, receipt)


def test_complete_native_recording_in_combined_browser_data_is_accepted(
    qualified_proof: tuple[Path, Path], tmp_path: Path
) -> None:
    """Preserve the actual CI contract that combines native and other browser owners.

    Parameters
    ----------
    qualified_proof : tuple[Path, Path]
        Actual complete native recording and its unchanged original report.
    tmp_path : Path
        Owned directory for the genuine other-owner recording and combined data.

    """
    root, receipt = qualified_proof
    value = json.loads(receipt.read_text(encoding="utf-8"))
    original_data = Path(value["data"])
    original_digest = hashlib.sha256(original_data.read_bytes()).hexdigest()
    foreign_path = tmp_path / "other-browser-owner.coverage"
    measured = coverage.Coverage(
        data_file=str(foreign_path),
        config_file=False,
        branch=True,
        include=[str(root / "tools/studio_browser_journey.py")],
    )
    measured.start()
    try:
        assert loopback_url("http://127.0.0.1:4198/") == "http://127.0.0.1:4198/"
    finally:
        measured.stop()
        measured.save()
    combined_path = tmp_path / "combined-browser.coverage"
    combined = coverage.CoverageData(basename=str(combined_path))
    recordings: list[coverage.CoverageData] = []
    for path in (original_data, foreign_path):
        raw = coverage.CoverageData(basename=str(path))
        raw.read()
        recordings.append(raw)
        combined.update(raw)
    combined.write()
    assert recordings[1].measured_files() == {str(root / "tools/studio_browser_journey.py")}
    assert combined.measured_files() == (
        recordings[0].measured_files() | recordings[1].measured_files()
    )
    native_owner = next(
        name
        for name in recordings[0].measured_files()
        if Path(name).name == "studio_workspace_browser_coverage.py"
    )
    assert combined.arcs(native_owner) == recordings[0].arcs(native_owner)
    value["data"] = str(combined_path)
    value["data_sha256"] = hashlib.sha256(combined_path.read_bytes()).hexdigest()
    candidate = tmp_path / "combined-receipt.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")
    check_native_coverage(root, candidate)
    assert hashlib.sha256(original_data.read_bytes()).hexdigest() == original_digest
    assert hashlib.sha256(Path(value["report"]).read_bytes()).hexdigest() == value["report_sha256"]
