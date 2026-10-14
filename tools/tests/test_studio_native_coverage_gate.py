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
from pathlib import Path

import pytest

from tools.studio_native_coverage_gate import (
    begin_native_qualification,
    check_native_coverage,
    finish_native_qualification,
)


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
    "damage",
    [
        "source",
        "report",
        "data",
        "owner",
        "duplicate",
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
