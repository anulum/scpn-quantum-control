# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-bound native collector publication gate
"""Require unrounded complete native collector coverage tied to tested sources."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import coverage

_OWNER = "tools/studio_workspace_browser_coverage.py"
_INPUTS = (
    _OWNER,
    "tools/tests/test_studio_workspace_browser_coverage.py",
    "tools/studio_native_coverage_gate.py",
    "tools/tests/test_studio_native_coverage_gate.py",
    "tools/preflight.py",
    ".pre-commit-config.yaml",
    ".github/workflows/ci-studio.yml",
    ".github/actions/studio-native-coverage/action.yml",
    "requirements-ci-py312-linux.txt",
    "requirements-ci-studio-browser.txt",
    "studio-web/package.json",
    "studio-web/pnpm-lock.yaml",
    "studio-web/quality/browserCohorts.ts",
)


def _source_hashes(root: Path) -> dict[str, str]:
    """Hash explicit qualification inputs and the real browser source trees."""
    paths = {root / name for name in _INPUTS}
    for name in ("src", "browser-tests", "public"):
        paths.update(
            path
            for path in (root / "studio-web" / name).rglob("*")
            if path.is_file() and path.suffix in {".ts", ".tsx", ".json", ".html", ".svg"}
        )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _check_report(filename: Path) -> None:
    """Require a single named owner with complete statement and branch counters."""
    report = json.loads(filename.read_text(encoding="utf-8"))
    owners = [
        value
        for name, value in report["files"].items()
        if name.replace("\\", "/").endswith(_OWNER)
    ]
    if len(owners) != 1 or len(report["files"]) != 1:
        raise ValueError("Exactly one native collector coverage owner required")
    owner = owners[0]
    summary = owner["summary"]
    if (
        summary["num_statements"] <= 0
        or summary["num_branches"] <= 0
        or summary["missing_lines"] != 0
        or summary["missing_branches"] != 0
        or owner["missing_lines"]
        or owner["missing_branches"]
    ):
        raise ValueError("Native collector requires 100% statements and branches before push")


def _check_data_report(root: Path, data: Path, report: Path) -> None:
    """Reconstruct every owner fact from the original arcs and admitted source."""
    original = coverage.CoverageData(basename=str(data))
    original.read()
    files = [
        name
        for name in original.measured_files()
        if ("/" + name.replace("\\", "/")).endswith("/" + _OWNER)
    ]
    if len(files) != 1:
        raise ValueError("Native raw data requires exactly one collector owner")
    filename = files[0]
    if not original.has_arcs():
        raise ValueError("Native raw data requires a branch-enabled collector owner")
    measured = coverage.Coverage(data_file=None, config_file=False)
    measured.get_data().add_arcs({str(root / _OWNER): original.arcs(filename) or []})
    with redirect_stdout(io.StringIO()) as output:
        measured.json_report(morfs=[str(root / _OWNER)], outfile="-")
    reconstructed = json.loads(output.getvalue())
    recorded = json.loads(report.read_text(encoding="utf-8"))
    if (next(iter(recorded["files"].values())), recorded["totals"]) != (
        next(iter(reconstructed["files"].values())),
        reconstructed["totals"],
    ):
        raise ValueError("Native raw data does not support the reported collector facts")


def begin_native_qualification(root: Path, output: Path) -> None:
    """Bind source inputs before collecting fresh native coverage counters.

    Parameters
    ----------
    root : Path
        Canonical repository whose original source inputs will be exercised.
    output : Path
        New owned evidence directory; reusing a qualification is refused.

    Raises
    ------
    FileExistsError
        The qualification directory already exists.

    """
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-before.json").write_text(
        json.dumps(_source_hashes(root.resolve()), indent=2) + "\n", encoding="utf-8"
    )


def finish_native_qualification(root: Path, output: Path, data: Path) -> Path:
    """Check original collected counters and seal them against unchanged source inputs.

    Parameters
    ----------
    root : Path
        Original repository exercised by the completed native acceptance command.
    output : Path
        Evidence directory created before the successful acceptance command.
    data : Path
        Actual branch-enabled coverage database written by that command.

    Returns
    -------
    Path
        Source-bound complete-coverage receipt for the publication check.

    Raises
    ------
    ValueError
        Source inputs changed or statement/branch coverage is incomplete.

    """
    before = json.loads((output / "source-before.json").read_text(encoding="utf-8"))
    if _source_hashes(root.resolve()) != before:
        raise ValueError("Native qualification source inputs changed during execution")
    measured = coverage.Coverage(data_file=str(data), config_file=False)
    measured.load()
    report = output / "native-coverage.json"
    measured.json_report(outfile=str(report), include=[f"*/{_OWNER}"])
    _check_report(report)
    _check_data_report(root.resolve(), data, report)
    receipt = output / "native-coverage-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "source_sha256": before,
                "report": str(report.resolve()),
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "data": str(data.resolve()),
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def check_native_coverage(root: Path, receipt: Path | None = None) -> None:
    """Refuse publication without complete counters for the exact current source inputs.

    Parameters
    ----------
    root : Path
        Current canonical repository; its Git-local configuration selects the receipt.
    receipt : Path | None
        Explicit receipt, otherwise SCPN_STUDIO_NATIVE_COVERAGE_RECEIPT or the
        scpn.studioNativeCoverageReceipt Git configuration is required.

    Raises
    ------
    ValueError
        Proof is absent, stale, altered or below exact statement/branch coverage.
    OSError
        Required raw qualification evidence is unavailable.

    """
    root = root.resolve()
    if receipt is None:
        selected = os.environ.get("SCPN_STUDIO_NATIVE_COVERAGE_RECEIPT")
        if not selected:
            selected = subprocess.run(
                ["git", "config", "--get", "scpn.studioNativeCoverageReceipt"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        if not selected:
            raise ValueError(
                "Source-bound 100% native collector qualification required before push"
            )
        receipt = Path(selected)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if value["source_sha256"] != _source_hashes(root):
        raise ValueError("Native coverage receipt is stale for current source inputs")
    for name in ("report", "data"):
        filename = Path(value[name])
        if hashlib.sha256(filename.read_bytes()).hexdigest() != value[f"{name}_sha256"]:
            raise ValueError("Native coverage evidence digest differs")
    _check_report(Path(value["report"]))
    _check_data_report(root, Path(value["data"]), Path(value["report"]))
