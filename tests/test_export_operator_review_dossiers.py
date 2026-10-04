# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native dossier exporter acceptance
"""Exercise real native example variants, exclusive files and physical CLI entry."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.studio.workspace import read_json
from tools.export_operator_review_dossiers import (
    EXAMPLE_CASES,
    build_operator_review_example,
    main,
)


@pytest.mark.parametrize("case", EXAMPLE_CASES)
def test_native_review_case_preserves_original_identity(case: str) -> None:
    """Every browser conformance case originates in the real native public handler."""
    bundle = build_operator_review_example(case)
    raw = cast(dict[str, Any], bundle["body"])
    dossier = cast(dict[str, Any], read_json(raw["dossier_text"]))
    body = dossier["body"]
    assert raw["dossier_sha256"] == dossier["sha256"]
    assert body["no_submit"] is True
    assert body["plan"]["parameters"]["shots"] == (512 if case == "changed_shots" else 1024)
    assert body["plan"]["parameters"]["endpoint"] == (
        "synthetic-device-b" if case == "changed_target" else "synthetic-device"
    )
    assert body["policy_decision"]["allowed"] is (case != "unknown_price")
    assert body["settings"]["body"]["effective"]["seed"] == 9007199254740993
    assert (
        body["calibration"] is None
        if case == "unknown_calibration"
        else body["calibration"]["sha256"] == ("d" if case == "changed_calibration" else "c") * 64
    )
    assert raw["script"]["filename"] == "verify_operator_review.py"
    assert body["profile"]["capabilities"].get("max_shots") == (
        2048 if case == "declared_shot_capacity" else None
    )


def test_theme_only_native_case_preserves_execution_and_price_dates() -> None:
    """Display metadata changes the original source hash, not semantic execution."""
    bodies = [
        cast(
            dict[str, Any],
            read_json(
                cast(dict[str, Any], build_operator_review_example(case)["body"])["dossier_text"]
            ),
        )
        for case in ("pending", "theme_light")
    ]
    assert bodies[0]["sha256"] != bodies[1]["sha256"]
    assert bodies[0]["body"]["execution_sha256"] == bodies[1]["body"]["execution_sha256"]
    assert (
        bodies[0]["body"]["policy_decision"]["estimate"]["observed_at"] == "2026-10-03T00:00:00Z"
    )


def test_export_cli_checks_exact_bytes_and_never_overwrites(tmp_path: Path) -> None:
    """Missing, exact and stale checks retain custody, including refused overwrite."""
    output = tmp_path / "review.json"
    args = ["--output", str(output)]
    assert main([*args, "--check"]) == 1
    assert main(args) == 0
    original = output.read_bytes()
    assert main([*args, "--check"]) == 0
    assert main([*args, "--case", "changed_payload", "--check"]) == 1
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "case,as_of", [("unknown", "2026-10-04T00:00:00Z"), ("pending", "2026-02-30T00:00:00Z")]
)
def test_unknown_cases_and_invalid_dates_refuse(case: str, as_of: str) -> None:
    """Invalid inputs cannot silently turn into default source evidence."""
    with pytest.raises(ValueError):
        build_operator_review_example(case, as_of=as_of)


def test_actual_export_cli_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real command entry calls the native producer with an explicit source date."""
    output = tmp_path / "fresh.json"
    args = ["--output", str(output), "--as-of", "2026-12-01T12:00:00Z"]
    monkeypatch.setattr(sys, "argv", ["export_operator_review_dossiers.py", *args])
    with pytest.raises(SystemExit) as result:
        runpy.run_path("tools/export_operator_review_dossiers.py", run_name="__main__")
    assert result.value.code == 0
    assert main([*args, "--check"]) == 0
