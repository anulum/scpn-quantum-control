# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real calibration inspector runtime and no-submit custody
"""Exercise the built calibration inspector and fail on actual runtime or network faults."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_calibration_browser import run_calibration_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host
from tools.tests.test_studio_program_authoring_browser import program_bundle as program_bundle


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:4173/",
        "https://127.0.0.1:4173/",
        "http://127.0.0.1:0/",
        "http://user:password@127.0.0.1:4173/",
    ],
)
def test_calibration_journey_refuses_unowned_preview_before_browser_creation(url: str) -> None:
    """Reject unowned URLs before allocating a browser.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_calibration_journey(url)


def test_calibration_journey_through_the_built_operations_view(program_bundle: Path) -> None:
    """The built inspector shows the source record's history, drift and verdicts exactly.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_calibration_journey(preview)
    assert observed["no_submit"] is True
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    assert observed["calibration_storage_write_observations"] == []
    observations = observed["observations"]
    assert isinstance(observations, list)
    assert observations[:3] == [
        "source-record-admitted-by-digest",
        "history-and-exact-drift-visible",
        "plan-verdicts-and-superseded-reference-blocked",
    ]
    history = observed["history"]
    assert isinstance(history, dict)
    assert history["first_drift_changes"] == 50
    assert history["first_set"] != history["current_set"]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_calibration_journey_cannot_pass_after_runtime_custody_failure(
    program_bundle: Path, fault: str
) -> None:
    """Retain partial real evidence and close the owned browser after a fault.

    Parameters
    ----------
    program_bundle
        Actual production UI, retained unchanged.
    fault
        Actual page exception or attempted request outside the owned preview.

    """
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_calibration_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_calibration_journey_refuses_an_actual_same_origin_submission(
    program_bundle: Path, tmp_path: Path
) -> None:
    """An actual POST cannot pass the no-submit browser boundary.

    Parameters
    ----------
    program_bundle
        Original genuine built UI.
    tmp_path
        Owned copy retaining all artifacts while HTML attempts an actual POST.

    """
    preview_dir = copytree(program_bundle, tmp_path / "built")
    html = preview_dir / "index.html"
    html.write_bytes(
        html.read_bytes()
        + b'<script>fetch(location.href,{method:"POST",body:"owned-refusal"}).catch(()=>{});</script>'
    )
    evidence: dict[str, object] = {}
    with owned_fault_host(preview_dir) as preview, pytest.raises(AssertionError):
        run_calibration_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]


def test_public_calibration_dispatch_writes_passing_evidence(
    program_bundle: Path, tmp_path: Path
) -> None:
    """The shared public runner owns the scenario and writes its evidence.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.
    tmp_path
        Directory for the evidence file.

    """
    from tools.studio_browser_journey import main

    output = tmp_path / "calibration-journey.json"
    with owned_fault_host(program_bundle) as preview:
        code = main(
            [
                "--scenario",
                "calibration_snapshot_diff",
                "--base-url",
                preview,
                "--output",
                str(output),
            ]
        )
    evidence = json.loads(output.read_text())
    assert code == 0
    assert evidence["passed"] is True
    assert evidence["record_digest"]
