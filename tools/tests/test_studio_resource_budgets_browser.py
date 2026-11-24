# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real resource-budget measurement and no-submit custody
"""Measure the declared budgets on the built workbench and fail on actual runtime faults."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING

import pytest

import tools.studio_resource_budgets_browser as budgets_browser
from tools.studio_resource_budgets_browser import (
    host_environment,
    percentile95,
    run_resource_budgets_journey,
)
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host
from tools.tests.test_studio_program_authoring_browser import program_bundle as program_bundle

if TYPE_CHECKING:
    from playwright.sync_api import Page


def test_the_95th_percentile_is_the_nearest_rank() -> None:
    """Twenty samples give the 19th smallest; one sample gives itself."""
    assert percentile95([float(value) for value in range(1, 21)]) == 19.0
    assert percentile95([7.0]) == 7.0


def test_a_wait_that_never_ends_is_a_failed_measurement() -> None:
    """A page that never reaches the awaited state fails the journey with what it showed."""
    from playwright.sync_api import Error

    class Stalled:
        """A target whose in-page timing never completes."""

        def evaluate(self, _script: str, _argument: object = None, /) -> object:
            """Raise as the page does when its wait times out.

            Parameters
            ----------
            _script : str
                The timing script.
            _argument : object
                Its argument.

            Returns
            -------
            object
                Never.

            """
            raise Error(
                "timed out waiting for the full view; the page shows: Showing 500 of 10000 nodes"
            )

    with pytest.raises(
        AssertionError, match="could not be made: timed out waiting for the full view"
    ):
        budgets_browser._timed(Stalled(), "async () => 0")  # a stand-in for the page locator


def test_the_host_environment_says_none_where_the_host_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable or unmatched host file gives None rather than a guess.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the file reader.

    """
    real = host_environment()
    assert isinstance(real["cores"], int) and isinstance(real["load_average"], list)

    def unreadable(_path: Path, encoding: str = "utf-8") -> str:
        raise OSError(encoding)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert (host_environment()["cpu"], host_environment()["memory_kib"]) == (None, None)
    monkeypatch.setattr(Path, "read_text", lambda _path, encoding="utf-8": "")
    assert host_environment()["cpu"] is None


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:4173/",
        "https://127.0.0.1:4173/",
        "http://127.0.0.1:0/",
        "http://user:password@127.0.0.1:4173/",
    ],
)
def test_resource_budgets_journey_refuses_unowned_preview_before_browser_creation(
    url: str,
) -> None:
    """Reject unowned URLs before allocating a browser.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_resource_budgets_journey(url)


def test_resource_budgets_journey_records_every_verdict(program_bundle: Path) -> None:
    """Every declared budget gets a verdict; misses are listed, nothing is relaxed.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_resource_budgets_journey(preview)
    assert observed["no_submit"] is True
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    record = observed["budget_record"]
    assert isinstance(record, dict)
    statuses = {item["id"]: item["status"] for item in record["verdicts"]}
    assert set(statuses) >= {
        "cold_load_lcp",
        "graph_query_p95",
        "worker_cancel_ack",
        "worker_leak",
    }
    assert statuses["trajectory_points"] == "fail"
    assert "trajectory_points" in record["missed"]
    assert statuses["worker_leak"] == "pass"
    assert record["environment"]["cores"]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_resource_budgets_journey_cannot_pass_after_runtime_custody_failure(
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
        run_resource_budgets_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_resource_budgets_journey_refuses_an_actual_same_origin_submission(
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
        run_resource_budgets_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]


def test_resource_budgets_journey_refuses_a_preview_without_an_entry_page(
    program_bundle: Path, tmp_path: Path
) -> None:
    """A preview that serves no entry page for measurement cannot yield a record.

    Parameters
    ----------
    program_bundle
        Original genuine built UI.
    tmp_path
        Owned copy whose entry page is replaced after the load.

    """
    preview_dir = copytree(program_bundle, tmp_path / "built")
    evidence: dict[str, object] = {}
    original = budgets_browser._cold_load

    def cold_then_remove(page: Page, url: str) -> dict[str, object]:
        measured = original(page, url)  # the real cold load runs first
        (preview_dir / "index.html").unlink()
        return measured

    with owned_fault_host(preview_dir) as preview, pytest.MonkeyPatch.context() as patch:
        patch.setattr(budgets_browser, "_cold_load", cold_then_remove)
        with pytest.raises(AssertionError, match="serves no entry page"):
            run_resource_budgets_journey(preview, evidence)
    assert "budget_record" not in evidence


def test_public_resource_budgets_dispatch_writes_the_record(
    program_bundle: Path, tmp_path: Path
) -> None:
    """The shared public runner owns the scenario and writes its record.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.
    tmp_path
        Directory for the evidence file.

    """
    from tools.studio_browser_journey import main

    output = tmp_path / "resource-budgets-journey.json"
    with owned_fault_host(program_bundle) as preview:
        code = main(
            [
                "--scenario",
                "workbench_resource_budgets",
                "--base-url",
                preview,
                "--output",
                str(output),
            ]
        )
    evidence = json.loads(output.read_text())
    assert code == 0
    assert evidence["passed"] is True
    assert evidence["budget_record"]["schema"] == "scpn_qc_studio_resource_budget_record_v1"
