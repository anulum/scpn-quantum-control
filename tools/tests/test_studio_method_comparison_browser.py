# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Real native method-comparison browser boundaries
"""Exercise built and original-source controls; no numerical/browser API mocks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tools.studio_method_comparison_browser import run_method_comparison_journey
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
def test_method_comparison_refuses_unowned_preview(url: str) -> None:
    """Reject unowned origins before opening a native browser."""
    with pytest.raises(ValueError):
        run_method_comparison_journey(url)


@pytest.mark.parametrize("source", ["http://127.0.0.1:4173/", "http://127.0.0.1:4174/prefix/"])
def test_method_comparison_refuses_same_or_nonroot_source(source: str) -> None:
    """Native counters require a distinct original source origin."""
    with pytest.raises(ValueError):
        run_method_comparison_journey("http://127.0.0.1:4173/", source_url=source)


def test_built_method_comparison_retained_population_and_original_wasm(
    program_bundle: Path,
) -> None:
    """Whole source-owned import/selection/refusal/download preserves original workspace."""
    with owned_fault_host(program_bundle) as preview:
        observed = run_method_comparison_journey(preview)
    assert observed["no_submit"] is True
    assert observed["storage_write_observations"] == []
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    records = observed["records"]
    assert isinstance(records, list) and len(records) == 11
    transports = observed["file_transport_checks"]
    assert isinstance(transports, list) and len(transports) == 5


def test_source_method_comparison_collects_real_download_and_source_counters(
    program_bundle: Path,
) -> None:
    """Retain actual debugger code/ranges for the same original source and DOM download."""
    source = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if source is None:
        raise RuntimeError("Supply the owned original Vite source in STUDIO_WORKSPACE_SOURCE_URL")
    with owned_fault_host(program_bundle) as preview:
        observed = run_method_comparison_journey(preview, source_url=source)
    assert observed["coverage_percentage"] == "not_calculated"
    assert observed["native_v8_coverage"]
    records = observed["records"]
    assert isinstance(records, list) and len(records) == 22
    transports = observed["file_transport_checks"]
    assert isinstance(transports, list) and len(transports) == 10
    assert observed["workers"] == 0


@pytest.mark.parametrize("fault", ["page", "external"])
def test_method_comparison_actual_runtime_custody_failure_cannot_pass(
    program_bundle: Path,
    fault: str,
) -> None:
    """Actual page faults or unowned requests stay failures with retained diagnostics."""
    partial: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_method_comparison_journey(preview, evidence=partial)
    assert partial["page_errors"] or partial["external_requests"]


def test_method_comparison_refuses_an_actual_submission_attempt(
    program_bundle: Path,
    tmp_path: Path,
) -> None:
    """A real page POST stays refused even when the original comparison UI otherwise works."""
    damaged = tmp_path / "submission-preview"
    shutil.copytree(program_bundle, damaged)
    index = damaged / "index.html"
    index.write_bytes(
        index.read_bytes()
        + b'<script>fetch("/declared-submission",{method:"POST"}).catch(()=>{});</script>'
    )
    partial: dict[str, object] = {}
    with owned_fault_host(damaged) as preview, pytest.raises(AssertionError):
        run_method_comparison_journey(preview, evidence=partial)
    assert partial["submission_requests"] == ["POST"]
    assert (program_bundle / "index.html").read_bytes() != index.read_bytes()
