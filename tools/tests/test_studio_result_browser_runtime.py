# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — genuine result inspector browser runtime
"""Exercise real result controls and retain genuine host or transport failures."""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest
from playwright.sync_api import Error, TimeoutError

from tools.studio_result_browser import run_result_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def preview_directory() -> Path:
    """Require an actual original production bundle for native result acceptance.

    Returns
    -------
    Path
        Current real build with a native deploy manifest.

    Raises
    ------
    RuntimeError
        The owning runner has not provided its actual production build.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual production build in STUDIO_PREVIEW_DIR")
    directory = Path(supplied).resolve()
    assert (directory / "deploy-manifest.json").is_file()
    return directory


def test_actual_result_journey_raw_values_and_saved_custody() -> None:
    """Inspect actual CLI and WASM values, local downloads and unchanged saved state."""
    with owned_fault_host(preview_directory()) as origin:
        receipt = run_result_journey(origin)
    rows = receipt["observations"]
    assert isinstance(rows, list)
    assert {row["outcome"] for row in rows} == {
        "actual-native-raw-values-and-linked-final-coordinate",
        "actual-cli-nonuniform-coordinates-raw-int64-and-absent-intervals",
        "refused-import-and-complete-exports-retain-saved-source-and-worker-count",
    }


@pytest.mark.parametrize("fault", ["page", "external"])
def test_result_runtime_preserves_actual_fault_evidence(fault: str) -> None:
    """Retain native completed source observations while refusing genuine host faults.

    Parameters
    ----------
    fault : str
        ``"page"`` emits an uncaught host error; ``"external"`` requests a refused origin.

    """
    observed: dict[str, object] = {}
    with (
        owned_fault_host(
            preview_directory(), page_error=fault == "page", external_request=fault == "external"
        ) as origin,
        pytest.raises(
            AssertionError,
            match="owned page failure" if fault == "page" else "owned-refused-request",
        ),
    ):
        run_result_journey(origin, evidence=observed)
    assert observed["observations"]


def test_result_transport_failure_retains_original_error_and_actual_diagnostics() -> None:
    """Retain refused navigation independently of the native error page's diagnostics."""
    observed: dict[str, object] = {}
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{reserved.getsockname()[1]}/"
        with pytest.raises(Error, match="Page.goto.*ERR_CONNECTION_REFUSED"):
            run_result_journey(origin, evidence=observed)
    assert observed["observations"] == []
    assert "Page.goto" in str(observed["original_error"])
    assert "ERR_CONNECTION_REFUSED" in str(observed["original_traceback"])
    if "diagnostic_capture_error" in observed:
        assert any(
            operation in str(observed["diagnostic_capture_error"])
            for operation in ("Page.evaluate", "Locator.all_text_contents", "Locator.evaluate_all")
        )
    else:
        assert observed["failed_native_workers"] in (
            None,
            {"started": 0, "disposed": 0, "active": 0, "commands": [], "events": []},
        )
        assert isinstance(observed["failed_page_url"], str)
        assert observed["failed_page_labels"] == []
        assert observed["failed_page_textareas"] == []
    assert observed["failed_context"] == origin
    assert observed["actual_original_cli_source_sha256"]


def test_result_control_failure_retains_original_error_when_diagnostics_refuse(
    tmp_path: Path,
) -> None:
    """Preserve an actual missing-control timeout when a real diagnostic getter throws.

    Parameters
    ----------
    tmp_path
        Owned copy of the actual production bundle for explicit host interference.

    """
    original = preview_directory()
    damaged = tmp_path / "result-diagnostic-fault"
    shutil.copytree(original, damaged)
    index = damaged / "index.html"
    content = index.read_text(encoding="utf-8")
    interference = """<script>
Object.defineProperty(window, '__ownedKernel', {
  get() { throw new Error('owned result diagnostic refusal'); }
});
new MutationObserver(() => {
  for (const button of document.querySelectorAll('button')) {
    if (button.textContent === 'Open Kuramoto sample') button.remove();
  }
}).observe(document.documentElement, {subtree:true, childList:true});
</script>"""
    assert content.count("</head>") == 1
    index.write_text(content.replace("</head>", interference + "</head>"), encoding="utf-8")
    observed: dict[str, object] = {}
    with (
        owned_fault_host(damaged) as origin,
        pytest.raises(TimeoutError, match="Open Kuramoto sample"),
    ):
        run_result_journey(origin, evidence=observed)
    assert original.joinpath("index.html").read_text(encoding="utf-8") == content
    assert observed["observations"] == []
    assert "Open Kuramoto sample" in str(observed["original_error"])
    assert "Locator.click" in str(observed["original_traceback"])
    assert "owned result diagnostic refusal" not in str(observed["original_error"])
    assert "Page.evaluate" in str(observed["diagnostic_capture_error"])
    assert "owned result diagnostic refusal" in str(observed["diagnostic_capture_error"])
    assert observed["failed_context"] == origin
    assert observed["actual_original_cli_source_sha256"]
