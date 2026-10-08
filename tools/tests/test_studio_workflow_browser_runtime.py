# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual workflow browser runtime
"""Run original browser controls and native workers against the current built source."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast
from urllib.request import urlopen

import pytest
from playwright.sync_api import Error

from tools.studio_workflow_browser import run_workflow_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def test_actual_workflow_graph_grid_restart_parent_failure_and_cancellation() -> None:
    """Require every original workflow case through actual browser, IndexedDB and WASM."""
    configured = os.environ.get("STUDIO_PREVIEW_DIR")
    if configured is None:
        raise RuntimeError("Supply the current original production bundle in STUDIO_PREVIEW_DIR")
    preview = Path(configured).resolve()
    assert (preview / "deploy-manifest.json").is_file()
    with owned_fault_host(preview) as origin:
        result = run_workflow_journey(origin)
    assert result["status"] == "passed"
    rows = cast(list[dict[str, object]], result["observations"])
    assert {row["case"] for row in rows} == {
        "cycle",
        "port",
        "six_cells",
        "restart",
        "failed_parent",
        "cancelled",
    }


@contextmanager
def workflow_source_fault_host(preview: Path, source: str) -> Iterator[str]:
    """Forward actual source bytes and emit original page and foreign-request faults.

    Parameters
    ----------
    preview
        Current built bundle retained unchanged by the underlying owned host.
    source
        Current owned Vite origin containing the original source and real WASM.

    Yields
    ------
    str
        Owned origin whose HTML triggers real untrusted requests during cancellation.

    """
    with owned_fault_host(
        preview, source=source, page_error=True, external_request=True
    ) as upstream:

        class Handler(BaseHTTPRequestHandler):
            """Forward the original fault fixture over this owned authority."""

            def do_GET(self) -> None:
                """Return real upstream bytes with the negative request at its UI boundary."""
                with urlopen(upstream.rstrip("/") + self.path, timeout=10) as response:
                    content = response.read()
                    content_type = response.headers.get("Content-Type", "application/octet-stream")
                if content_type.startswith("text/html"):
                    original = b'<script>fetch("http://127.0.0.2:54321/owned-refused-request").catch(()=>{});</script>'
                    replacement = b"""<script>
document.addEventListener('click', event => {
  if (event.target instanceof HTMLButtonElement &&
      event.target.textContent === 'Run or resume workflow' &&
      document.querySelector('[aria-label="Workflow JSON"]')?.value.includes('cancel-original-workflow')) {
    fetch('http://127.0.0.2:54321/kernelWorker-refused-request').catch(()=>{});
  }
});
</script>"""
                    if original not in content:
                        raise RuntimeError("Original negative request fixture missing")
                    content = content.replace(original, original + replacement)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, format: str, *args: object) -> None:
                """Keep request logs out of runtime evidence.

                Parameters
                ----------
                format
                    Native HTTP request log format.
                args
                    Native HTTP log arguments.

                """

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_actual_workflow_missing_preview_retains_original_transport_refusal(
    tmp_path: Path,
) -> None:
    """Keep real missing-page diagnostics without inventing completed observations.

    Parameters
    ----------
    tmp_path
        Caller-owned Samsung empty preview.

    """
    evidence: dict[str, object] = {}
    with owned_fault_host(tmp_path) as origin, pytest.raises(Error):
        run_workflow_journey(origin, evidence=evidence)
    assert evidence["observations"] == []
    assert "original_error" in evidence
    assert "diagnostic_capture_error" in evidence
    assert evidence["page_errors"] == []


def test_actual_workflow_missing_source_retains_completed_production_cases(tmp_path: Path) -> None:
    """Retain genuine completed cases when a later source page and native capture refuse.

    Parameters
    ----------
    tmp_path
        Caller-owned Samsung empty source preview.

    """
    configured = os.environ.get("STUDIO_PREVIEW_DIR")
    if configured is None:
        raise RuntimeError("Supply the current original production bundle")
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(Path(configured).resolve()) as original,
        owned_fault_host(tmp_path) as missing,
        pytest.raises(Error),
    ):
        run_workflow_journey(original, source_url=missing, evidence=evidence)
    rows = cast(list[dict[str, object]], evidence["observations"])
    assert len(rows) == 6
    assert {row["case"] for row in rows} == {
        "cycle",
        "port",
        "six_cells",
        "restart",
        "failed_parent",
        "cancelled",
    }
    assert evidence["failed_context"] == missing.rstrip("/") + "/"
    assert "native_coverage_error" in evidence
    assert "diagnostic_capture_error" in evidence


def test_actual_workflow_source_faults_keep_real_counters_and_original_history() -> None:
    """Reject real page errors and foreign worker-style transport without discarding native counters."""
    configured = os.environ.get("STUDIO_PREVIEW_DIR")
    source = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if configured is None or source is None:
        raise RuntimeError("Supply the current original bundle and owned source origin")
    preview = Path(configured).resolve()
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(preview) as original,
        workflow_source_fault_host(preview, source) as faulty,
        pytest.raises(AssertionError),
    ):
        run_workflow_journey(original, source_url=faulty, evidence=evidence)
    rows = cast(list[dict[str, object]], evidence["observations"])
    assert len(rows) == 12
    assert evidence["page_errors"]
    assert evidence["refused_requests"] == [
        "http://127.0.0.2:54321/owned-refused-request",
        "http://127.0.0.2:54321/owned-refused-request",
        "http://127.0.0.2:54321/kernelWorker-refused-request",
    ]
    assert len(cast(list[object], evidence["native_v8_coverage"])) == 26
    assert "native_coverage_error" not in evidence
    assert "diagnostic_capture_error" not in evidence
    assert cast(dict[str, object], evidence["failed_native_worker_counts"])["active"] == 0
