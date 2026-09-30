# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned browser network failure acceptance
"""Refuse actual page/network failures while retaining original runtime evidence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest
from playwright.sync_api import Error

from tools.studio_browser_journey import loopback_url, run_catalogue_journey, run_evidence_journey
from tools.studio_workspace_browser_journey import run_workspace_journey


@contextmanager
def owned_fault_host(
    preview: Path,
    *,
    source: str | None = None,
    failing_import: int = 0,
    page_error: bool = False,
    external_request: bool = False,
    corrupt_kernel: bool = False,
) -> Iterator[str]:
    """Own a HTTP fixture serving real built bytes and explicit transport failures.

    Parameters
    ----------
    preview
        Original built artifact directory, retained unchanged.
    source
        Optional owned Vite origin forwarding original native source bytes.
    failing_import
        One-based native helper request to refuse with HTTP 503; zero disables.
    page_error
        Emit a genuine uncaught page error after loading the original HTML.
    external_request
        Attempt an HTTP fetch outside the journey's owned authority.
    corrupt_kernel
        Truncate the third actual program-AD response while its predecessor waits.

    Yields
    ------
    str
        Bound loopback origin whose server and thread close at scope exit.

    """
    imports = 0
    kernels = 0

    class Handler(BaseHTTPRequestHandler):
        """Serve original artifacts without modifying repository files."""

        def do_GET(self) -> None:
            """Return built bytes, source bytes or the declared HTTP fault."""
            nonlocal imports, kernels
            if source is not None:
                if self.path.startswith("/browser-tests/workspaceStore.native.ts"):
                    imports += 1
                    if imports == failing_import:
                        self.send_error(503, "Owned native import transport failure")
                        return
                with urlopen(source.rstrip("/") + self.path, timeout=10) as response:
                    content = response.read()
                    content_type = response.headers.get("Content-Type", "application/octet-stream")
            else:
                relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
                path = (preview / relative).resolve()
                if not path.is_relative_to(preview.resolve()) or not path.is_file():
                    self.send_error(404)
                    return
                content = path.read_bytes()
                content_type = (
                    "text/html"
                    if relative.endswith(".html")
                    else "application/javascript"
                    if relative.endswith(".js")
                    else "application/wasm"
                    if relative.endswith(".wasm")
                    else "application/octet-stream"
                )
                if relative.endswith("scpn_quantum_studio_program_ad_wasm.wasm"):
                    kernels += 1
                    if corrupt_kernel and kernels == 3:
                        content = b"owned transport returned a truncated real kernel"
            if content_type.startswith("text/html"):
                if page_error:
                    content += b'<script>setTimeout(()=>{throw new Error("owned page failure")},0);</script>'
                if external_request:
                    content += b'<script>fetch("http://127.0.0.2:54321/owned-refused-request").catch(()=>{});</script>'
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            """Keep request logging out of test output.

            Parameters
            ----------
            format
                Original HTTP request log format.
            args
                Original request log arguments.

            """

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


@pytest.fixture
def preview_directory() -> Path:
    """Require the actual owned build rather than substitute UI fixtures.

    Returns
    -------
    Path
        Existing production build containing its deployment manifest.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual built Studio directory in STUDIO_PREVIEW_DIR")
    directory = Path(supplied).resolve()
    assert (directory / "deploy-manifest.json").is_file()
    return directory


@pytest.mark.parametrize("scenario", ["catalogue", "evidence"])
@pytest.mark.parametrize("fault", ["page", "external"])
def test_original_journeys_refuse_page_and_external_request_failures(
    preview_directory: Path, scenario: str, fault: str
) -> None:
    """A genuine loaded UI must not pass after an observed runtime failure.

    Parameters
    ----------
    preview_directory
        Actual built Studio artifacts.
    scenario
        Original catalogue or evidence public journey.
    fault
        Page exception or request to an unowned authority.

    """
    with owned_fault_host(
        preview_directory, page_error=fault == "page", external_request=fault == "external"
    ) as origin:
        journey = run_catalogue_journey if scenario == "catalogue" else run_evidence_journey
        with pytest.raises(
            AssertionError, match="owned page failure" if fault == "page" else "outside-preview"
        ):
            journey(origin)


def test_evidence_failure_releases_held_real_kernel_request(preview_directory: Path) -> None:
    """A failed second recomputation closes the pending original request and context.

    Parameters
    ----------
    preview_directory
        Original production build with real program-AD WASM.

    """
    with (
        owned_fault_host(preview_directory, corrupt_kernel=True) as origin,
        pytest.raises(AssertionError),
    ):
        run_evidence_journey(origin)


@pytest.mark.parametrize(
    ("failing_import", "field"), [(1, "native_failure"), (2, "fresh_import_failure")]
)
def test_workspace_import_network_failure_preserves_partial_evidence(
    preview_directory: Path, failing_import: int, field: str
) -> None:
    """Preserve actual native counters and diagnostics after a real HTTP 503.

    Parameters
    ----------
    preview_directory
        Actual production build retaining native UI acceptance.
    failing_import
        First or fresh-context native import request to refuse.
    field
        Original partial-evidence diagnostic field required after failure.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host in STUDIO_WORKSPACE_SOURCE_URL")
    source = loopback_url(supplied)
    observed: dict[str, object] = {}
    with (
        owned_fault_host(preview_directory) as preview,
        owned_fault_host(
            preview_directory,
            source=source,
            failing_import=failing_import,
            page_error=True,
            external_request=True,
        ) as native,
        pytest.raises(Error),
    ):
        run_workspace_journey(preview, native, observed)
    assert field in observed
    assert "503" in str(observed[field]) or "Failed to fetch dynamically imported module" in str(
        observed[field]
    )
    assert observed["external_requests"]
    assert observed["page_errors"]
    assert observed["native_v8_coverage"]
