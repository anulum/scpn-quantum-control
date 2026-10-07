# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned worker browser refusal custody
"""Require the native worker journey to reject actual host and page failures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tools.studio_owned_worker_journey import run_owned_worker_journey
from tools.studio_resource_browser_journey import run_resource_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def preview_directory() -> Path:
    """Require the original deployed bundle with its verified worker manifest.

    Returns
    -------
    Path
        Actual built artifacts supplied by the caller.

    Raises
    ------
    RuntimeError
        The caller has not supplied the owned production build.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual built Studio directory in STUDIO_PREVIEW_DIR")
    directory = Path(supplied).resolve()
    assert (directory / "deploy-manifest.json").is_file()
    return directory


@pytest.mark.parametrize("fault", ["page", "external"])
def test_owned_journey_refuses_observed_page_and_network_faults(fault: str) -> None:
    """Preserve partial real-worker evidence when an actual page or request fails.

    Parameters
    ----------
    fault
        Uncaught page exception or a fetch outside the owned preview.

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
        run_owned_worker_journey(origin, evidence=observed)
    assert observed["observations"]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_owned_resource_journey_refuses_observed_host_faults(fault: str) -> None:
    """Keep the original admission journey subject to real host failure refusal.

    Parameters
    ----------
    fault
        Uncaught page exception or a fetch outside the owned preview.

    """
    with (
        owned_fault_host(
            preview_directory(), page_error=fault == "page", external_request=fault == "external"
        ) as origin,
        pytest.raises(
            AssertionError,
            match="owned page failure" if fault == "page" else "owned-refused-request",
        ),
    ):
        run_resource_journey(origin)


@contextmanager
def refused_termination_host(preview: Path) -> Iterator[str]:
    """Serve original artifacts with a genuine host disposal failure.

    Parameters
    ----------
    preview
        Original built directory, read without changing any saved bytes.

    Yields
    ------
    str
        Owned origin; the HTTP server and thread close after the browser fails.

    """

    class Handler(SimpleHTTPRequestHandler):
        """Retain the real UI and worker while refusing native host termination."""

        def do_GET(self) -> None:
            """Inject a host exception only after a real worker result exists."""
            if self.path == "/":
                content = (
                    (preview / "index.html").read_bytes()
                    + b"""<script>
                  const dispose = Worker.prototype.terminate;
                  const send = Worker.prototype.postMessage;
                  Worker.prototype.postMessage = function(message, transfers) {
                    this.ownedFaultRun = message.run_id;
                    return send.call(this, message, transfers);
                  };
                  Worker.prototype.terminate = function() {
                    const trace = window.__ownedKernel;
                    const completed = trace?.commands.some(command =>
                      command.n === 2 && command.steps === 8 && trace.events.some(event =>
                        event.kind === 'result' && event.run_id === command.run_id));
                    const result = trace?.events.some(event =>
                      event.kind === 'result' && event.run_id === this.ownedFaultRun);
                    if (completed && !result) {
                      throw new Error('owned host refused worker termination');
                    }
                    return dispose.call(this);
                  };
                </script>"""
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            """Keep original request logs out of the regression receipt.

            Parameters
            ----------
            format
                Native request format.
            args
                Native request arguments.

            """

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(preview)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_owned_journey_refuses_native_disposal_failure() -> None:
    """A real native worker that cannot terminate never earns journey acceptance."""
    observed: dict[str, object] = {}
    with refused_termination_host(preview_directory()) as origin, pytest.raises(AssertionError):
        run_owned_worker_journey(origin, evidence=observed)
    assert "real-two-node-original-WASM" in str(observed["observations"])


def test_owned_journey_waits_for_delayed_native_worker_closure() -> None:
    """Complete all genuine kernel boundaries despite asynchronous native closure."""
    with owned_fault_host(preview_directory(), worker_termination_delay_ms=1000) as origin:
        result = run_owned_worker_journey(origin)
    observations = result["observations"]
    assert isinstance(observations, list)
    outcomes = {row["outcome"] for row in observations}
    assert {
        boundary + "-disposed-no-stale-worker"
        for boundary in ("cancel", "replacement", "route", "project", "timeout")
    } <= outcomes


def test_owned_journey_refuses_native_worker_leak_after_observer_disposal() -> None:
    """A completed JavaScript disposal counter cannot certify a leaked native worker."""
    with (
        owned_fault_host(preview_directory(), worker_termination_delay_ms=None) as origin,
        pytest.raises(AssertionError),
    ):
        run_owned_worker_journey(origin)
