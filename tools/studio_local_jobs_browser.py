# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — local job service through the built Studio and a real service
"""Run, disconnect from and reconnect to a real local job in a real browser."""

from __future__ import annotations

import json
import signal
import subprocess  # nosec B404 - fixed module of this repository, no shell
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


def run_local_jobs_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Start the service, run a job from the Run stage, reload and reconnect.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser/runtime identity, observed contracts, the service requests and
        the absence of outside requests and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network, service or runtime contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    service_requests: list[str] = []
    observed.update(
        scenario="local_job_service",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        service_requests=service_requests,
    )
    origin = urlsplit(url)
    with tempfile.TemporaryDirectory(prefix="studio-local-jobs-") as scratch:
        token_file = Path(scratch) / "token"
        service = subprocess.Popen(  # nosec B603 - fixed interpreter and module
            [
                sys.executable,
                "-m",
                "scpn_quantum_control.studio_workspace.jobs.service",
                "--root",
                str(Path(scratch) / "jobs"),
                "--port",
                "0",
                "--token-file",
                str(token_file),
                "--allow-origin",
                f"{origin.scheme}://{origin.netloc}",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        stdout = service.stdout
        assert stdout is not None
        try:
            service_url = json.loads(stdout.readline())["url"]
            token = token_file.read_text(encoding="utf-8")
            service_netloc = urlsplit(service_url).netloc
            with sync_playwright() as runtime:
                browser = runtime.chromium.launch()
                context = browser.new_context(service_workers="block")
                context.set_default_timeout(30000)
                try:

                    def bound(route: Route) -> None:
                        target = urlsplit(route.request.url)
                        if target.scheme == "http" and target.netloc == service_netloc:
                            service_requests.append(f"{route.request.method} {target.path}")
                            route.continue_()
                        elif target.scheme != "http" or target.netloc != origin.netloc:
                            external.append("outside-preview")
                            route.abort()
                        elif route.request.method != "GET":
                            external.append(route.request.method)
                            route.abort()
                        else:
                            route.continue_()

                    context.route("**/*", bound)
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url + "#/experiments/local-jobs", wait_until="networkidle")
                    panel = page.get_by_role("region", name="Local job service", exact=True)
                    expect(panel).to_be_focused()
                    observations.append("deep-link-focuses-the-panel")
                    panel.get_by_label("Service address").fill(service_url)
                    panel.get_by_label("Session token").fill(token)
                    panel.get_by_role("button", name="Run job").click()
                    status = panel.get_by_role("status", name="Local job")
                    expect(status).to_contain_text("succeeded")
                    attempt_id = panel.get_by_label("Attempt").input_value()
                    expect(panel.get_by_role("note", name="Sealed record")).to_contain_text(
                        "sha256:"
                    )
                    observations.append("real-handler-runs-to-a-sealed-record")

                    page.reload(wait_until="networkidle")
                    panel = page.get_by_role("region", name="Local job service", exact=True)
                    panel.get_by_label("Service address").fill(service_url)
                    panel.get_by_label("Session token").fill(token)
                    panel.get_by_label("Attempt").fill(attempt_id)
                    panel.get_by_role("button", name="Reconnect").click()
                    expect(panel.get_by_role("status", name="Local job")).to_contain_text(
                        "succeeded"
                    )
                    rows = panel.get_by_role("table", name="Local job events").get_by_role("row")
                    expect(rows).to_have_count(4)
                    observations.append("reconnect-replays-the-same-attempt")

                    panel.get_by_label("Job request").fill(
                        json.dumps(
                            {
                                "project": "p",
                                "revision": "r",
                                "idempotency_key": "provider",
                                "verb": "execute",
                                "action_id": "a",
                                "parameters": {},
                            }
                        )
                    )
                    panel.get_by_role("button", name="Run job").click()
                    expect(panel.get_by_role("alert", name="Local job refused")).to_contain_text(
                        "verb_not_local"
                    )
                    observations.append("provider-verb-refused-by-the-service")
                    assert not errors, errors
                    assert not external, external
                    assert any(item.startswith("POST /v1/jobs") for item in service_requests)
                    observed.update(browser=browser.version, attempt_id=attempt_id)
                finally:
                    context.close()
                    browser.close()
        finally:
            service.send_signal(signal.SIGTERM)
            service.wait(timeout=30)
            stdout.close()
    observed.update(service_exit=service.returncode)
    return observed
