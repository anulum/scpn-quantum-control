# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — local job service browser journey
"""Run, disconnect from and reconnect to a real local job in a real browser."""

from __future__ import annotations

import json
import signal
import subprocess  # nosec B404 - fixed module of this repository, no shell
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page, Route

_PROVIDER_REQUEST = json.dumps(
    {
        "project": "p",
        "revision": "r",
        "idempotency_key": "provider",
        "verb": "execute",
        "action_id": "a",
        "parameters": {},
    }
)


@dataclass(frozen=True)
class _Service:
    """Address and session token of the started local job service."""

    url: str
    token: str

    def connect(self, panel: Locator) -> None:
        """Enter the service address and session token in the panel.

        Parameters
        ----------
        panel
            The local job service region.

        """
        panel.get_by_label("Service address").fill(self.url)
        panel.get_by_label("Session token").fill(self.token)


def _start_service(scratch: Path, origin: str) -> subprocess.Popen[str]:
    """Start the repository's local job service on a free loopback port.

    Parameters
    ----------
    scratch
        Owned temporary directory for the job root and the token file.
    origin
        Preview origin the service admits.

    Returns
    -------
    subprocess.Popen
        The running service, announcing its address on standard output.

    """
    return subprocess.Popen(  # nosec B603 - fixed interpreter and module
        [
            sys.executable,
            "-m",
            "scpn_quantum_control.studio_workspace.jobs.service",
            "--root",
            str(scratch / "jobs"),
            "--port",
            "0",
            "--token-file",
            str(scratch / "token"),
            "--allow-origin",
            origin,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


def _run_job(page: Page, url: str, service: _Service, observations: list[str]) -> str:
    """Run a job from the deep-linked panel to its sealed record.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    service
        The started local job service.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    str
        The attempt the job ran as.

    """
    from playwright.sync_api import expect

    page.goto(url + "#/experiments/local-jobs", wait_until="networkidle")
    panel = page.get_by_role("region", name="Local job service", exact=True)
    expect(panel).to_be_focused()
    observations.append("deep-link-focuses-the-panel")
    service.connect(panel)
    panel.get_by_role("button", name="Run job").click()
    expect(panel.get_by_role("status", name="Local job")).to_contain_text("succeeded")
    attempt_id = panel.get_by_label("Attempt").input_value()
    expect(panel.get_by_role("note", name="Sealed record")).to_contain_text("sha256:")
    observations.append("real-handler-runs-to-a-sealed-record")
    return attempt_id


def _reconnect_and_refuse(
    page: Page, service: _Service, attempt_id: str, observations: list[str]
) -> None:
    """Reload, replay the same attempt, then have the service refuse a provider verb.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    service
        The started local job service.
    attempt_id
        The attempt the job ran as.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    page.reload(wait_until="networkidle")
    panel = page.get_by_role("region", name="Local job service", exact=True)
    service.connect(panel)
    panel.get_by_label("Attempt").fill(attempt_id)
    panel.get_by_role("button", name="Reconnect").click()
    expect(panel.get_by_role("status", name="Local job")).to_contain_text("succeeded")
    rows = panel.get_by_role("table", name="Local job events").get_by_role("row")
    expect(rows).to_have_count(4)
    observations.append("reconnect-replays-the-same-attempt")
    panel.get_by_label("Job request").fill(_PROVIDER_REQUEST)
    panel.get_by_role("button", name="Run job").click()
    refused = panel.get_by_role("alert", name="Local job refused")
    expect(refused).to_contain_text("verb_not_local")
    observations.append("provider-verb-refused-by-the-service")


def _use_service(
    page: Page,
    url: str,
    observations: list[str],
    announced: str,
    token: str,
    service_requests: list[str],
) -> str:
    """Admit the service's own requests, then run, reconnect and refuse through it.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    announced
        First line the service printed, carrying its address.
    token
        Session token the service wrote.
    service_requests
        Caller-owned record of each request sent to the service.

    Returns
    -------
    str
        The attempt the job ran as.

    """
    service = _Service(json.loads(announced)["url"], token)
    target = urlsplit(service.url).netloc

    def to_service(route: Route) -> None:
        service_requests.append(f"{route.request.method} {urlsplit(route.request.url).path}")
        route.continue_()

    page.route(lambda address: urlsplit(address)[:2] == ("http", target), to_service)
    page.set_default_timeout(30000)
    attempt_id = _run_job(page, url, service, observations)
    _reconnect_and_refuse(page, service, attempt_id, observations)
    assert any(item.startswith("POST /v1/jobs") for item in service_requests)
    return attempt_id


def _walk(
    page: Page, url: str, observations: list[str], *, service_requests: list[str]
) -> dict[str, object]:
    """Start the local job service, use it from the Run stage and stop it.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    service_requests
        Caller-owned record of each request sent to the service.

    Returns
    -------
    dict
        The attempt the job ran as and the service exit status.

    """
    origin = urlsplit(url)
    with tempfile.TemporaryDirectory(prefix="studio-local-jobs-") as scratch:
        process = _start_service(Path(scratch), f"{origin.scheme}://{origin.netloc}")
        stdout = process.stdout
        assert stdout is not None
        try:
            announced = stdout.readline()
            token = (Path(scratch) / "token").read_text(encoding="utf-8")
            attempt_id = _use_service(page, url, observations, announced, token, service_requests)
        finally:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
            stdout.close()
    return {"attempt_id": attempt_id, "service_exit": process.returncode}


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
        the absence of outside requests, submissions and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network, service or runtime contract fails.

    """
    service_requests: list[str] = []
    observed = {} if evidence is None else evidence
    observed["service_requests"] = service_requests
    return run_custodied_journey(
        base_url,
        observed,
        "local_job_service",
        partial(_walk, service_requests=service_requests),
    )
