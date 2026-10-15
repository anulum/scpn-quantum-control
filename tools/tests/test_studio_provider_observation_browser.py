# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual existing-job browser boundary tests
"""Exercise original provider evidence through the built Studio and genuine WASM."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_provider_observation_browser import run_provider_observation_journey
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
def test_unowned_job_preview_refused(url: str) -> None:
    """Refuse an unowned preview before opening a browser.

    Parameters
    ----------
    url
        Proposed address outside the shared preview ownership contract.

    """
    with pytest.raises(ValueError):
        run_provider_observation_journey(url)


def test_real_job_observations_exact_export_and_original_wasm(program_bundle: Path) -> None:
    """All five native snapshots retain exact identity through the actual operator UI.

    Parameters
    ----------
    program_bundle
        Actual current production bundle with its genuine original WASM.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_provider_observation_journey(preview)
    assert observed["no_submit"] is True
    assert observed["job_storage_write_observations"] == []
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    assert observed["submission_requests"] == []
    assert observed["wasm_requests"]
    records = observed["native_observations"]
    assert isinstance(records, list) and len(records) == 5


@pytest.mark.parametrize("fault", ["page", "external"])
def test_real_job_runtime_fault_refuses(program_bundle: Path, fault: str) -> None:
    """Actual page/network failures cannot produce a successful job-inspection receipt.

    Parameters
    ----------
    program_bundle
        Original production artifacts retained by the owned fault server.
    fault
        Uncaught browser error or actual attempted unowned request.

    """
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_provider_observation_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_real_job_same_origin_post_refuses(program_bundle: Path, tmp_path: Path) -> None:
    """A real attempted POST is blocked and recorded before any provider access.

    Parameters
    ----------
    program_bundle
        Actual original bundle and kernels.
    tmp_path
        Owned copy receiving only a benign negative transport script.

    """
    preview_dir = copytree(program_bundle, tmp_path / "built")
    html = preview_dir / "index.html"
    html.write_bytes(
        html.read_bytes()
        + b'<script>fetch(location.href,{method:"POST",body:"owned-refusal"}).catch(()=>{});</script>'
    )
    evidence: dict[str, object] = {}
    with owned_fault_host(preview_dir) as preview, pytest.raises(AssertionError):
        run_provider_observation_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]


@pytest.mark.parametrize("fault", [False, True])
def test_public_job_dispatch_retains_original_evidence_and_partial_failure(
    program_bundle: Path, tmp_path: Path, fault: bool
) -> None:
    """The original shared runner dispatches the actual scenario and retains runtime refusals.

    Parameters
    ----------
    program_bundle
        Real built current source and original kernels.
    tmp_path
        Owned public dispatcher output path.
    fault
        Inject a real page failure while preserving the original product UI.

    """
    from tools.studio_browser_journey import main

    output = tmp_path / "public-job-journey.json"
    with owned_fault_host(program_bundle, page_error=fault) as preview:
        code = main(
            [
                "--scenario",
                "provider_observation_journal",
                "--base-url",
                preview,
                "--output",
                str(output),
            ]
        )
    evidence = json.loads(output.read_text())
    assert code == (1 if fault else 0)
    assert evidence["passed"] is (not fault)
    assert evidence["observations"]
    if fault:
        assert evidence["page_errors"] and "AssertionError" in evidence["error"]
    else:
        assert evidence["no_submit"] is True
        assert len(evidence["native_observations"]) == 5
