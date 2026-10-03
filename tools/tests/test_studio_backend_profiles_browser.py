# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real backend profile runtime and saved-state custody
"""Exercise genuine built UI/WASM and fail on actual runtime/network faults."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_backend_profiles_browser import run_backend_profiles_journey
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
def test_profile_refuses_unowned_preview_before_browser_creation(url: str) -> None:
    """Reject unowned URLs before allocating a browser or loading profile metadata.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_backend_profiles_journey(url)


def test_original_backend_profiles_browser_and_wasm(program_bundle: Path) -> None:
    """Inspect native profiles through the actual operator route and original WASM.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts and WASM compiler.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_backend_profiles_journey(preview)
    assert observed["no_submit"] is True
    assert observed["workspace_writes"] == 0
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    observations = observed["observations"]
    assert isinstance(observations, list) and len(observations) == 8


@pytest.mark.parametrize("fault", ["page", "external"])
def test_profile_cannot_pass_after_runtime_custody_failure(
    program_bundle: Path, fault: str
) -> None:
    """Retain partial real evidence and close the owned browser after a fault.

    Parameters
    ----------
    program_bundle
        Actual production UI and original WASM, retained unchanged.
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
        run_backend_profiles_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_profile_refuses_an_actual_same_origin_submission(
    program_bundle: Path, tmp_path: Path
) -> None:
    """An actual POST cannot pass the no-submit browser boundary.

    Parameters
    ----------
    program_bundle
        Original genuine built UI and WASM.
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
        run_backend_profiles_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]
