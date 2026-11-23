# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real Atlas selection to Studio draft runtime and no-submit custody
"""Exercise the built Atlas-to-Studio selection and fail on actual runtime or network faults."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_atlas_selection_browser import private_identifiers, run_atlas_selection_journey
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
def test_atlas_studio_selection_journey_refuses_unowned_preview_before_browser_creation(
    url: str,
) -> None:
    """Reject unowned URLs before allocating a browser.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_atlas_selection_journey(url)


def test_private_identifiers_name_the_project_and_the_selected_revisions() -> None:
    """The private set is the archive's project and every revision hash the text names."""
    project = "10000000-0000-4000-8000-000000000001"
    archive = f'{{"body": {{"project_id": "{project}"}}, "x": {{"project_id":"{project}"}}}}'
    text = f"revision:sha256:{'a' * 64} · revision:sha256:{'b' * 64} · revision:sha256:{'a' * 64}"

    assert private_identifiers(archive, text) == [project, "a" * 64, "b" * 64]
    assert private_identifiers("{}", "no selection") == []


def test_atlas_studio_selection_journey_through_the_built_atlas_stage(
    program_bundle: Path,
) -> None:
    """Select a revision in the Atlas, draft from it, save a child and come back.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_atlas_selection_journey(preview)
    assert observed["no_submit"] is True
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    assert observed["observations"] == [
        "atlas-selection-reaches-the-studio-inspector",
        "private-identity-absent-from-shared-urls",
        "draft-saves-a-child-and-keeps-the-revision",
        "atlas-to-studio-to-atlas-keeps-identity",
        "accessible-at-desktop-and-mobile",
    ]
    assert observed["child"] != observed["revision"]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_atlas_studio_selection_journey_cannot_pass_after_runtime_custody_failure(
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
        run_atlas_selection_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_atlas_studio_selection_journey_refuses_an_actual_same_origin_submission(
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
        run_atlas_selection_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]
