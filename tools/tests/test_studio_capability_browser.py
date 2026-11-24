# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real capability list runtime and no-submit custody
"""Exercise the genuine built UI and fail on actual runtime or network faults."""

from __future__ import annotations

import re
from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_capability_browser import CAPABILITIES, run_capability_journey
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
def test_capability_resolver_refuses_unowned_preview_before_browser_creation(url: str) -> None:
    """Reject unowned URLs before allocating a browser.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_capability_journey(url)


def test_the_journey_names_every_capability_the_modules_declare() -> None:
    """The journey's capability set equals the capabilities the module sources declare.

    A module that declares a capability the journey does not name fails here, before any
    browser starts, instead of only in the late browser job.
    """
    modules = Path(__file__).resolve().parents[2] / "studio-web/src/modules"
    declared: set[str] = set()
    for index in sorted(modules.glob("*/index.tsx")):
        block = re.search(
            r"\n  capabilities: \[\n(.*?)\n  \],", index.read_text(encoding="utf-8"), re.S
        )
        if block is not None:
            declared.update(re.findall(r'\bid: "([^"]+)"', block.group(1)))
    assert declared == CAPABILITIES


def test_capability_resolver_through_the_built_evidence_stage(program_bundle: Path) -> None:
    """Read every capability state through the actual Workspace stage.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_capability_journey(preview)
    assert observed["no_submit"] is True
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    assert observed["observations"] == [
        "every-capability-listed-and-kernels-wait-for-the-check",
        "runtime-check-verifies-both-kernels",
        "accessible-at-desktop-and-mobile",
    ]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_capability_resolver_cannot_pass_after_runtime_custody_failure(
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
        run_capability_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_capability_resolver_refuses_an_actual_same_origin_submission(
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
        run_capability_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]
