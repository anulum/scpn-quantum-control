# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real Atlas navigation runtime and no-submit custody
"""Exercise the genuine built Atlas stage and fail on actual runtime or network faults."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest

from scpn_quantum_control.atlas import atlas_snapshot_digest, parse_atlas_snapshot
from tools.studio_atlas_browser import LARGE_NODES, large_pin, run_atlas_journey
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
def test_atlas_graph_navigation_refuses_unowned_preview_before_browser_creation(url: str) -> None:
    """Reject unowned URLs before allocating a browser.

    Parameters
    ----------
    url
        Preview violating the original shared literal-loopback contract.

    """
    with pytest.raises(ValueError):
        run_atlas_journey(url)


def test_the_large_pin_is_a_valid_snapshot_with_its_recorded_digest(tmp_path: Path) -> None:
    """The generated pin passes the Python contract, links successors and carries its digest.

    Parameters
    ----------
    tmp_path
        Directory that receives the pin.

    """
    pin = json.loads(large_pin(tmp_path, nodes=3).read_text(encoding="utf-8"))
    snapshot = parse_atlas_snapshot(pin["snapshot"])
    assert len(snapshot.nodes) == 3
    assert {(edge.source[-1], edge.target[-1]) for edge in snapshot.edges} == {
        ("0", "1"),
        ("0", "2"),
        ("1", "2"),
        ("1", "0"),
        ("2", "0"),
        ("2", "1"),
    }
    assert pin["snapshot_digest"] == atlas_snapshot_digest(snapshot)
    assert LARGE_NODES == 10_000


def test_atlas_graph_navigation_through_the_built_atlas_stage(program_bundle: Path) -> None:
    """Navigate the committed snapshot, then a large pin in a worker, through the built stage.

    Parameters
    ----------
    program_bundle
        Genuine source-owned built Studio artifacts.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_atlas_journey(preview)
    assert observed["no_submit"] is True
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    assert observed["large_pin_nodes"] == LARGE_NODES
    assert observed["observations"] == [
        "committed-snapshot-admitted-on-the-main-thread",
        "graph-table-and-inspector-share-the-selection",
        "keyboard-moves-the-selection",
        "deep-link-restores-snapshot-and-filter",
        "semantic-zoom-changes-the-representation",
        "accessible-at-desktop-and-mobile",
        "controls-fit-at-200-percent-zoom",
        "large-pin-admission-cancelled-and-its-worker-closed",
        "large-pin-admitted-and-queried-in-a-worker",
        "worker-answers-the-next-query",
        "main-thread-stays-responsive-with-the-large-pin",
        "worker-disposed-on-leaving-the-atlas",
    ]


@pytest.mark.parametrize("fault", ["page", "external"])
def test_atlas_graph_navigation_cannot_pass_after_runtime_custody_failure(
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
        run_atlas_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_atlas_graph_navigation_refuses_an_actual_same_origin_submission(
    program_bundle: Path, tmp_path: Path
) -> None:
    """An actual POST cannot pass the no-submit browser boundary.

    The journey loads the page twice, once at the start and once to prove the deep link
    after a reload, and the altered page attempts one POST per load.

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
        run_atlas_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST", "POST"]
