# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — genuine immutable comparison browser runtime
"""Exercise native comparison and retain genuine source/transport/disposal failures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

import pytest

from scpn_quantum_control.studio_workspace import parse_document, read_json, write_json
from tools.studio_comparison_browser import incompatible_comparison_archive, run_comparison_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def preview_directory() -> Path:
    """Require the actual current original production bundle.

    Returns
    -------
    Path
        Current original production preview with a native deploy manifest.

    Raises
    ------
    RuntimeError
        The owning runner has not supplied the real production build.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual production build in STUDIO_PREVIEW_DIR")
    path = Path(supplied).resolve()
    assert (path / "deploy-manifest.json").is_file()
    return path


def test_actual_comparison_custody_and_unmatched_times() -> None:
    """Compare actual native WASM runs and retain all unmatched values and original archives."""
    with owned_fault_host(preview_directory()) as origin:
        receipt = run_comparison_journey(origin)
    observations = cast(list[dict[str, object]], receipt["observations"])
    assert {row["outcome"] for row in observations} == {
        "actual-immutable-runs-exact-time-unmatched-and-independent-deltas",
        "declared-unit-shots-backend-precision-block-arithmetic-without-save",
        "compare-back-reopen-reload-retains-exact-archive-and-original-raw-members",
    }
    matched = observations[0]
    assert (
        matched["matched"],
        matched["baseline_only"],
        matched["candidate_only"],
        matched["raw_rows"],
    ) == (151, 162, 162, 475)
    original = cast(str, receipt["original_saved_archive_json"])
    source = cast(dict[str, object], read_json(original))
    original_raw = [
        row for row in cast(list[dict[str, object]], source["members"]) if row["kind"] == "raw"
    ]
    for meaning in ("unit", "shots", "backend", "precision"):
        negative = cast(
            dict[str, object],
            read_json(incompatible_comparison_archive(original, meaning)),
        )
        raw = [
            row
            for row in cast(list[dict[str, object]], negative["members"])
            if row["kind"] == "raw"
        ]
        assert raw == original_raw
    with pytest.raises(ValueError, match="Explicit incompatible"):
        incompatible_comparison_archive(original, cast(Literal["unit"], "unknown"))
    wire = cast(dict[str, object], read_json(original))
    members = cast(list[dict[str, object]], wire["members"])
    revision = next(row for row in members if row["schema"] == "experiment_revision.v1")
    modified = parse_document(read_json(cast(str, revision["content"]))).to_dict()
    cast(dict[str, object], modified["extensions"])["negative_fixture"] = True
    child = parse_document(modified)
    members.append({**revision, "sha256": child.digest, "content": write_json(child.to_dict())})
    with pytest.raises(ValueError, match="One original immutable revision"):
        incompatible_comparison_archive(write_json(wire), "unit")


@pytest.mark.parametrize("fault", ["page", "external"])
def test_comparison_retains_genuine_host_fault(fault: str) -> None:
    """Refuse a real page or external transport failure after preserving completed observations.

    Parameters
    ----------
    fault
        Actual host/page boundary to break in the original build.

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
        run_comparison_journey(origin, evidence=observed)
    assert observed["observations"]


def test_comparison_requires_actual_native_disposal() -> None:
    """Withholding actual native termination still refuses completion and closes the context."""
    with (
        owned_fault_host(preview_directory(), worker_termination_delay_ms=None) as origin,
        pytest.raises(AssertionError, match="Studio native worker disposal"),
    ):
        run_comparison_journey(origin)


def test_comparison_observes_delayed_native_disposal() -> None:
    """A genuinely delayed native close must complete without treating its request counter as proof."""
    with owned_fault_host(preview_directory(), worker_termination_delay_ms=1000) as origin:
        receipt = run_comparison_journey(origin)
    assert receipt["observations"]
