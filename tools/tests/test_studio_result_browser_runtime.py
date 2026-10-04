# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — genuine result inspector browser runtime
"""Exercise real result controls and retain genuine host or transport failures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.studio_result_browser import run_result_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def preview_directory() -> Path:
    """Require an actual original production bundle for native result acceptance.

    Returns
    -------
    Path
        Current real build with a native deploy manifest.

    Raises
    ------
    RuntimeError
        The owning runner has not provided its actual production build.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual production build in STUDIO_PREVIEW_DIR")
    directory = Path(supplied).resolve()
    assert (directory / "deploy-manifest.json").is_file()
    return directory


def test_actual_result_journey_raw_values_and_saved_custody() -> None:
    """Inspect actual CLI and WASM values, local downloads and unchanged saved state."""
    with owned_fault_host(preview_directory()) as origin:
        receipt = run_result_journey(origin)
    rows = receipt["observations"]
    assert isinstance(rows, list)
    assert {row["outcome"] for row in rows} == {
        "actual-native-raw-values-and-linked-final-coordinate",
        "actual-cli-nonuniform-coordinates-raw-int64-and-absent-intervals",
        "refused-import-and-complete-exports-retain-saved-source-and-worker-count",
    }


@pytest.mark.parametrize("fault", ["page", "external"])
def test_result_runtime_preserves_actual_fault_evidence(fault: str) -> None:
    """Retain native completed source observations while refusing genuine host faults.

    Parameters
    ----------
    fault
        Actual uncaught host error or refused external fetch.

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
        run_result_journey(origin, evidence=observed)
    assert observed["observations"]
