# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual local experiment browser runtime
"""Qualify real product controls and reject genuine page or network failures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tools.studio_local_experiment_browser import run_local_experiment_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def preview_directory() -> Path:
    """Require actual original production artifacts rather than a synthetic page.

    Returns
    -------
    Path
        Caller-supplied real current production bundle.

    Raises
    ------
    RuntimeError
        No actual build has been supplied.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the real built Studio directory in STUDIO_PREVIEW_DIR")
    directory = Path(supplied).resolve()
    assert (directory / "deploy-manifest.json").is_file()
    return directory


def test_local_experiment_journey_all_original_cases() -> None:
    """Run every original case with actual Chromium, IndexedDB and native WASM."""
    with owned_fault_host(preview_directory()) as origin:
        result = run_local_experiment_journey(origin)
    rows = result["observations"]
    assert isinstance(rows, list)
    outcomes = {row["outcome"] for row in rows}
    assert {f"test_local_experiment_journey_0{index}" for index in range(1, 5)} <= outcomes
    assert "native-worker-failure-and-tamper-refusal-retain-saved-source" in outcomes
    assert "missing-native-kernel-refuses-without-draft-or-worker" in outcomes


@pytest.mark.parametrize("fault", ["page", "external"])
def test_experiment_runtime_refuses_real_host_fault(fault: str) -> None:
    """Retain actual native partial evidence while refusing a genuine runtime fault.

    Parameters
    ----------
    fault
        Actual uncaught page exception or refused external request.

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
        run_local_experiment_journey(origin, evidence=observed)
    assert observed["observations"]


def test_blocked_worker_entry_is_closed_after_an_actual_host_control_failure(
    tmp_path: Path,
) -> None:
    """Close the actual held worker transport when host JavaScript removes its cancel control.

    Parameters
    ----------
    tmp_path
        Task-owned copy of the genuine production build for the negative host input.

    """
    from playwright.sync_api import TimeoutError

    original = preview_directory()
    damaged = tmp_path / "host-control-fault"
    shutil.copytree(original, damaged)
    index = damaged / "index.html"
    content = index.read_text(encoding="utf-8")
    interference = """<script>
new MutationObserver(() => {
  for (const button of document.querySelectorAll('button')) {
    if (button.textContent === 'Cancel experiment' && !button.disabled) button.remove();
  }
}).observe(document.documentElement, {subtree:true, childList:true, attributes:true});
</script>"""
    index.write_text(content.replace("</head>", interference + "</head>"), encoding="utf-8")
    observed: dict[str, object] = {}
    with (
        owned_fault_host(damaged) as origin,
        pytest.raises(TimeoutError, match="Cancel experiment"),
    ):
        run_local_experiment_journey(origin, evidence=observed)
    assert original.joinpath("index.html").read_text(encoding="utf-8") == content
    assert observed["observations"]
