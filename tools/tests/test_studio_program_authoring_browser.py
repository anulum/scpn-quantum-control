# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real supported-source journey and network custody
"""Exercise the original built editor and genuine source compiler artifact."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tools.studio_program_authoring_browser import run_program_authoring_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:4173/",
        "https://127.0.0.1:4173/",
        "http://127.0.0.1:0/",
        "http://user:password@127.0.0.1:4173/",
    ],
)
def test_authoring_refuses_unowned_preview_before_browser_creation(url: str) -> None:
    """Refuse an unsafe server without loading UI or allocating a browser.

    Parameters
    ----------
    url
        Proposed server that violates the shared loopback ownership boundary.

    """
    with pytest.raises(ValueError):
        run_program_authoring_journey(url)


@pytest.fixture
def program_bundle() -> Path:
    """Require the actual built bundle with its genuine source compiler.

    Returns
    -------
    Path
        Source-owned deployment artifact supplied by the test runner.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual built Studio bundle in STUDIO_PREVIEW_DIR")
    bundle = Path(supplied).resolve()
    assert (bundle / "deploy-manifest.json").is_file()
    assert (bundle / "wasm/scpn_quantum_studio_wasm_kernel.wasm").is_file()
    return bundle


def test_original_program_authoring_real_wasm_and_exact_export(program_bundle: Path) -> None:
    """Qualify original source, all native gates, readout and stale-result recovery.

    Parameters
    ----------
    program_bundle
        Actual production bundle and locked Rust/WASM compiler.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_program_authoring_journey(preview)
    assert observed["execution_status"] == "emitted_not_executed"
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    records = observed["records"]
    assert isinstance(records, list) and len(records) == 8


@pytest.mark.parametrize("fault", ["page", "external"])
def test_original_authoring_cannot_pass_after_runtime_custody_failure(
    program_bundle: Path, fault: str
) -> None:
    """Retain real partial evidence while refusing page errors or unowned requests.

    Parameters
    ----------
    program_bundle
        Original built bytes, retained unchanged by the owned fault server.
    fault
        Real page exception or an actual network request outside the preview.

    """
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_program_authoring_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_pending_native_request_is_disposed_after_editor_transport_failure(
    program_bundle: Path, tmp_path: Path
) -> None:
    """Close an actual held WASM request when its source control becomes unavailable.

    Parameters
    ----------
    program_bundle
        Original built bytes and genuine WASM, retained unchanged.
    tmp_path
        Owned temporary host that adds a DOM transport fault to the original HTML.

    """
    shutil.copytree(program_bundle, tmp_path, dirs_exist_ok=True)
    fault = """<script>
        let pending = false, attempts = 0;
        new MutationObserver(() => {
            const source = document.querySelector('textarea[aria-label="Program source"]');
            const compiling = [...document.querySelectorAll('button')].some(button => button.textContent === 'Compiling source…');
            if (compiling && !pending && source && source.value.includes('cx q[0],q[1];') && source.value.includes('if(c==2) rz(-0.7853981633974492)')) {
                attempts += 1;
                if (attempts === 2) source.readOnly = true;
            }
            pending = compiling;
        }).observe(document, {subtree:true, childList:true, characterData:true, attributes:true});
    </script>"""
    (tmp_path / "index.html").write_text((program_bundle / "index.html").read_text() + fault)
    evidence: dict[str, object] = {}
    from playwright.sync_api import TimeoutError as BrowserTimeout

    with owned_fault_host(tmp_path) as preview, pytest.raises(BrowserTimeout) as refusal:
        run_program_authoring_journey(preview, evidence)
    completed = evidence["observations"]
    assert isinstance(completed, list)
    assert "structured-phase-condition-and-draft-invalidation" in completed, (
        str(refusal.value),
        evidence,
    )
    assert "delayed-real-wasm-result-cannot-revalidate-changed-draft" not in completed
