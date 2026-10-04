# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real native policy browser runtime refusals
"""Exercise actual built policy UI/WASM and retain real network/runtime failures."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest

from tools.studio_operator_policy_browser import run_operator_policy_journey
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
def test_unowned_policy_preview_refused(url: str) -> None:
    """No unowned URL reaches a browser or native policy producer."""
    with pytest.raises(ValueError):
        run_operator_policy_journey(url)


def test_real_policy_verdicts_and_original_wasm(program_bundle: Path) -> None:
    """All actual native verdicts agree through the reachable built operator UI."""
    with owned_fault_host(program_bundle) as preview:
        observed = run_operator_policy_journey(preview)
    assert observed["no_submit"] is True
    assert observed["policy_storage_write_observations"] == []
    assert observed["workers"] == 0
    assert observed["page_errors"] == []
    assert observed["external_requests"] == []
    decisions = observed["native_decisions"]
    assert isinstance(decisions, list) and len(decisions) == 8


@pytest.mark.parametrize("fault", ["page", "external"])
def test_real_policy_runtime_fault_refuses(program_bundle: Path, fault: str) -> None:
    """Actual browser errors and unowned requests cannot produce passing evidence."""
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_operator_policy_journey(preview, evidence)
    assert evidence["observations"]
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]


def test_real_policy_same_origin_post_refuses(program_bundle: Path, tmp_path: Path) -> None:
    """A real attempted POST is blocked and retained before any external call."""
    preview_dir = copytree(program_bundle, tmp_path / "built")
    html = preview_dir / "index.html"
    html.write_bytes(
        html.read_bytes()
        + b'<script>fetch(location.href,{method:"POST",body:"owned-refusal"}).catch(()=>{});</script>'
    )
    evidence: dict[str, object] = {}
    with owned_fault_host(preview_dir) as preview, pytest.raises(AssertionError):
        run_operator_policy_journey(preview, evidence)
    assert evidence["submission_requests"] == ["POST"]


@pytest.mark.parametrize("fault", [False, True])
def test_public_policy_dispatch_retains_actual_verdict_and_partial_failure(
    program_bundle: Path,
    tmp_path: Path,
    fault: bool,
) -> None:
    """The original public runner owns the real scenario and preserves partial runtime failures."""
    from tools.studio_browser_journey import main

    output = tmp_path / "public-policy-journey.json"
    with owned_fault_host(program_bundle, page_error=fault) as preview:
        code = main(
            [
                "--scenario",
                "operator_policy_decisions",
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
        assert evidence["submission_requests"] == []
    else:
        assert evidence["no_submit"] is True
        assert len(evidence["native_decisions"]) == 8
