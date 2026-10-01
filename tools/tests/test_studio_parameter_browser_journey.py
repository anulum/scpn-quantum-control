# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native parameter journey acceptance
"""Exercise the original browser journey with real built and source surfaces."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tools.studio_parameter_browser_journey import run_parameter_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


@pytest.mark.parametrize(
    ("preview", "source"),
    [
        ("http://example.com:4173/", "http://127.0.0.1:4174/"),
        ("https://127.0.0.1:4173/", "http://127.0.0.1:4174/"),
        ("http://127.0.0.1:4173/", "http://example.com:4174/"),
        ("http://127.0.0.1:4173/", "http://127.0.0.1:4174/nested/"),
        ("http://127.0.0.1:4173/", "http://127.0.0.1:4173/"),
        ("http://127.0.0.1:4173/preview/", "http://127.0.0.1:4173/"),
    ],
)
def test_parameter_journey_refuses_unsafe_servers_before_browser_import(
    preview: str, source: str
) -> None:
    """Refuse unsafe ownership boundaries even when browser extras are unavailable.

    Parameters
    ----------
    preview
        Proposed built portal URL.
    source
        Proposed independent source URL.

    """
    with pytest.raises(ValueError):
        run_parameter_journey(preview, source)
    root = Path(__file__).resolve().parents[2]
    script = """import sys
sys.path.insert(0, sys.argv[1])
from tools.studio_parameter_browser_journey import run_parameter_journey
try:
    run_parameter_journey(sys.argv[2], sys.argv[3])
except ValueError:
    pass
else:
    raise AssertionError('Unsafe parameter origins admitted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root), preview, source],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def parameter_bundle() -> Path:
    """Require the actual newly built portal and original Rust WASM.

    Returns
    -------
    Path
        Owned compiled artifact directory with both genuine kernels.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual Studio bundle in STUDIO_PREVIEW_DIR")
    bundle = Path(supplied).resolve()
    assert (bundle / "index.html").is_file()
    assert (bundle / "wasm/scpn_quantum_studio_wasm_kernel.wasm").is_file()
    assert (bundle / "wasm/scpn_quantum_studio_program_ad_wasm.wasm").is_file()
    return bundle


@pytest.fixture
def parameter_source() -> str:
    """Require the separately owned production-source Vite server.

    Returns
    -------
    str
        Actual loopback source origin supplied by the native cohort owner.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host in STUDIO_WORKSPACE_SOURCE_URL")
    return supplied


def test_parameter_journey_real_signed_matrix_and_native_revision_recovery(
    parameter_bundle: Path, parameter_source: str
) -> None:
    """Cross the real UI, parser, native storage, reload and Python identity boundary.

    Parameters
    ----------
    parameter_bundle
        Original compiled portal with the real shipped Rust WASM kernels.
    parameter_source
        Independent source host exposing the original WorkspacePanel.

    """
    with owned_fault_host(parameter_bundle) as preview:
        observed = run_parameter_journey(preview, parameter_source)
    assert observed["observations"] == [
        "built-original-wasm-and-workspace-link",
        "signed-directed-graph-form-and-exact-undo",
        "invalid-values-domain-unit-refusal-and-explicit-conversion",
        "symmetric-edit-mask-native-save-and-python-digest-parity",
        "reload-mask-and-second-native-save-preserve-all-prior-results",
        "unicode-source-key-linked-graph-and-native-marker-resolution",
    ]
    assert observed["workers"] == 0
    assert observed["page_errors"] == [] and observed["external_requests"] == []
    assert observed["original_members"] == 12
    assert observed["first_child_members"] == 13
    assert observed["second_child_members"] == 14
    records = observed["native_v8_coverage"]
    identity = observed["python_child_digest"]
    assert isinstance(records, list) and len(records) == 16
    owners: Counter[str] = Counter()
    for record in records:
        assert isinstance(record, dict)
        coverage = record["coverage"]
        assert isinstance(coverage, dict) and isinstance(coverage["url"], str)
        owners[urlsplit(coverage["url"]).path] += 1
    assert owners == Counter(
        {
            "/src/shared/storage/workspaceStore.ts": 2,
            "/src/shared/storage/workspaceArchive.ts": 2,
            "/src/features/workspace/WorkspacePanel.tsx": 2,
            "/src/features/workspace/useWorkspace.ts": 2,
            "/src/features/parameters/parameterDraft.ts": 2,
            "/src/features/parameters/parameterRevision.ts": 2,
            "/src/features/parameters/ParameterEditor.tsx": 2,
            "/src/features/parameters/ParameterWorkspace.tsx": 2,
        }
    )
    assert isinstance(identity, str) and len(identity) == 64
    controller = observed["native_controller_refusal"]
    assert isinstance(controller, dict)
    assert all(
        controller[name] is True
        for name in (
            "staleSource",
            "cancelledBeforeTransaction",
            "crossProject",
            "newerDraftRetained",
            "concurrentCommitRetained",
            "explicitReload",
            "previewChildCommitted",
            "lateCommitRetained",
        )
    )


@pytest.mark.parametrize("fault", ["page", "external"])
def test_parameter_journey_retains_actual_runtime_faults(
    parameter_bundle: Path, parameter_source: str, fault: str
) -> None:
    """Retain completed source cases while refusing real page or network faults.

    Parameters
    ----------
    parameter_bundle
        Actual built portal and unchanged original WASM kernels.
    parameter_source
        Independent original-source host.
    fault
        Actual uncaught page error or external fetch injected by the transport host.

    """
    observed: dict[str, object] = {}
    with (
        owned_fault_host(
            parameter_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_parameter_journey(preview, parameter_source, observed)
    completed = observed["observations"]
    assert (
        isinstance(completed, list)
        and completed[-1] == "unicode-source-key-linked-graph-and-native-marker-resolution"
    )
    if fault == "page":
        errors = observed["page_errors"]
        assert isinstance(errors, list) and any("owned page failure" in error for error in errors)
    else:
        rejected = observed["external_requests"]
        assert isinstance(rejected, list) and any(
            "owned-refused-request" in url for url in rejected
        )
