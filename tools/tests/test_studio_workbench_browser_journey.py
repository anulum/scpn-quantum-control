# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native workbench URL and runtime acceptance
"""Exercise the public workbench journey and its owned-server boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import pytest

from tools.studio_workbench_browser_journey import run_workbench_journey
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host


def assert_native_source_owners(
    records: object, cohort: Literal["workbench", "parameters"], minimum_captures: int
) -> Counter[str]:
    """Bind actual browser counters to every declared owner and its original bytes.

    Parameters
    ----------
    records : object
        Actual V8 records returned by the public native journey.
    cohort : Literal["workbench", "parameters"]
        Original public browser registry cohort required by the journey.
    minimum_captures : int
        Required completed capture intervals for every source owner.

    Returns
    -------
    Counter[str]
        Native record multiplicities, retaining legitimate repeated script identities.

    """
    root = Path(__file__).resolve().parents[2] / "studio-web"
    declared = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--input-type=module",
            "-e",
            'import {readBrowserCohorts} from "./quality/browserCohorts.ts"; '
            "process.stdout.write(JSON.stringify(await readBrowserCohorts(process.cwd())));",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=os.environ | {"NODE_OPTIONS": "--max-old-space-size=768"},
    )
    metadata: object = json.loads(declared.stdout)
    assert isinstance(metadata, dict)
    paths = metadata[cohort]
    assert isinstance(paths, list) and paths
    hashes: dict[str, str] = {}
    for path in paths:
        assert isinstance(path, str) and path.startswith("/src/")
        source = (root / path.lstrip("/")).resolve()
        assert source.is_relative_to(root) and source.is_file() and path not in hashes
        hashes[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert isinstance(records, list)
    owners: Counter[str] = Counter()
    for record in records:
        assert isinstance(record, dict)
        coverage, code = record["coverage"], record["code"]
        assert isinstance(coverage, dict) and isinstance(coverage["url"], str)
        path = urlsplit(coverage["url"]).path
        assert path in hashes and record["source_sha256"] == hashes[path]
        assert isinstance(code, str)
        assert record["code_sha256"] == hashlib.sha256(code.encode("utf-8")).hexdigest()
        owners[path] += 1
    assert set(owners) == set(hashes)
    assert all(count >= minimum_captures for count in owners.values())
    return owners


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
def test_workbench_refuses_unsafe_origins_without_browser_extras(
    preview: str, source: str
) -> None:
    """Reject unsafe or coincident servers through the public API before loading extras.

    Parameters
    ----------
    preview
        Built preview address requiring literal loopback ownership.
    source
        Independent root source address whose authority must be distinct.

    """
    with pytest.raises(ValueError):
        run_workbench_journey(preview, source)
    root = Path(__file__).resolve().parents[2]
    script = """import sys
sys.path.insert(0, sys.argv[1])
from tools.studio_workbench_browser_journey import run_workbench_journey
try:
    run_workbench_journey(sys.argv[2], sys.argv[3])
except ValueError:
    pass
else:
    raise AssertionError("Unsafe workbench origins admitted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root), preview, source],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def workbench_bundle() -> Path:
    """Require the actual built portal, WASM and independent federation host.

    Returns
    -------
    Path
        Owned production bundle with the genuine acceptance host under its own prefix.

    """
    supplied = os.environ.get("STUDIO_PREVIEW_DIR")
    if supplied is None:
        raise RuntimeError("Supply the actual Studio bundle in STUDIO_PREVIEW_DIR")
    bundle = Path(supplied).resolve()
    assert (bundle / "deploy-manifest.json").is_file()
    assert (bundle / "acceptance-host/browser-tests/federation.html").is_file()
    return bundle


@pytest.fixture
def workbench_source() -> str:
    """Require the independent original-source host owned by this native cohort.

    Returns
    -------
    str
        Explicit source URL whose ownership is checked by the public journey.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host in STUDIO_WORKSPACE_SOURCE_URL")
    return supplied


def test_workbench_real_public_navigation_and_federation(
    workbench_bundle: Path, workbench_source: str
) -> None:
    """Use the original API with independent built/source hosts and native counters.

    Parameters
    ----------
    workbench_bundle
        Genuine compiled portal and original Rust WASM kernels.
    workbench_source
        Original-source Vite origin for native owner capture.

    """
    with owned_fault_host(workbench_bundle) as preview:
        evidence = run_workbench_journey(preview, workbench_source)
    cases = evidence["observations"]
    assert isinstance(cases, list)
    assert [case["case"] for case in cases] == ["standalone", "embedded", "chunk-error", "source"]
    assert all(case["workers"] == 0 for case in cases)
    assert cases[0]["real_wasm"] is True and cases[1]["real_wasm"] is True
    assert cases[2]["draft_retained"] is True and cases[2]["real_chunk_status"] == 503
    assert cases[3]["history_and_reload"] is True
    native = evidence["native_v8_coverage"]
    assert_native_source_owners(native, "workbench", 1)


@pytest.mark.parametrize("fault", ["page", "external"])
def test_workbench_failure_retains_actual_runtime_diagnostics(
    workbench_bundle: Path, workbench_source: str, fault: str
) -> None:
    """Refuse an actual page exception or outside fetch after exercising the real UI.

    Parameters
    ----------
    workbench_bundle
        Genuine built bytes served without altering the repository.
    workbench_source
        Original-source server retained by the outer native cohort.
    fault
        Genuine uncaught page exception or an attempted outside-authority request.

    """
    observed: dict[str, object] = {}
    with (
        owned_fault_host(
            workbench_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_workbench_journey(preview, workbench_source, observed)
    cases = observed["observations"]
    assert isinstance(cases, list) and len(cases) == 1
    assert cases[0]["history_and_reload"] is True
    if fault == "page":
        assert any("owned page failure" in error for error in cases[0]["page_errors"])
    else:
        assert any("owned-refused-request" in url for url in cases[0]["external_requests"])


def test_workbench_source_failure_retains_counters_and_completed_cases(
    workbench_bundle: Path, workbench_source: str
) -> None:
    """Keep actual original source counters when the final real source page fails.

    Parameters
    ----------
    workbench_bundle
        Original built portal/WASM and genuine independent host.
    workbench_source
        Owned original source forwarded without changing its modules or mappings.

    """
    observed: dict[str, object] = {}
    with (
        owned_fault_host(workbench_bundle) as preview,
        owned_fault_host(workbench_bundle, source=workbench_source, page_error=True) as source,
        pytest.raises(AssertionError),
    ):
        run_workbench_journey(preview, source, observed)
    cases = observed["observations"]
    assert isinstance(cases, list) and len(cases) == 4
    assert all(case["workers"] == 0 for case in cases[:3])
    assert any("owned page failure" in error for error in cases[3]["page_errors"])
    records = observed["native_v8_coverage"]
    assert_native_source_owners(records, "workbench", 1)
