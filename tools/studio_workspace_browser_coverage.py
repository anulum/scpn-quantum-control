# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — browser native coverage capture
"""Retain native browser script counters for qualified source coverage mapping."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import CDPSession, Page


def start_native_coverage(page: Page) -> CDPSession:
    """Collect actual native V8 counters without replacing storage behavior.

    Parameters
    ----------
    page
        Owned native browser case page.

    Returns
    -------
    CDPSession
        Page-scoped native profiler session.

    """
    session = page.context.new_cdp_session(page)
    session.send("Debugger.enable")
    session.send("Profiler.enable")
    session.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
    return session


def take_native_coverage(
    session: CDPSession,
    records: list[dict[str, object]] | None = None,
    *,
    include_panel: bool = False,
    include_workbench: bool = False,
    include_parameters: bool = False,
    include_experiments: bool = False,
    include_results: bool = False,
) -> list[dict[str, object]]:
    """Retain actual executed scripts and counters for source mapping.

    Parameters
    ----------
    session
        Running page-scoped native profiler.
    records
        Caller-owned destination preserving collected scripts if capture fails.
    include_panel
        Also require the original panel during a real damaged-source journey.
    include_workbench
        Require the original facade, catalogue and all eight navigation owners.
    include_parameters
        Require the linked parameter editor, binding, draft and revision owners.
    include_experiments
        Require the original Workbench and complete source-bound experiment owners.
    include_results
        Require original Results routing and all read-only source result owners.

    Returns
    -------
    list[dict[str, object]]
        Actual script code with its original native coverage counters.

    Raises
    ------
    AssertionError
        The current page omits a required original production owner.
    RuntimeError
        Capture and profiler stop both fail; the original capture is retained as cause.
    Exception
        The original capture or profiler-stop exception when only one operation fails.

    """
    owners = {
        "/src/shared/storage/workspaceStore.ts",
        "/src/shared/storage/workspaceArchive.ts",
        "/src/features/workspace/WorkspacePanel.tsx",
        "/src/features/workspace/useWorkspace.ts",
    }
    if include_panel:
        owners.add("/src/QuantumStudioPanel.tsx")
    if include_workbench:
        owners.update(
            {
                "/src/QuantumStudioPanel.tsx",
                "/src/features/catalogue/CapabilityCatalogue.tsx",
                "/src/app/Workbench.tsx",
                "/src/app/WorkbenchInspector.tsx",
                "/src/app/RouteBoundary.tsx",
                "/src/app/routing.ts",
                "/src/app/useWorkbenchRoute.ts",
                "/src/app/routes/BuildView.tsx",
                "/src/app/routes/ResultsView.tsx",
                "/src/app/routes/UnavailableView.tsx",
            }
        )
    if include_parameters:
        owners.update(
            {
                "/src/features/parameters/parameterDraft.ts",
                "/src/features/parameters/parameterRevision.ts",
                "/src/features/parameters/ParameterEditor.tsx",
                "/src/features/parameters/ParameterWorkspace.tsx",
            }
        )
    if include_experiments:
        owners.update(
            {
                "/src/app/Workbench.tsx",
                "/src/features/experiments/kuramotoArtifacts.ts",
                "/src/features/experiments/experimentArchive.ts",
                "/src/features/experiments/experimentPlan.ts",
                "/src/features/experiments/useExperimentRun.ts",
                "/src/features/experiments/ExperimentRunner.tsx",
            }
        )
    if include_results:
        owners.update(
            {
                "/src/app/Workbench.tsx",
                "/src/app/routes/ResultsView.tsx",
                "/src/features/results/resultModel.ts",
                "/src/features/results/resultSources.ts",
                "/src/features/results/resultExport.ts",
                "/src/features/results/ResultInspector.tsx",
                "/src/features/results/ResultLoader.tsx",
            }
        )
    root = Path(__file__).resolve().parents[1] / "studio-web"
    records = [] if records is None else records
    captured: set[str] = set()
    capture_error: Exception | None = None
    try:
        result = session.send("Profiler.takePreciseCoverage")
        for entry in result["result"]:
            if urlsplit(entry["url"]).path not in owners:
                continue
            code = session.send("Debugger.getScriptSource", {"scriptId": entry["scriptId"]})[
                "scriptSource"
            ]
            source = root / urlsplit(entry["url"]).path.lstrip("/")
            records.append(
                {
                    "coverage": entry,
                    "code": code,
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            captured.add(urlsplit(entry["url"]).path)
    except Exception as error:
        capture_error = error
        raise
    finally:
        try:
            session.send("Profiler.stopPreciseCoverage")
        except Exception as stop_error:
            if capture_error is None:
                raise
            raise RuntimeError(
                f"Native coverage capture failed ({type(capture_error).__name__}: {capture_error}); "
                f"profiler stop also failed ({type(stop_error).__name__}: {stop_error})"
            ) from capture_error
    assert captured == owners, f"Missing current-page native owners: {sorted(owners - captured)}"
    return records
