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
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from playwright.sync_api import CDPSession, Page


_native_script_sources: WeakKeyDictionary[CDPSession, dict[str, str]] = WeakKeyDictionary()
_native_script_listeners: WeakKeyDictionary[CDPSession, Callable[[dict[str, object]], None]] = (
    WeakKeyDictionary()
)


def _source_cohorts(root: Path) -> dict[str, set[str]]:
    """Read exact compiled-owner cohorts from the same TypeScript owner as qualification.

    Parameters
    ----------
    root : Path
        Original installed web workspace, whose registry is read without executing modules.

    Returns
    -------
    dict[str, set[str]]
        Validated source paths by the fixed native capture cohort.

    Raises
    ------
    ValueError
        The metadata is not the expected source-path protocol.
    subprocess.SubprocessError
        The bounded original Node metadata reader fails.

    """
    command = [
        "node",
        "--experimental-strip-types",
        "--input-type=module",
        "-e",
        'import {readBrowserCohorts} from "./quality/browserCohorts.ts"; '
        "process.stdout.write(JSON.stringify(await readBrowserCohorts(process.cwd())));",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=os.environ | {"NODE_OPTIONS": "--max-old-space-size=768"},
    )
    return _cohort_paths(json.loads(completed.stdout))


def _cohort_paths(value: object) -> dict[str, set[str]]:
    """Validate the original compiled-owner protocol from live or archived metadata."""
    names = {
        "compiled",
        "workspace",
        "panel",
        "parameters",
        "workbench",
        "experiments",
        "results",
        "workflows",
    }
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("Compiled browser source cohort protocol required")
    cohorts: dict[str, set[str]] = {}
    for name, paths in value.items():
        if not isinstance(paths, list) or any(
            not isinstance(path, str)
            or not path.startswith("/src/")
            or not path.endswith((".ts", ".tsx"))
            or ".." in path.split("/")
            or "?" in path
            or "#" in path
            for path in paths
        ):
            raise ValueError("Original browser source paths required")
        cohorts[name] = set(paths)
    return cohorts


def _retain_source(
    event: dict[str, object], sources: dict[str, str], read_source: Callable[[str], object]
) -> None:
    """Retain source text using identical live and archived script metadata rules."""
    identifier, url = event.get("scriptId"), event.get("url")
    if not isinstance(identifier, str) or not isinstance(url, str):
        return
    path = urlsplit(url).path
    if not path.startswith("/src/") or not path.endswith((".ts", ".tsx")):
        return
    code = read_source(identifier)
    if not isinstance(code, str):
        raise ValueError("Actual parsed debugger source text required")
    sources[identifier] = code


def verify_native_coverage(artifact: Path, cohort_snapshot: Path, root: Path) -> None:
    """Check archived native source metadata and hashes before counter conversion.

    Parameters
    ----------
    artifact : Path
        Original journey JSON containing native V8 records, including zero counters.
    cohort_snapshot : Path
        JSON emitted by the original TypeScript compiled-owner reader.
    root : Path
        Original web source root against which archived source hashes are checked.

    Raises
    ------
    ValueError
        Cohorts, script metadata, source text or source digests are invalid.

    Notes
    -----
    This verifies retained source identity without executing scripts or recalculating
    counters. Scenario admission, owner completeness and source-map qualification
    remain the responsibility of the original TypeScript coverage converter.

    """
    cohorts = _cohort_paths(json.loads(cohort_snapshot.read_text(encoding="utf-8")))
    owners = set().union(*cohorts.values())
    evidence = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or not isinstance(evidence.get("native_v8_coverage"), list):
        raise ValueError("Native coverage record list required")
    records = evidence["native_v8_coverage"]
    if not records:
        raise ValueError("Nonempty native coverage records required")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("coverage"), dict):
            raise ValueError("Native script record required")
        coverage = cast("dict[str, object]", record["coverage"])
        sources: dict[str, str] = {}
        archived_sources: dict[str, object] = {str(coverage.get("scriptId")): record.get("code")}
        _retain_source(coverage, sources, archived_sources.get)
        if not sources:
            raise ValueError("Original source script metadata required")
        code = next(iter(sources.values()))
        path = urlsplit(cast("str", coverage["url"])).path
        if path not in owners:
            raise ValueError("Script is outside the compiled browser cohorts")
        if record.get("code_sha256") != hashlib.sha256(code.encode("utf-8")).hexdigest():
            raise ValueError("Archived debugger source digest differs")
        source = root / path.lstrip("/")
        if record.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Archived original source digest differs")


def start_native_coverage(page: Page, *, session: CDPSession | None = None) -> CDPSession:
    """Collect actual native V8 counters without replacing storage behavior.

    Parameters
    ----------
    page : Page
        Owned native browser case page.
    session : CDPSession | None
        Existing session for this page after its previous interval stopped. Reuse
        the same page session between intervals instead of detaching it.

    Returns
    -------
    CDPSession
        Page-scoped native profiler session.

    """
    if session is None:
        session = page.context.new_cdp_session(page)
    else:
        session.send("Debugger.disable")
    sources: dict[str, str] = {}
    _native_script_sources[session] = sources

    def retain_source(event: dict[str, object]) -> None:
        """Keep actual debugger text while its parsed source script remains readable.

        Parameters
        ----------
        event : dict[str, object]
            Native debugger notification containing the script identity and URL.

        Raises
        ------
        ValueError
            The debugger returns a source value other than text.

        """
        _retain_source(
            event,
            sources,
            lambda identifier: session.send("Debugger.getScriptSource", {"scriptId": identifier})[
                "scriptSource"
            ],
        )

    session.on("Debugger.scriptParsed", retain_source)
    _native_script_listeners[session] = retain_source
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
    include_workflows: bool = False,
) -> list[dict[str, object]]:
    """Retain actual executed scripts and counters for source mapping.

    Parameters
    ----------
    session : CDPSession
        Running page-scoped native profiler.
    records : list[dict[str, object]] | None
        Caller-owned destination preserving collected scripts if capture fails.
    include_panel : bool
        Also require the original panel during a real damaged-source journey.
    include_workbench : bool
        Require original navigation and all compiled generic shell owners.
    include_parameters : bool
        Require the linked parameter editor, binding, draft and revision owners.
    include_experiments : bool
        Require the original Workbench and complete source-bound experiment owners.
    include_results : bool
        Require original Results routing and all read-only source result owners.
    include_workflows : bool
        Require the original workflow owners and compiled graph job composition.

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
    root = Path(__file__).resolve().parents[1] / "studio-web"
    owners: set[str] = set()
    records = [] if records is None else records
    captured: set[str] = set()
    capture_error: Exception | None = None
    try:
        cohorts = _source_cohorts(root)
        owners.update(cohorts["workspace"])
        if (
            include_panel
            or include_workbench
            or include_experiments
            or include_results
            or include_workflows
        ):
            owners.discard("/src/WorkspacePanel.tsx")
        for enabled, name in [
            (include_panel, "panel"),
            (include_workbench, "workbench"),
            (include_parameters, "parameters"),
            (include_experiments, "experiments"),
            (include_results, "results"),
            (include_workflows, "workflows"),
        ]:
            if enabled:
                owners.update(cohorts[name])
        result = session.send("Profiler.takePreciseCoverage")
        for entry in result["result"]:
            if urlsplit(entry["url"]).path not in owners:
                continue
            # Reload may discard a context after its real native counters were recorded.
            # The cache contains only text obtained from this exact debugger session.
            code = _native_script_sources.get(session, {}).get(entry["scriptId"])
            if code is None:
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
        finally:
            _native_script_sources.pop(session, None)
            listener = _native_script_listeners.pop(session, None)
            if listener is not None:
                session.remove_listener("Debugger.scriptParsed", listener)
    assert captured == owners, f"Missing current-page native owners: {sorted(owners - captured)}"
    return records
