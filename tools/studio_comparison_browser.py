# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original immutable comparison browser acceptance
"""Compare genuine archived WASM runs through the original production controls."""

from __future__ import annotations

import hashlib
import re
from importlib.metadata import version
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

from tools.studio_owned_worker_journey import _OBSERVE_WORKERS, wait_for_worker_disposal

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Locator, Page, Route

    from scpn_quantum_control.studio_workspace import WorkspaceDocument

_Meaning = Literal["unit", "shots", "backend", "precision"]
_MEANINGS: tuple[_Meaning, ...] = ("unit", "shots", "backend", "precision")
_SETTING_VALUES = {"shots": "4096", "backend": "unqualified-backend", "precision": "float32"}
_COMPARE = "Compare selected immutable sources"


def _redeclare_unit(
    documents: dict[str, WorkspaceDocument],
    body: dict[str, object],
    archive: dict[str, object],
    changed: dict[str, dict[str, object]],
) -> None:
    """Declare ``theta0`` in degrees and point the revision at the new declaration.

    Parameters
    ----------
    documents
        Saved documents by their original digest.
    body
        Body of the revision being re-addressed, changed in place.
    archive
        Archive whose declared parameter units change in place.
    changed
        Replacement documents by the digest they replace, extended in place.

    """
    from scpn_quantum_control.studio_workspace import parse_document

    specification = next(
        document
        for document in documents.values()
        if document.schema == "parameter_spec.v1" and document.body["key"] == "theta0"
    )
    replacement = specification.to_dict()
    cast(dict[str, object], replacement["body"])["unit"] = "deg"
    replacement_document = parse_document(replacement)
    changed[specification.digest] = replacement_document.to_dict()
    body["input_refs"] = [
        {**reference, "sha256": replacement_document.digest}
        if reference["sha256"] == specification.digest
        else reference
        for reference in cast(list[dict[str, object]], body["input_refs"])
    ]
    cast(dict[str, object], archive["parameter_units"])["theta0"] = "deg"


def _redeclare_setting(
    documents: dict[str, WorkspaceDocument],
    body: dict[str, object],
    meaning: str,
    changed: dict[str, dict[str, object]],
) -> None:
    """Declare one incompatible semantic setting and point the revision at it.

    Parameters
    ----------
    documents
        Saved documents by their original digest.
    body
        Body of the revision being re-addressed, changed in place.
    meaning
        Shots, backend or precision.
    changed
        Replacement documents by the digest they replace, extended in place.

    """
    from scpn_quantum_control.studio_workspace import parse_document

    reference = cast(dict[str, object], body["semantic_settings_ref"])
    settings = documents[cast(str, reference["sha256"])]
    replacement = settings.to_dict()
    settings_body = cast(dict[str, object], replacement["body"])
    for kind in ("requested", "effective", "origins"):
        cast(dict[str, object], settings_body[kind])[meaning] = (
            "deliberately incompatible negative fixture"
            if kind == "origins"
            else _SETTING_VALUES[meaning]
        )
    replacement_document = parse_document(replacement)
    changed[settings.digest] = replacement_document.to_dict()
    body["semantic_settings_ref"] = {**reference, "sha256": replacement_document.digest}


def _readdress(
    archive: dict[str, object],
    documents: dict[str, WorkspaceDocument],
    changed: dict[str, dict[str, object]],
    revision: str,
) -> None:
    """Rebind run records to the new revision and re-address members and manifest.

    Parameters
    ----------
    archive
        Archive re-addressed in place.
    documents
        Saved documents by their original digest.
    changed
        Replacement documents by the digest they replace, extended in place.
    revision
        Digest of the replacement revision.

    """
    from scpn_quantum_control.studio_workspace import parse_document, write_json

    for document in documents.values():
        if document.schema == "local_run_record.v1":
            replacement = document.to_dict()
            cast(dict[str, object], replacement["body"])["revision_hash"] = revision
            changed[document.digest] = parse_document(replacement).to_dict()
    identities = {old: parse_document(document).digest for old, document in changed.items()}
    archive["members"] = [
        {
            **member,
            "sha256": identities[member["sha256"]],
            "name": f"documents/{identities[member['sha256']]}.json",
            "content": write_json(changed[member["sha256"]]),
        }
        if member["sha256"] in changed
        else member
        for member in cast(list[dict[str, str]], archive["members"])
    ]
    manifest = cast(dict[str, object], archive["manifest"])
    manifest_body = cast(dict[str, object], manifest["body"])
    for field in ("revision_refs", "artefact_refs"):
        manifest_body[field] = [
            {**reference, "sha256": identities.get(reference["sha256"], reference["sha256"])}
            for reference in cast(list[dict[str, str]], manifest_body[field])
        ]
    selected = cast(dict[str, object], manifest_body["draft_ref"])
    manifest_body["draft_ref"] = {**selected, "sha256": revision}
    archive["manifest"] = parse_document(manifest).to_dict()


def incompatible_comparison_archive(original: str, meaning: _Meaning) -> str:
    """Re-address an explicitly negative declaration without altering original raw bytes.

    Parameters
    ----------
    original
        Actual first saved original archive containing one immutable revision.
    meaning
        One declared meaning to make incompatible, without unit conversion or execution.

    Returns
    -------
    str
        Separate negative fixture whose metadata can be inspected but whose
        original recorded numerical output must not be rebound to the new declaration.

    Raises
    ------
    ValueError
        The input is not a supported single-revision saved source or the requested
        declaration is unknown. Original source text and raw members are unchanged.

    """
    from scpn_quantum_control.studio_workspace import parse_document, read_json, write_json

    if meaning not in _MEANINGS:
        raise ValueError("Explicit incompatible comparison meaning required")
    archive = cast(dict[str, object], read_json(original))
    documents = {
        cast(str, member["sha256"]): parse_document(read_json(cast(str, member["content"])))
        for member in cast(list[dict[str, object]], archive["members"])
        if member["kind"] == "document"
    }
    revisions = [
        document for document in documents.values() if document.schema == "experiment_revision.v1"
    ]
    if len(revisions) != 1:
        raise ValueError("One original immutable revision required for the negative fixture")
    changed: dict[str, dict[str, object]] = {}
    wire = revisions[0].to_dict()
    body = cast(dict[str, object], wire["body"])
    if meaning == "unit":
        _redeclare_unit(documents, body, archive, changed)
    else:
        _redeclare_setting(documents, body, meaning, changed)
    replacement_revision = parse_document(wire)
    changed[revisions[0].digest] = replacement_revision.to_dict()
    _readdress(archive, documents, changed, replacement_revision.digest)
    return write_json(archive)


def _navigate(page: Page, name: str) -> None:
    """Use the original workbench navigation to change the visible route.

    Parameters
    ----------
    page
        Actual owned Studio page.
    name
        Existing accessible route label.

    """
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name=name, exact=True
    ).click()


def _archive(page: Page) -> str:
    """Read exact current archive text without changing the original editor.

    Parameters
    ----------
    page
        Original continuously mounted Workbench.

    Returns
    -------
    str
        Exact unchanged current portable source.

    """
    return page.get_by_label("Workspace archive JSON").input_value()


def _run_attempt(page: Page) -> None:
    """Run and save the explicitly selected original numerical source.

    Parameters
    ----------
    page
        Owned page on the original Experiments route.

    Raises
    ------
    AssertionError
        A genuine original run, native disposal or exact archive save fails.

    """
    from playwright.sync_api import expect

    experiment = page.get_by_label("Local experiment", exact=True)
    experiment.get_by_role("button", name="Validate experiment archive").click()
    prepare = experiment.get_by_role("button", name="Prepare numerical plan")
    expect(prepare).to_be_enabled()
    prepare.click()
    run = experiment.get_by_role("button", name="Run experiment", exact=True)
    expect(run).to_be_enabled()
    run.click()
    expect(experiment.get_by_text("Experiment succeeded", exact=True)).to_be_visible()
    wait_for_worker_disposal(page)
    experiment.get_by_role("button", name="Save experiment attempt").click()
    expect(experiment.get_by_role("status", name="Experiment operation")).to_contain_text(
        "Exact experiment attempt committed"
    )


def _rows(panel: Locator) -> list[list[str]]:
    """Read every original row through bounded native table pagination.

    Parameters
    ----------
    panel
        Actual comparison panel after completed admission.

    Returns
    -------
    list[list[str]]
        Original object, time, status, baseline, candidate, difference and state.

    """
    from playwright.sync_api import expect

    rows: list[list[str]] = []
    table = panel.get_by_role("table", name="Original matched and unmatched values")
    advance = panel.get_by_role("button", name="Next comparison values")
    while True:
        rows.extend(
            row.locator("th,td").all_text_contents() for row in table.get_by_role("row").all()[1:]
        )
        if advance.is_disabled():
            break
        advance.click()
        expect(
            panel.get_by_text(re.compile(f"^Original observations {len(rows) + 1}–"))
        ).to_be_visible()
    return rows


def _double_dt(page: Page) -> None:
    """Double the saved ``dt`` value in model time and commit a parameter revision.

    Parameters
    ----------
    page
        Owned page on the Workspace route.

    """
    from playwright.sync_api import expect

    editor = page.get_by_label("Linked parameter editor", exact=True)
    dt_control = editor.get_by_role("button", name="dt[scalar]", exact=False)
    original_dt = dt_control.inner_text().partition(" = ")[2]
    dt_control.click()
    field = editor.get_by_label("Selected value", exact=True)
    expect(field).to_have_value(original_dt)
    expect(editor.get_by_label("Input unit", exact=True)).to_have_value("model-time")
    field.fill(str(float(original_dt) * 2))
    editor.get_by_role("button", name="Apply value", exact=True).click()
    editor.get_by_role("button", name="Save parameter revision").click()
    expect(
        page.get_by_role("region", name="Local workspace").get_by_role("status")
    ).to_contain_text("Parameter revision transaction committed")


def _save_two_attempts(page: Page, target: str, receipt: dict[str, object]) -> tuple[str, str]:
    """Run the Kuramoto sample, double ``dt``, run again and keep both archives.

    Parameters
    ----------
    page
        Owned page bound to one host.
    target
        Owned host URL.
    receipt
        Caller-owned evidence receiving both saved archives.

    Returns
    -------
    tuple[str, str]
        The first saved archive and the final saved archive.

    """
    page.goto(target + "#/experiments", wait_until="networkidle")
    page.get_by_label("Local experiment", exact=True).get_by_role(
        "button", name="Open Kuramoto sample"
    ).click()
    _run_attempt(page)
    original = _archive(page)
    receipt["original_saved_archive_json"] = original
    _navigate(page, "Workspace")
    _double_dt(page)
    _navigate(page, "Run")
    _run_attempt(page)
    saved = _archive(page)
    receipt["final_saved_archive_json"] = saved
    return original, saved


def _admit_both(page: Page) -> Locator:
    """Read both saved archives in Compare and select the first revision as baseline.

    Parameters
    ----------
    page
        Owned page after both attempts are saved.

    Returns
    -------
    Locator
        The comparison panel with both sides admitted.

    """
    from playwright.sync_api import expect

    _navigate(page, "Compare")
    wait_for_worker_disposal(page)
    panel = page.get_by_label("Immutable revision and run comparison", exact=True)
    for side in ("baseline", "candidate"):
        panel.get_by_role("button", name=f"Read {side} archive").click()
        expect(panel.get_by_label(side.title() + " revision", exact=True)).to_be_visible()
    baseline = panel.get_by_label("Baseline revision", exact=True)
    revisions = baseline.locator("option").evaluate_all(
        "options => options.map(option => option.value)"
    )
    assert isinstance(revisions, list) and len(revisions) == 2
    baseline.select_option(revisions[0])
    return panel


def _check_values(page: Page, panel: Locator, saved: str, target: str) -> dict[str, object]:
    """Compare the two runs without a kernel and check every independent difference.

    Parameters
    ----------
    page
        Owned page on the Compare route.
    panel
        The comparison panel with both sides admitted.
    saved
        Final saved archive, which must stay unchanged.
    target
        Owned host URL.

    Returns
    -------
    dict
        Observation with the matched, baseline-only and candidate-only counts.

    """
    from playwright.sync_api import expect

    before = page.evaluate("window.__ownedKernel.started")
    panel.get_by_role("button", name=_COMPARE).click()
    expect(panel.get_by_role("table", name="Original semantic differences")).to_contain_text(
        "parameters.dt"
    )
    expect(
        panel.get_by_role("table", name="Original matched and unmatched values")
    ).to_be_visible()
    values = _rows(panel)
    matched = [row for row in values if row[2] == "matched"]
    baseline_only = [row for row in values if row[2] == "baseline-only"]
    candidate_only = [row for row in values if row[2] == "candidate-only"]
    assert matched and baseline_only and candidate_only
    for row in matched:
        assert float(row[5]) == float(row[4]) - float(row[3]) and row[6] == "available"
    assert all(row[5] == "unavailable" for row in baseline_only + candidate_only)
    assert page.evaluate("window.__ownedKernel.started") == before
    assert _archive(page) == saved
    return {
        "outcome": "actual-immutable-runs-exact-time-unmatched-and-independent-deltas",
        "source": target,
        "matched": len(matched),
        "baseline_only": len(baseline_only),
        "candidate_only": len(candidate_only),
        "raw_rows": len(values),
    }


def _check_meaning_refusals(page: Page, panel: Locator, original: str, saved: str) -> None:
    """Block arithmetic for each incompatible declared meaning without saving.

    Parameters
    ----------
    page
        Owned page on the Compare route.
    panel
        The comparison panel.
    original
        First saved archive the negative fixtures derive from.
    saved
        Final saved archive, which must stay unchanged.

    """
    from playwright.sync_api import expect

    for meaning in _MEANINGS:
        panel.get_by_label("Candidate archive JSON", exact=True).fill(
            incompatible_comparison_archive(original, meaning)
        )
        panel.get_by_role("button", name="Read candidate archive").click()
        panel.get_by_role("button", name=_COMPARE).click()
        expect(panel.get_by_role("alert")).to_contain_text("Numerical differences blocked")
        expect(panel.get_by_role("table", name="Original semantic differences")).to_contain_text(
            "units.theta0" if meaning == "unit" else f"effective_settings.{meaning}"
        )
        assert _archive(page) == saved


def _check_retained(page: Page, panel: Locator, saved: str, target: str) -> dict[str, object]:
    """Refuse a malformed candidate, reload, reopen and find the exact archive kept.

    Parameters
    ----------
    page
        Owned page on the Compare route.
    panel
        The comparison panel.
    saved
        Final saved archive, which must stay unchanged.
    target
        Owned host URL.

    Returns
    -------
    dict
        Observation with the kept archive digest and the kernel start count.

    """
    from playwright.sync_api import expect

    panel.get_by_label("Candidate archive JSON", exact=True).fill("{")
    panel.get_by_role("button", name="Read candidate archive").click()
    expect(panel.get_by_role("status", name="Comparison status")).to_contain_text(
        "previous admitted comparison and saved workspace retained"
    )
    _navigate(page, "Workspace")
    page.get_by_role("button", name="Reload saved workspace").click()
    expect(
        page.get_by_role("region", name="Local workspace").get_by_role("status")
    ).to_contain_text("Restored exact saved workspace")
    assert _archive(page) == saved
    _navigate(page, "Compare")
    expect(page.get_by_label("Candidate archive JSON", exact=True)).to_have_value(saved)
    assert _archive(page) == saved
    wait_for_worker_disposal(page)
    return {
        "outcome": "compare-back-reopen-reload-retains-exact-archive-and-original-raw-members",
        "source": target,
        "archive_sha256": hashlib.sha256(saved.encode("utf-8")).hexdigest(),
        "started": page.evaluate("window.__ownedKernel.started"),
    }


def _owned_page(
    context: BrowserContext, target: str, refused: list[str], errors: list[str]
) -> Page:
    """Restrict a context to its exact owned host and open a page that records failures.

    Parameters
    ----------
    context
        Fresh browser context for one host.
    target
        Owned host URL.
    refused
        Caller-owned record of refused request URLs.
    errors
        Caller-owned record of uncaught page failures.

    Returns
    -------
    Page
        The owned page.

    """
    authority = urlsplit(target).netloc

    def bound_request(route: Route) -> None:
        request = urlsplit(route.request.url)
        if request.scheme != "http" or request.netloc != authority:
            refused.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    context.route("**/*", bound_request)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    return page


def _compare_on(
    browser: Browser,
    target: str,
    receipt: dict[str, object],
    observations: list[dict[str, object]],
) -> None:
    """Run the whole comparison journey on one owned host in its own context.

    Parameters
    ----------
    browser
        Launched browser.
    target
        Owned host URL.
    receipt
        Caller-owned evidence, also receiving this context's failure records.
    observations
        Caller-owned list that records each outcome once it holds.

    """
    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    context.add_init_script(_OBSERVE_WORKERS)
    rejected: list[str] = []
    errors: list[str] = []
    receipt.update(
        current_owned_context=target, current_page_errors=errors, current_refused_requests=rejected
    )
    try:
        page = _owned_page(context, target, rejected, errors)
        original, saved = _save_two_attempts(page, target, receipt)
        panel = _admit_both(page)
        observations.append(_check_values(page, panel, saved, target))
        _check_meaning_refusals(page, panel, original, saved)
        observations.append(
            {
                "outcome": "declared-unit-shots-backend-precision-block-arithmetic-without-save",
                "source": target,
            }
        )
        observations.append(_check_retained(page, panel, saved, target))
        assert not errors, errors
        assert not rejected, rejected
    finally:
        context.close()


def _owned_hosts(base_url: str, source_url: str | None) -> tuple[str, ...]:
    """Validate the built host and the optional distinct root source host.

    Parameters
    ----------
    base_url
        Actual original production build served by an owned literal loopback host.
    source_url
        Optional distinct owned root source preview of the same current frontend.

    Returns
    -------
    tuple[str, ...]
        The built host, then the source host when one is supplied.

    Raises
    ------
    ValueError
        A supplied host is external, ambiguous, shared or a nested source URL.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is None:
        return (url,)
    if urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc:
        raise ValueError("Comparison source requires a distinct owned root source host")
    return (url, source)


def run_comparison_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise genuine immutable revision/run comparison on original built and optional source hosts.

    Parameters
    ----------
    base_url
        Actual original production build served by an owned literal loopback host.
    source_url
        Optional distinct owned root source preview of the same current frontend.
    evidence
        Caller-owned destination retaining completed observations and runtime faults.

    Returns
    -------
    dict[str, object]
        Actual browser versions, all comparison outcomes, exact source identities
        and unchanged saved-data evidence. This is functional evidence, not device qualification.

    Raises
    ------
    ValueError
        A supplied host is external, ambiguous, shared or a nested source URL.
    AssertionError
        Original values, meaning refusals, native disposal, saved custody or host boundaries fail.

    """
    hosts = _owned_hosts(base_url, source_url)
    from playwright.sync_api import sync_playwright

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    receipt.update(scenario="immutable_run_comparison", observations=observations)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            for target in hosts:
                _compare_on(browser, target, receipt, observations)
        finally:
            browser.close()
    return receipt
