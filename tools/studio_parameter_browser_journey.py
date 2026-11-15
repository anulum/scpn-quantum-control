# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native linked parameter browser journey
"""Exercise linked parameter edits and immutable native workspace persistence."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page, Route

_NATIVE = "/browser-tests/parameterEditor.native.tsx"
_ARCHIVE = f"async corpus => (await import('{_NATIVE}')).createParameterConformanceArchive(corpus)"
_NAMED_ARCHIVE = (
    f"async corpus => (await import('{_NATIVE}'))"
    ".createParameterConformanceArchive(corpus, 'Coupling (A → B)')"
)
_CONTROLLER_CASES = (
    f"async corpus => (await import('{_NATIVE}')).runParameterControllerCases(corpus)"
)
_MARKER = (
    "path => ({id: path.ownerSVGElement.querySelector('marker').id, "
    "reference: path.getAttribute('marker-end'), computed: getComputedStyle(path).markerEnd})"
)
_SYMMETRIC_VALUES = [
    "0000000000000000",
    "c020000000000000",
    "0000000000000000",
    "c020000000000000",
    "0000000000000000",
    "4008000000000000",
    "c010000000000000",
    "0000000000000000",
    "0000000000000000",
]
_COMMITTED = "Parameter revision transaction committed"
_MEMBER_COUNTS = ("original_members", "first_child_members", "second_child_members")
_CORPUS = Path(__file__).resolve().parents[1] / "tests/data/studio_workspace/documents.json"


@dataclass
class _Lists:
    """Caller-visible evidence lists the journey fills."""

    completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    records: list[dict[str, object]] = field(default_factory=list)


def _edit(editor: Locator, value: str) -> None:
    """Apply decimal text through the actual selected-parameter form.

    Parameters
    ----------
    editor : Locator
        Original production editor region in the native browser.
    value : str
        Decimal text to validate against the original source specification.

    """
    editor.get_by_label("Selected value", exact=True).fill(value)
    editor.get_by_role("button", name="Apply value", exact=True).click()


def _element(editor: Locator, name: str) -> Locator:
    """Return one named matrix element or graph edge button.

    Parameters
    ----------
    editor : Locator
        Linked parameter editor.
    name : str
        Exact accessible name.

    Returns
    -------
    Locator
        The button.

    """
    return editor.get_by_role("button", name=name, exact=True)


def _check_built(built: Page, preview: str, completed: list[str]) -> None:
    """Find the built WASM simulator verified and follow its link to the workspace.

    Parameters
    ----------
    built : Page
        Owned page on the built preview.
    preview : str
        Owned built preview.
    completed : list[str]
        Caller-owned completed observations.

    """
    from playwright.sync_api import expect

    built.goto(preview, wait_until="networkidle")
    expect(built.get_by_role("img", name="order parameter over time")).to_be_visible()
    verified = built.get_by_text("verified against the committed ground truth", exact=False)
    expect(verified).to_be_visible()
    built.get_by_role("link", name="Edit source parameters in Workspace").click()
    expect(built).to_have_url(preview + "#/workspace")
    completed.append("built-original-wasm-and-workspace-link")


def _admit_archive(page: Page, source: str, corpus: str) -> tuple[Locator, str]:
    """Build the conformance archive natively, admit it and save it.

    Parameters
    ----------
    page : Page
        Owned page on the source host.
    source : str
        Owned root Vite source server.
    corpus : str
        Committed workspace document corpus.

    Returns
    -------
    tuple[Locator, str]
        The local workspace region and the admitted archive text.

    """
    from playwright.sync_api import expect

    page.goto(source + "browser-tests/parameterEditor.html", wait_until="networkidle")
    native = cast(dict[str, object], page.evaluate(_ARCHIVE, corpus))
    original_json = cast(str, native["json"])
    workspace = page.get_by_role("region", name="Local workspace")
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
    workspace.get_by_label("Workspace archive JSON").fill(original_json)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(workspace.get_by_role("status")).to_contain_text("Workspace transaction committed")
    return workspace, original_json


def _check_signed_edit(editor: Locator, completed: list[str]) -> None:
    """Edit a signed directed edge, then undo and redo it exactly.

    Parameters
    ----------
    editor : Locator
        Linked parameter editor.
    completed : list[str]
        Caller-owned completed observations.

    """
    from playwright.sync_api import expect

    expect(editor.get_by_role("img", name="K_nm coupling graph diagram")).to_be_visible()
    digest = editor.get_by_label("Draft semantic digest")
    expect(digest).to_have_text(re.compile("^[0-9a-f]{64}$"))
    initial_digest = digest.inner_text()
    _element(editor, "Edge 1 → 0: -2 rad/s").click()
    expect(editor.get_by_label("Selected value", exact=True)).to_have_value("-2")
    _edit(editor, "-7")
    expect(_element(editor, "K_nm[0,1] = -7")).to_be_visible()
    expect(_element(editor, "K_nm[1,0] = 5")).to_be_visible()
    expect(_element(editor, "Edge 1 → 0: -7 rad/s")).to_be_visible()
    expect(digest).not_to_have_text(initial_digest)
    editor.get_by_role("button", name="Undo parameter edit").click()
    expect(digest).to_have_text(initial_digest)
    editor.get_by_role("button", name="Redo parameter edit").click()
    expect(_element(editor, "K_nm[0,1] = -7")).to_be_visible()
    completed.append("signed-directed-graph-form-and-exact-undo")


def _check_refusals(
    editor: Locator, archive_input: Locator, original_json: str, completed: list[str]
) -> None:
    """Refuse non-finite and out-of-domain values and a unit mismatch; convert explicitly.

    Parameters
    ----------
    editor : Locator
        Linked parameter editor.
    archive_input : Locator
        Workspace archive editor, which must keep the admitted archive.
    original_json : str
        Admitted archive text.
    completed : list[str]
        Caller-owned completed observations.

    """
    from playwright.sync_api import expect

    for refused in ("NaN", "Infinity", "11"):
        _edit(editor, refused)
        expect(editor.get_by_role("alert")).to_be_visible()
        expect(_element(editor, "K_nm[0,1] = -7")).to_be_visible()
        expect(archive_input).to_have_value(original_json)
        expect(editor.get_by_role("button", name="Save parameter revision")).to_be_disabled()
    editor.get_by_label("Input unit").fill("seconds")
    _edit(editor, "-6")
    expect(editor.get_by_role("alert")).to_contain_text("unit mismatch")
    editor.get_by_label("Input unit").fill("mrad/s")
    editor.get_by_label("Selected value", exact=True).fill("-6000")
    editor.get_by_role("button", name="Convert mrad/s → rad/s and apply value").click()
    expect(editor.get_by_label("Input unit")).to_have_value("rad/s")
    expect(_element(editor, "K_nm[0,1] = -6")).to_be_visible()
    completed.append("invalid-values-domain-unit-refusal-and-explicit-conversion")


def _save_symmetric_child(
    workspace: Locator, original_json: str, observed: dict[str, object]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Edit symmetrically, mask trainability, save a child and check its native digest.

    Parameters
    ----------
    workspace : Locator
        Local workspace region with the admitted archive.
    original_json : str
        Admitted archive text.
    observed : dict[str, object]
        Caller-owned evidence receiving the Python child digest.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], str]
        The admitted archive, the saved archive and its text.

    """
    from playwright.sync_api import expect

    from scpn_quantum_control.studio_workspace.contracts import parse_experiment_revision
    from scpn_quantum_control.studio_workspace.json_transport import read_json

    editor = workspace.get_by_role("region", name="Linked parameter editor")
    editor.get_by_label("Matrix edit policy").select_option("symmetric")
    expect(_element(editor, "K_nm[1,0] = 5")).to_be_visible()
    _edit(editor, "-8")
    expect(_element(editor, "K_nm[1,0] = -8")).to_be_visible()
    editor.get_by_label("Selected element trainable").uncheck()
    child_save = editor.get_by_role("button", name="Save parameter revision")
    expect(child_save).to_be_enabled()
    child_save.click()
    expect(workspace.get_by_role("status")).to_contain_text(_COMMITTED)
    saved_json = workspace.get_by_label("Workspace archive JSON").input_value()
    original, saved = json.loads(original_json), json.loads(saved_json)
    assert saved["members"][:-1] == original["members"]
    assert len(saved["members"]) == len(original["members"]) + 1
    member = saved["members"][-1]
    child = parse_experiment_revision(read_json(member["content"]))
    assert child.digest == member["sha256"]
    observed["python_child_digest"] = child.digest
    values = json.loads(member["content"])["body"]["parameters"]["K_nm"]["values"]
    assert values == _SYMMETRIC_VALUES
    return original, saved, saved_json


def _first_interval(
    page: Page, hosts: tuple[str, str], corpus: str, lists: _Lists, observed: dict[str, object]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Admit, edit, refuse and save the first child under the first coverage interval.

    Parameters
    ----------
    page : Page
        Owned page for the source host.
    hosts : tuple[str, str]
        Owned built preview and root Vite source server.
    corpus : str
        Committed workspace document corpus.
    lists : _Lists
        Evidence lists.
    observed : dict[str, object]
        Caller-owned evidence.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], str]
        The admitted archive, the saved archive and its text.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    profiler = start_native_coverage(page)
    try:
        workspace, original_json = _admit_archive(page, hosts[1], corpus)
        editor = workspace.get_by_role("region", name="Linked parameter editor")
        _check_signed_edit(editor, lists.completed)
        archive_input = workspace.get_by_label("Workspace archive JSON")
        _check_refusals(editor, archive_input, original_json, lists.completed)
        saved = _save_symmetric_child(workspace, original_json, observed)
        lists.completed.append("symmetric-edit-mask-native-save-and-python-digest-parity")
        return saved
    finally:
        take_native_coverage(profiler, lists.records, include_parameters=True)


def _reload_and_save_again(page: Page, saved: dict[str, Any], saved_json: str) -> str:
    """Reload, find the mask and unit kept, and save a second child after the first.

    Parameters
    ----------
    page : Page
        Owned page on the source host.
    saved : dict[str, Any]
        First saved archive.
    saved_json : str
        Its text.

    Returns
    -------
    str
        The second saved archive text.

    """
    from playwright.sync_api import expect

    page.reload(wait_until="networkidle")
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(saved_json)
    editor = page.get_by_role("region", name="Linked parameter editor")
    _element(editor, "K_nm[1,0] = -8").click()
    expect(editor.get_by_label("Selected element trainable")).not_to_be_checked()
    expect(editor.get_by_label("Selected value", exact=True)).to_have_value("-8")
    expect(editor.get_by_label("Input unit", exact=True)).to_have_value("rad/s")
    _edit(editor, "-9")
    expect(_element(editor, "K_nm[1,0] = -9")).to_be_visible()
    child_save = editor.get_by_role("button", name="Save parameter revision")
    expect(child_save).to_be_enabled()
    child_save.click()
    expect(page.get_by_role("status")).to_contain_text(_COMMITTED)
    restored_json = page.get_by_label("Workspace archive JSON").input_value()
    restored = json.loads(restored_json)
    assert restored["members"][:-1] == saved["members"]
    assert len(restored["members"]) == len(saved["members"]) + 1
    return restored_json


def _check_named_marker(page: Page, corpus: str, restored_json: str) -> None:
    """Preview a Unicode-keyed archive, resolve its native marker and reload the save.

    Parameters
    ----------
    page : Page
        Owned page on the source host.
    corpus : str
        Committed workspace document corpus.
    restored_json : str
        Second saved archive text, which the reload must restore.

    """
    from playwright.sync_api import expect

    named = cast(dict[str, object], page.evaluate(_NAMED_ARCHIVE, corpus))
    page.get_by_label("Workspace archive JSON").fill(cast(str, named["json"]))
    page.get_by_role("button", name="Preview archive", exact=True).click()
    diagram = page.get_by_role("img", name="Coupling (A → B) coupling graph diagram")
    expect(diagram).to_be_visible()
    marker = cast(dict[str, object], diagram.locator(":scope > path").first.evaluate(_MARKER))
    identity = marker["id"]
    assert isinstance(identity, str) and not re.search(r"[\s()]", identity)
    assert marker["reference"] == f"url(#{identity})"
    assert marker["computed"] != "none"
    editor = page.get_by_role("region", name="Linked parameter editor")
    _element(editor, "Edge 1 → 0: -2 rad/s").click()
    named_element = _element(editor, "Coupling (A → B)[0,1] = -2")
    expect(named_element).to_have_attribute("aria-pressed", "true")
    page.get_by_role("button", name="Reload saved workspace", exact=True).click()
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(restored_json)


def _second_interval(
    page: Page, corpus: str, saved: tuple[dict[str, Any], str], lists: _Lists
) -> tuple[dict[str, Any], dict[str, object]]:
    """Reload, save again and resolve the Unicode-keyed graph under a second interval.

    Parameters
    ----------
    page : Page
        Owned page on the source host.
    corpus : str
        Committed workspace document corpus.
    saved : tuple[dict[str, Any], str]
        First saved archive and its text.
    lists : _Lists
        Evidence lists.

    Returns
    -------
    tuple[dict[str, Any], dict[str, object]]
        The second saved archive and the native controller refusal cases.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    profiler = start_native_coverage(page)
    try:
        restored_json = _reload_and_save_again(page, *saved)
        lists.completed.append("reload-mask-and-second-native-save-preserve-all-prior-results")
        _check_named_marker(page, corpus, restored_json)
        lists.completed.append("unicode-source-key-linked-graph-and-native-marker-resolution")
        refusals = cast(dict[str, object], page.evaluate(_CONTROLLER_CASES, corpus))
        return json.loads(restored_json), refusals
    finally:
        take_native_coverage(profiler, lists.records, include_parameters=True)


def _parameter_context(
    browser: Browser, hosts: tuple[str, str], lists: _Lists, observed: dict[str, object]
) -> None:
    """Run the built and source pages in one context bound to both owned origins.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    hosts : tuple[str, str]
        Owned built preview and root Vite source server.
    lists : _Lists
        Evidence lists.
    observed : dict[str, object]
        Caller-owned evidence.

    """
    origins = {urlsplit(host).netloc for host in hosts}
    corpus = _CORPUS.read_text(encoding="utf-8")
    with browser.new_context(service_workers="block") as context:
        context.set_default_timeout(15_000)

        def request(route: Route) -> None:
            target = urlsplit(route.request.url)
            if target.scheme == "http" and target.netloc in origins:
                route.continue_()
            else:
                lists.rejected.append(route.request.url)
                route.abort()

        context.route("**/*", request)
        context.on(
            "page",
            lambda page: page.on("pageerror", lambda error: lists.errors.append(str(error))),
        )
        built = context.new_page()
        _check_built(built, hosts[0], lists.completed)
        page = context.new_page()
        original, saved, saved_json = _first_interval(page, hosts, corpus, lists, observed)
        restored, observed["native_controller_refusal"] = _second_interval(
            page, corpus, (saved, saved_json), lists
        )
        for refused in (lists.errors, lists.rejected):
            assert not refused, refused
        assert not page.workers and not built.workers
        members = (len(archive["members"]) for archive in (original, saved, restored))
        observed.update(dict(zip(_MEMBER_COUNTS, members, strict=True)), workers=0)


def _owned_hosts(base_url: str, source_url: str) -> tuple[str, str]:
    """Validate the built preview and the distinct root Vite source server.

    Parameters
    ----------
    base_url : str
        Owned built Studio preview containing the genuine Rust WASM kernel.
    source_url : str
        Distinct owned root Vite origin with the unchanged production source.

    Returns
    -------
    tuple[str, str]
        The preview and the source server.

    Raises
    ------
    ValueError
        Addresses are unsafe, coincident, or the source has a path prefix.

    """
    from tools.studio_browser_journey import loopback_url

    preview, source = loopback_url(base_url), loopback_url(source_url)
    built_address, source_address = urlsplit(preview), urlsplit(source)
    same_server = (
        ipaddress.ip_address(cast(str, built_address.hostname))
        == ipaddress.ip_address(cast(str, source_address.hostname))
        and built_address.port == source_address.port
    )
    if same_server or source_address.path != "/":
        raise ValueError("Parameter journey requires a distinct owned root source server")
    return preview, source


def _start_evidence(
    observations: dict[str, object] | None, source: str
) -> tuple[dict[str, object], _Lists]:
    """Record the scenario, the source and the empty evidence lists.

    Parameters
    ----------
    observations : dict[str, object] | None
        Caller-owned partial evidence, or None for a fresh one.
    source : str
        Owned root Vite source server.

    Returns
    -------
    tuple[dict[str, object], _Lists]
        The evidence mapping and the lists it exposes.

    """
    lists = _Lists()
    observed = {} if observations is None else observations
    observed.update(
        scenario="parameter_graph_editor",
        source_url=source,
        observations=lists.completed,
        page_errors=lists.errors,
        external_requests=lists.rejected,
        native_v8_coverage=lists.records,
        coverage_percentage="not_calculated",
        playwright=version("playwright"),
        boundary="Original metadata corpus and native storage; no matrix solver or provider claim",
    )
    return observed, lists


def run_parameter_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Verify signed edits, refusal, history and immutable native child saves.

    Parameters
    ----------
    base_url : str
        Owned built Studio preview containing the genuine Rust WASM kernel.
    source_url : str
        Distinct owned root Vite origin with the unchanged production source.
    observations : dict[str, object] | None
        Caller-owned partial evidence retained when a later assertion fails.

    Returns
    -------
    dict[str, object]
        Actual browser observations, document identities and native counters.
        The explicit metadata corpus proves storage and UI behaviour only.

    Raises
    ------
    ValueError
        Addresses are unsafe, coincident, or the source has a path prefix.
    AssertionError
        A rendered edit, immutable save, recovery or runtime contract fails.

    """
    hosts = _owned_hosts(base_url, source_url)
    from playwright.sync_api import sync_playwright

    observed, lists = _start_evidence(observations, hosts[1])
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        observed["browser"] = browser.version
        try:
            _parameter_context(browser, hosts, lists, observed)
        finally:
            browser.close()
    return observed
