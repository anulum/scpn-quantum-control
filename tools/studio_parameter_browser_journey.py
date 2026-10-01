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
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Locator


def _edit(editor: Locator, value: str) -> None:
    """Apply decimal text through the actual selected-parameter form.

    Parameters
    ----------
    editor
        Original production editor region in the native browser.
    value
        Decimal text to validate against the original source specification.

    """
    editor.get_by_label("Selected value", exact=True).fill(value)
    editor.get_by_role("button", name="Apply value", exact=True).click()


def run_parameter_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Verify signed edits, refusal, history and immutable native child saves.

    Parameters
    ----------
    base_url
        Owned built Studio preview containing the genuine Rust WASM kernel.
    source_url
        Distinct owned root Vite origin with the unchanged production source.
    observations
        Caller-owned partial evidence retained when a later assertion fails.

    Returns
    -------
    dict[str, object]
        Actual browser observations, document identities and native counters.
        The explicit metadata corpus proves storage and UI behavior only.

    Raises
    ------
    ValueError
        Addresses are unsafe, coincident, or the source has a path prefix.
    AssertionError
        A rendered edit, immutable save, recovery or runtime contract fails.

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
    from playwright.sync_api import Route, expect, sync_playwright

    from scpn_quantum_control.studio_workspace.contracts import parse_experiment_revision
    from scpn_quantum_control.studio_workspace.json_transport import read_json
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    observed = {} if observations is None else observations
    completed: list[str] = []
    errors: list[str] = []
    rejected: list[str] = []
    records: list[dict[str, object]] = []
    observed.update(
        scenario="parameter_graph_editor",
        source_url=source,
        observations=completed,
        page_errors=errors,
        external_requests=rejected,
        native_v8_coverage=records,
        coverage_percentage="not_calculated",
        playwright=version("playwright"),
        boundary="Original metadata corpus and native storage; no matrix solver or provider claim",
    )
    origins = {built_address.netloc, source_address.netloc}
    corpus = (
        Path(__file__).resolve().parents[1] / "tests/data/studio_workspace/documents.json"
    ).read_text(encoding="utf-8")
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            with browser.new_context(service_workers="block") as context:
                context.set_default_timeout(15_000)

                def request(route: Route) -> None:
                    target = urlsplit(route.request.url)
                    if target.scheme == "http" and target.netloc in origins:
                        route.continue_()
                    else:
                        rejected.append(route.request.url)
                        route.abort()

                context.route("**/*", request)
                context.on(
                    "page",
                    lambda page: page.on("pageerror", lambda error: errors.append(str(error))),
                )
                built = context.new_page()
                built.goto(preview, wait_until="networkidle")
                expect(built.get_by_role("img", name="order parameter over time")).to_be_visible()
                expect(
                    built.get_by_text("verified against the committed ground truth", exact=False)
                ).to_be_visible()
                built.get_by_role("link", name="Edit source parameters in Workspace").click()
                expect(built).to_have_url(preview + "#/workspace")
                completed.append("built-original-wasm-and-workspace-link")
                page = context.new_page()
                profiler = start_native_coverage(page)
                try:
                    page.goto(
                        source + "browser-tests/parameterEditor.html", wait_until="networkidle"
                    )
                    native = cast(
                        dict[str, object],
                        page.evaluate(
                            "async corpus => (await import('/browser-tests/parameterEditor.native.tsx')).createParameterConformanceArchive(corpus)",
                            corpus,
                        ),
                    )
                    original_json = cast(str, native["json"])
                    workspace = page.get_by_role("region", name="Local workspace")
                    archive_input = workspace.get_by_label("Workspace archive JSON")
                    expect(
                        workspace.get_by_role("button", name="Create empty project")
                    ).to_be_enabled()
                    archive_input.fill(original_json)
                    workspace.get_by_role("button", name="Preview archive", exact=True).click()
                    save = workspace.get_by_role(
                        "button", name="Save draft and revision references"
                    )
                    expect(save).to_be_enabled()
                    save.click()
                    expect(workspace.get_by_role("status")).to_contain_text(
                        "Workspace transaction committed"
                    )
                    editor = workspace.get_by_role("region", name="Linked parameter editor")
                    expect(
                        editor.get_by_role("img", name="K_nm coupling graph diagram")
                    ).to_be_visible()
                    digest = editor.get_by_label("Draft semantic digest")
                    expect(digest).to_have_text(re.compile("^[0-9a-f]{64}$"))
                    initial_digest = digest.inner_text()
                    editor.get_by_role("button", name="Edge 1 → 0: -2 rad/s", exact=True).click()
                    expect(editor.get_by_label("Selected value", exact=True)).to_have_value("-2")
                    _edit(editor, "-7")
                    expect(
                        editor.get_by_role("button", name="K_nm[0,1] = -7", exact=True)
                    ).to_be_visible()
                    expect(
                        editor.get_by_role("button", name="K_nm[1,0] = 5", exact=True)
                    ).to_be_visible()
                    expect(
                        editor.get_by_role("button", name="Edge 1 → 0: -7 rad/s", exact=True)
                    ).to_be_visible()
                    expect(digest).not_to_have_text(initial_digest)
                    editor.get_by_role("button", name="Undo parameter edit").click()
                    expect(digest).to_have_text(initial_digest)
                    editor.get_by_role("button", name="Redo parameter edit").click()
                    expect(
                        editor.get_by_role("button", name="K_nm[0,1] = -7", exact=True)
                    ).to_be_visible()
                    completed.append("signed-directed-graph-form-and-exact-undo")
                    for refused in ("NaN", "Infinity", "11"):
                        _edit(editor, refused)
                        expect(editor.get_by_role("alert")).to_be_visible()
                        expect(
                            editor.get_by_role("button", name="K_nm[0,1] = -7", exact=True)
                        ).to_be_visible()
                        expect(archive_input).to_have_value(original_json)
                        expect(
                            editor.get_by_role("button", name="Save parameter revision")
                        ).to_be_disabled()
                    editor.get_by_label("Input unit").fill("seconds")
                    _edit(editor, "-6")
                    expect(editor.get_by_role("alert")).to_contain_text("unit mismatch")
                    editor.get_by_label("Input unit").fill("mrad/s")
                    editor.get_by_label("Selected value", exact=True).fill("-6000")
                    editor.get_by_role(
                        "button", name="Convert mrad/s → rad/s and apply value"
                    ).click()
                    expect(editor.get_by_label("Input unit")).to_have_value("rad/s")
                    expect(
                        editor.get_by_role("button", name="K_nm[0,1] = -6", exact=True)
                    ).to_be_visible()
                    completed.append("invalid-values-domain-unit-refusal-and-explicit-conversion")
                    editor.get_by_label("Matrix edit policy").select_option("symmetric")
                    expect(
                        editor.get_by_role("button", name="K_nm[1,0] = 5", exact=True)
                    ).to_be_visible()
                    _edit(editor, "-8")
                    expect(
                        editor.get_by_role("button", name="K_nm[1,0] = -8", exact=True)
                    ).to_be_visible()
                    editor.get_by_label("Selected element trainable").uncheck()
                    child_save = editor.get_by_role("button", name="Save parameter revision")
                    expect(child_save).to_be_enabled()
                    child_save.click()
                    expect(workspace.get_by_role("status")).to_contain_text(
                        "Parameter revision transaction committed"
                    )
                    saved_json = archive_input.input_value()
                    original, saved = json.loads(original_json), json.loads(saved_json)
                    assert saved["members"][:-1] == original["members"]
                    assert len(saved["members"]) == len(original["members"]) + 1
                    member = saved["members"][-1]
                    child = parse_experiment_revision(read_json(member["content"]))
                    assert child.digest == member["sha256"]
                    observed["python_child_digest"] = child.digest
                    values = json.loads(member["content"])["body"]["parameters"]["K_nm"]["values"]
                    assert values == [
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
                    completed.append("symmetric-edit-mask-native-save-and-python-digest-parity")
                finally:
                    take_native_coverage(profiler, records, include_parameters=True)
                page.reload(wait_until="networkidle")
                profiler = start_native_coverage(page)
                try:
                    expect(page.get_by_label("Workspace archive JSON")).to_have_value(saved_json)
                    editor = page.get_by_role("region", name="Linked parameter editor")
                    editor.get_by_role("button", name="K_nm[1,0] = -8", exact=True).click()
                    expect(editor.get_by_label("Selected element trainable")).not_to_be_checked()
                    _edit(editor, "-9")
                    child_save = editor.get_by_role("button", name="Save parameter revision")
                    expect(child_save).to_be_enabled()
                    child_save.click()
                    expect(page.get_by_role("status")).to_contain_text(
                        "Parameter revision transaction committed"
                    )
                    restored_json = page.get_by_label("Workspace archive JSON").input_value()
                    restored = json.loads(restored_json)
                    assert restored["members"][:-1] == saved["members"]
                    assert len(restored["members"]) == len(saved["members"]) + 1
                    completed.append(
                        "reload-mask-and-second-native-save-preserve-all-prior-results"
                    )
                    named = cast(
                        dict[str, object],
                        page.evaluate(
                            "async corpus => (await import('/browser-tests/parameterEditor.native.tsx')).createParameterConformanceArchive(corpus, 'Coupling (A → B)')",
                            corpus,
                        ),
                    )
                    page.get_by_label("Workspace archive JSON").fill(cast(str, named["json"]))
                    page.get_by_role("button", name="Preview archive", exact=True).click()
                    diagram = page.get_by_role(
                        "img", name="Coupling (A → B) coupling graph diagram"
                    )
                    expect(diagram).to_be_visible()
                    marker = cast(
                        dict[str, object],
                        diagram.locator(":scope > path").first.evaluate(
                            "path => ({id: path.ownerSVGElement.querySelector('marker').id, reference: path.getAttribute('marker-end'), computed: getComputedStyle(path).markerEnd})"
                        ),
                    )
                    identity = marker["id"]
                    assert isinstance(identity, str) and not re.search(r"[\s()]", identity)
                    assert marker["reference"] == f"url(#{identity})"
                    assert marker["computed"] != "none"
                    editor.get_by_role("button", name="Edge 1 → 0: -2 rad/s", exact=True).click()
                    expect(
                        editor.get_by_role("button", name="Coupling (A → B)[0,1] = -2", exact=True)
                    ).to_have_attribute("aria-pressed", "true")
                    page.get_by_role("button", name="Reload saved workspace", exact=True).click()
                    expect(page.get_by_label("Workspace archive JSON")).to_have_value(
                        restored_json
                    )
                    completed.append(
                        "unicode-source-key-linked-graph-and-native-marker-resolution"
                    )
                    observed["native_controller_refusal"] = page.evaluate(
                        "async corpus => (await import('/browser-tests/parameterEditor.native.tsx')).runParameterControllerCases(corpus)",
                        corpus,
                    )
                finally:
                    take_native_coverage(profiler, records, include_parameters=True)
                assert not errors, errors
                assert not rejected, rejected
                assert not page.workers and not built.workers
                observed.update(
                    original_members=len(original["members"]),
                    first_child_members=len(saved["members"]),
                    second_child_members=len(restored["members"]),
                    workers=0,
                )
        finally:
            browser.close()
    return observed
