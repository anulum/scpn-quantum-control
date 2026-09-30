# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — browser workspace UI recovery cases
"""Exercise the real workspace file, preview, save, reload and export controls."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from playwright.sync_api import Page


def run_empty_project_ui(page: Page, *, navigate: bool = True) -> dict[str, object]:
    """Save, reload and download a genuine empty metadata project.

    Parameters
    ----------
    page
        Built public Studio page in an isolated browser context.
    navigate
        Reload the full document, or use the same public component's reload action.

    Returns
    -------
    dict[str, object]
        Exact downloaded text, saved identity and actual download name.

    """
    from playwright.sync_api import expect

    workspace = page.get_by_role("region", name="Local workspace")
    create = workspace.get_by_role("button", name="Create empty project")
    expect(create).to_be_enabled()
    workspace.get_by_label("New project title").fill("Browser recovery conformance")
    create.click()
    editor = workspace.get_by_label("Workspace archive JSON")
    expect(workspace.get_by_role("status")).to_contain_text("Empty project draft created")
    exact = editor.input_value()
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(workspace.get_by_role("status")).to_contain_text("Workspace transaction committed")
    identity = workspace.get_by_label("Saved workspace identity").inner_text()
    if navigate:
        page.reload(wait_until="networkidle")
    else:
        workspace.get_by_role("button", name="Reload saved workspace").click()
        expect(workspace.get_by_role("status")).to_contain_text("Restored exact saved workspace")
    expect(editor).to_have_value(exact)
    expect(workspace.get_by_label("Saved workspace identity")).to_have_text(
        identity, use_inner_text=True
    )
    with page.expect_download() as pending:
        workspace.get_by_role("button", name="Export saved archive").click()
    download = pending.value
    path = download.path()
    assert path is not None, "Owned portable export did not produce a file"
    exported = Path(path).read_text(encoding="utf-8")
    assert exported == exact, "Portable download changed exact saved JSON"
    return {
        "archive": exported,
        "identity": identity,
        "download_name": download.suggested_filename,
    }


def run_graph_ui(page: Page, expected: dict[str, object]) -> dict[str, object]:
    """Recover the original full synthetic revision graph through native File and public controls.

    Parameters
    ----------
    page
        Actual public component mounted with explicit test-only trusted source codecs.
    expected
        Original native API report containing exact archive text and admitted identities.

    Returns
    -------
    dict[str, object]
        Actual preview/save/export observations and exact identity comparisons.

    """
    from playwright.sync_api import expect

    archive = cast(str, expected["archive"])
    workspace = page.get_by_role("region", name="Local workspace")
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
    editor = workspace.get_by_label("Workspace archive JSON")
    retained_editor = editor.input_value()
    workspace.get_by_label("Workspace archive file").set_input_files([])
    expect(editor).to_have_value(retained_editor)
    workspace.get_by_label("Workspace archive file").set_input_files(
        {
            "name": "full-workspace.json",
            "mimeType": "application/json",
            "buffer": archive.encode("utf-8"),
        }
    )
    expect(workspace.get_by_role("status")).to_contain_text("Archive read locally")
    editor = workspace.get_by_label("Workspace archive JSON")
    expect(editor).to_have_value(archive)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    preview = workspace.get_by_label("Archive preview")
    expect(preview).to_contain_text(cast(str, expected["workspaceHash"]))
    for key in ("documentHashes", "rawHashes"):
        hashes = cast(list[str], expected[key])
        assert hashes, f"Original full graph has no {key}"
        for digest in hashes:
            expect(preview).to_contain_text(digest)
    with page.expect_download() as pending:
        workspace.get_by_role("button", name="Export preview archive").click()
    preview_path = pending.value.path()
    assert preview_path is not None, "Graph preview export produced no native download"
    assert Path(preview_path).read_text(encoding="utf-8") == archive
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(workspace.get_by_role("status")).to_contain_text("Workspace transaction committed")
    saved = workspace.get_by_label("Saved workspace identity")
    expect(saved).to_contain_text(cast(str, expected["archiveDigest"]))
    expect(saved).to_contain_text(cast(str, expected["workspaceHash"]))
    identity = saved.inner_text()
    editor.fill("unsaved invalid edit; saved results must remain unchanged")
    expect(workspace.get_by_role("button", name="Export preview archive")).to_be_disabled()
    with page.expect_download() as pending:
        workspace.get_by_role("button", name="Export saved archive").click()
    saved_path = pending.value.path()
    assert saved_path is not None, "Saved graph export produced no native download"
    exported = Path(saved_path).read_text(encoding="utf-8")
    assert exported == archive, "Editing the draft rebound the saved export"
    workspace.get_by_role("button", name="Reload saved workspace").click()
    expect(workspace.get_by_role("status")).to_contain_text("Restored exact saved workspace")
    expect(editor).to_have_value(archive)
    future = archive.replace('"quantum_workspace_archive.v1"', '"quantum_workspace_archive.v2"', 1)
    assert future != archive, "Future-major case did not change the root container version"
    editor.fill(future)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    expect(workspace.get_by_role("status")).to_contain_text("Unsupported archive schema")
    expect(save).to_be_disabled()
    expect(saved).to_have_text(identity, use_inner_text=True)
    workspace.get_by_role("button", name="Reload saved workspace").click()
    expect(workspace.get_by_role("status")).to_contain_text("Restored exact saved workspace")
    expect(editor).to_have_value(archive)
    expect(saved).to_have_text(identity, use_inner_text=True)
    workspace.get_by_label("Workspace archive file").set_input_files(
        {"name": "invalid-encoding.json", "mimeType": "application/json", "buffer": b"\xff"}
    )
    expect(workspace.get_by_role("status")).to_contain_text("Archive file must be valid UTF-8")
    expect(editor).to_have_value(archive)
    expect(saved).to_have_text(identity, use_inner_text=True)
    bom_archive = "\ufeff" + archive
    workspace.get_by_label("Workspace archive file").set_input_files(
        {"name": "bom.json", "mimeType": "application/json", "buffer": bom_archive.encode("utf-8")}
    )
    expect(workspace.get_by_role("status")).to_contain_text("Archive read locally")
    expect(editor).to_have_value(bom_archive)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    expect(workspace.get_by_role("status")).to_contain_text("value required")
    expect(save).to_be_disabled()
    expect(saved).to_have_text(identity, use_inner_text=True)
    workspace.get_by_role("button", name="Reload saved workspace").click()
    expect(workspace.get_by_role("status")).to_contain_text("Restored exact saved workspace")
    expect(editor).to_have_value(archive)
    return {
        "archive": exported,
        "identity": identity,
        "documentHashes": expected["documentHashes"],
        "rawHashes": expected["rawHashes"],
        "future_major_refused": True,
        "saved_export_unchanged_after_edit": True,
        "invalid_utf8_refused_without_replacing_editor": True,
        "bom_preserved_and_original_json_reader_refused": True,
        "boundary": "Real UI/native store; explicitly synthetic original metadata, no hardware claim",
    }


def run_formatting_ui(page: Page, expected: dict[str, object]) -> dict[str, object]:
    """Persist a formatting-only draft while preserving original workspace and evidence identities.

    Parameters
    ----------
    page
        Real workspace component with its original full graph already saved.
    expected
        Original native graph report used by the preceding File import.

    Returns
    -------
    dict[str, object]
        Actual admitted snapshot and saved identity; no revision is rewritten.

    """
    from playwright.sync_api import expect

    formatted = "\n" + cast(str, expected["archive"]) + "\n"
    observed = cast(
        dict[str, object],
        page.evaluate(
            "async json => { const archive = await import('/src/shared/storage/workspaceArchive.ts'); const fixture = await import('/browser-tests/workspaceFixture.ts'); return archive.previewWorkspaceArchive(json, fixture.conformanceCodecs); }",
            formatted,
        ),
    )
    assert observed["archiveDigest"] != expected["archiveDigest"]
    for key in ("workspaceHash", "documentHashes", "rawHashes"):
        assert observed[key] == expected[key], f"Formatting changed original {key}"
    workspace = page.get_by_role("region", name="Local workspace")
    editor = workspace.get_by_label("Workspace archive JSON")
    editor.fill(formatted)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    expect(workspace.get_by_label("Archive preview")).to_contain_text(
        cast(str, observed["archiveDigest"])
    )
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(workspace.get_by_role("status")).to_contain_text("Workspace transaction committed")
    saved = workspace.get_by_label("Saved workspace identity")
    expect(saved).to_contain_text(cast(str, observed["archiveDigest"]))
    expect(saved).to_contain_text(cast(str, expected["workspaceHash"]))
    workspace.get_by_role("button", name="Reload saved workspace").click()
    expect(workspace.get_by_role("status")).to_contain_text("Restored exact saved workspace")
    expect(editor).to_have_value(formatted)
    return observed
