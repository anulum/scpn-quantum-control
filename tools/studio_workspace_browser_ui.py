# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — browser workspace UI recovery cases
"""Exercise the real workspace file, preview, save, reload and export controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


_RESTORED = "Restored exact saved workspace"
_COMMITTED = "Workspace transaction committed"


def _create_and_save(workspace: Locator) -> tuple[str, str]:
    """Create an empty project, preview it and save it.

    Parameters
    ----------
    workspace
        Local workspace region.

    Returns
    -------
    tuple[str, str]
        The exact draft text and the saved identity.

    """
    from playwright.sync_api import expect

    create = workspace.get_by_role("button", name="Create empty project")
    expect(create).to_be_enabled()
    workspace.get_by_label("New project title").fill("Browser recovery conformance")
    create.click()
    expect(workspace.get_by_role("status")).to_contain_text("Empty project draft created")
    exact = workspace.get_by_label("Workspace archive JSON").input_value()
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(workspace.get_by_role("status")).to_contain_text(_COMMITTED)
    return exact, workspace.get_by_label("Saved workspace identity").inner_text()


def _download_text(page: Page, button: Locator, missing: str) -> tuple[str, str]:
    """Click an export button and read the downloaded text.

    Parameters
    ----------
    page
        Page that receives the download.
    button
        Export button.
    missing
        Failure message when no file is produced.

    Returns
    -------
    tuple[str, str]
        The downloaded text and the suggested file name.

    """
    with page.expect_download() as pending:
        button.click()
    path = pending.value.path()
    assert path is not None, missing
    return Path(path).read_text(encoding="utf-8"), pending.value.suggested_filename


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
    exact, identity = _create_and_save(workspace)
    if navigate:
        page.reload(wait_until="networkidle")
    else:
        workspace.get_by_role("button", name="Reload saved workspace").click()
        expect(workspace.get_by_role("status")).to_contain_text(_RESTORED)
    expect(workspace.get_by_label("Workspace archive JSON")).to_have_value(exact)
    saved = workspace.get_by_label("Saved workspace identity")
    expect(saved).to_have_text(identity, use_inner_text=True)
    export = workspace.get_by_role("button", name="Export saved archive")
    exported, name = _download_text(page, export, "Owned portable export did not produce a file")
    assert exported == exact, "Portable download changed exact saved JSON"
    return {"archive": exported, "identity": identity, "download_name": name}


@dataclass(frozen=True)
class _Graph:
    """Workspace controls and the exact archive of the graph recovery cases."""

    page: Page
    workspace: Locator
    archive: str

    @property
    def editor(self) -> Locator:
        """Return the workspace archive editor.

        Returns
        -------
        Locator
            The editor.

        """
        return self.workspace.get_by_label("Workspace archive JSON")

    @property
    def saved(self) -> Locator:
        """Return the saved workspace identity.

        Returns
        -------
        Locator
            The identity.

        """
        return self.workspace.get_by_label("Saved workspace identity")

    @property
    def save(self) -> Locator:
        """Return the save button.

        Returns
        -------
        Locator
            The button.

        """
        return self.workspace.get_by_role("button", name="Save draft and revision references")

    def upload(self, name: str, content: bytes) -> None:
        """Choose an archive file through the native file input.

        Parameters
        ----------
        name
            File name.
        content
            Exact file bytes.

        """
        self.workspace.get_by_label("Workspace archive file").set_input_files(
            {"name": name, "mimeType": "application/json", "buffer": content}
        )

    def reload_saved(self) -> None:
        """Reload the saved workspace and find the exact archive restored."""
        from playwright.sync_api import expect

        self.workspace.get_by_role("button", name="Reload saved workspace").click()
        expect(self.workspace.get_by_role("status")).to_contain_text(_RESTORED)
        expect(self.editor).to_have_value(self.archive)


def _load_and_preview(graph: _Graph, expected: dict[str, object]) -> None:
    """Keep the draft on an empty file choice, load the archive and preview every hash.

    Parameters
    ----------
    graph
        The graph recovery controls.
    expected
        Original native API report with the admitted identities.

    """
    from playwright.sync_api import expect

    workspace = graph.workspace
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
    retained_editor = graph.editor.input_value()
    workspace.get_by_label("Workspace archive file").set_input_files([])
    expect(graph.editor).to_have_value(retained_editor)
    graph.upload("full-workspace.json", graph.archive.encode("utf-8"))
    expect(workspace.get_by_role("status")).to_contain_text("Archive read locally")
    expect(graph.editor).to_have_value(graph.archive)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    preview = workspace.get_by_label("Archive preview")
    expect(preview).to_contain_text(cast(str, expected["workspaceHash"]))
    for key in ("documentHashes", "rawHashes"):
        hashes = cast(list[str], expected[key])
        assert hashes, f"Original full graph has no {key}"
        for digest in hashes:
            expect(preview).to_contain_text(digest)


def _save_and_export(graph: _Graph, expected: dict[str, object]) -> tuple[str, str]:
    """Export the preview, save it, then export the save after an unsaved edit.

    Parameters
    ----------
    graph
        The graph recovery controls.
    expected
        Original native API report with the admitted identities.

    Returns
    -------
    tuple[str, str]
        The saved identity and the saved export.

    """
    from playwright.sync_api import expect

    workspace = graph.workspace
    preview_export = workspace.get_by_role("button", name="Export preview archive")
    missing = "Graph preview export produced no native download"
    assert _download_text(graph.page, preview_export, missing)[0] == graph.archive
    expect(graph.save).to_be_enabled()
    graph.save.click()
    expect(workspace.get_by_role("status")).to_contain_text(_COMMITTED)
    expect(graph.saved).to_contain_text(cast(str, expected["archiveDigest"]))
    expect(graph.saved).to_contain_text(cast(str, expected["workspaceHash"]))
    identity = graph.saved.inner_text()
    graph.editor.fill("unsaved invalid edit; saved results must remain unchanged")
    expect(preview_export).to_be_disabled()
    saved_export = workspace.get_by_role("button", name="Export saved archive")
    missing = "Saved graph export produced no native download"
    exported = _download_text(graph.page, saved_export, missing)[0]
    assert exported == graph.archive, "Editing the draft rebound the saved export"
    graph.reload_saved()
    return identity, exported


def _refuse_future_and_encodings(graph: _Graph, identity: str) -> None:
    """Refuse a future major, invalid UTF-8 and a BOM, keeping the save each time.

    Parameters
    ----------
    graph
        The graph recovery controls.
    identity
        Saved identity, which every refusal must keep.

    """
    from playwright.sync_api import expect

    status = graph.workspace.get_by_role("status")
    preview = graph.workspace.get_by_role("button", name="Preview archive", exact=True)
    future = graph.archive.replace(
        '"quantum_workspace_archive.v1"', '"quantum_workspace_archive.v2"', 1
    )
    assert future != graph.archive, "Future-major case did not change the root container version"
    graph.editor.fill(future)
    preview.click()
    expect(status).to_contain_text("Unsupported archive schema")
    expect(graph.save).to_be_disabled()
    expect(graph.saved).to_have_text(identity, use_inner_text=True)
    graph.reload_saved()
    expect(graph.saved).to_have_text(identity, use_inner_text=True)
    graph.upload("invalid-encoding.json", b"\xff")
    expect(status).to_contain_text("Archive file must be valid UTF-8")
    expect(graph.editor).to_have_value(graph.archive)
    expect(graph.saved).to_have_text(identity, use_inner_text=True)
    bom_archive = "\ufeff" + graph.archive
    graph.upload("bom.json", bom_archive.encode("utf-8"))
    expect(status).to_contain_text("Archive read locally")
    expect(graph.editor).to_have_value(bom_archive)
    preview.click()
    expect(status).to_contain_text("value required")
    expect(graph.save).to_be_disabled()
    expect(graph.saved).to_have_text(identity, use_inner_text=True)
    graph.reload_saved()


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
    graph = _Graph(
        page, page.get_by_role("region", name="Local workspace"), cast(str, expected["archive"])
    )
    _load_and_preview(graph, expected)
    identity, exported = _save_and_export(graph, expected)
    _refuse_future_and_encodings(graph, identity)
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
