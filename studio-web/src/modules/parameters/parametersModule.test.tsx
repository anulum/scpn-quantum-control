// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered parameter document source admission

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { declaredBy, workbenchRegistry } from "../../app/moduleRegistry";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import { studioManifest } from "../../shared/browser/data";
import type { ModuleDocument } from "../../shared/contracts/module";
import { ModuleDocumentViews } from "../../app/moduleViews/ModuleDocumentViews";
import { admitWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import { useWorkspace } from "../../shared/workspace/useWorkspace";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Keep an original selected document while its actual workspace admission changes. */
function SelectedParameterDocument({
  document,
  json,
}: {
  readonly document: ModuleDocument;
  readonly json: string;
}) {
  const workspace = useWorkspace(conformanceCodecs);
  return (
    <>
      <button type="button" disabled={workspace.busy} onClick={() => workspace.edit(json)}>
        Read original archive
      </button>
      <button type="button" onClick={() => void workspace.inspect()}>
        Admit original archive
      </button>
      <output aria-label="Current preview identity">
        {workspace.preview?.archiveDigest ?? "No preview"}
      </output>
      <label>
        Original archive bytes
        <textarea readOnly value={workspace.draft} />
      </label>
      <ModuleDocumentViews
        registry={workbenchRegistry}
        slot="editors"
        document={document}
        services={{ workspace, codecs: conformanceCodecs }}
        declared={declaredBy(studioManifest)}
      />
    </>
  );
}

it.each(["no preview", "different digest", "changed document bytes"])(
  "refuses the original registered parameter document with %s",
  async (cause) => {
    const archive = await conformanceArchive(corpusText, false);
    const other = await admitWorkspaceArchive(`${archive.json}\n`, conformanceCodecs);
    expect(other.preview.archiveDigest).not.toBe(archive.archiveDigest);
    const document: ModuleDocument = {
      schema: archive.schema,
      sha256: cause === "different digest" ? other.preview.archiveDigest : archive.archiveDigest,
      content: cause === "no preview" ? archive.json : other.preview.json,
    };
    render(<SelectedParameterDocument document={document} json={archive.json} />);
    if (cause !== "no preview") {
      const read = screen.getByRole("button", { name: "Read original archive" });
      await waitFor(() => expect(read).toMatchObject({ disabled: false }));
      fireEvent.click(read);
      fireEvent.click(screen.getByRole("button", { name: "Admit original archive" }));
      await waitFor(() =>
        expect(screen.getByLabelText("Current preview identity").textContent).toBe(
          archive.archiveDigest,
        ),
      );
    }
    expect(screen.getByRole("alert").textContent).toContain("Parameter source changed");
    expect(screen.queryByLabelText("Selected value")).toBeNull();
    expect(screen.queryByRole("button", { name: "Save parameter revision" })).toBeNull();
    expect(screen.getByLabelText("Original archive bytes")).toHaveProperty(
      "value",
      cause === "no preview" ? "" : archive.json,
    );
  },
);
