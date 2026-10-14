// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real workbench module contribution and source retention contracts

import { requirePresent } from "../../test-support/requirePresent";
import { webcrypto } from "node:crypto";
import type { ReactNode } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import settingsFixtureText from "../../../tests/data/studio_workspace/settings.json?raw";
import { sourceNotesModule } from "../../test-fixtures/source-notes";
import { unownedContributionsModule } from "../../test-fixtures/unowned-contributions";
import {
  canonicalDigest,
  documentDigest,
  parseResolvedSettings,
  parseWorkspaceManifest,
  readJson,
  writeJson,
} from "../shared/contracts";
import type { RawCodec } from "../shared/contracts";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { createWorkspaceArchive, admitWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { studioManifest } from "../shared/browser/data";
import { Workbench } from "./Workbench";
import { declaredBy, workbenchModules, workbenchRegistry } from "./moduleRegistry";

const originalCrypto = globalThis.crypto;
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  Object.defineProperty(globalThis, "crypto", { value: originalCrypto, configurable: true });
  window.history.replaceState(null, "", "/");
});

interface SettingsFixture {
  readonly document: unknown;
  readonly manifest: unknown;
  readonly raw: readonly {
    readonly digest: string;
    readonly record: {
      readonly schema: string;
      readonly body: { readonly role: string; readonly synthetic: boolean };
    };
  }[];
}

async function originalArchive() {
  const fixture = readJson(settingsFixtureText) as SettingsFixture;
  const manifest = parseWorkspaceManifest(fixture.manifest);
  const document = parseResolvedSettings(fixture.document);
  if (!manifest.ok || !document.ok) throw new Error("Original resolver fixture refused");
  const codec: RawCodec = async (content) => {
    const record = readJson(
      new TextDecoder().decode(content),
    ) as SettingsFixture["raw"][number]["record"];
    if (record.schema !== "review_fixture.v1" || record.body.synthetic !== true)
      throw new Error("Test-only raw source required");
    return {
      schema: record.schema,
      kind: record.body.role,
      digest: await canonicalDigest(record.schema, record),
    };
  };
  const codecs = new Map([["review_fixture.v1", codec]]);
  const raw = new Map(
    fixture.raw.map(({ digest, record }) => [
      digest,
      { schema: record.schema, content: new TextEncoder().encode(writeJson(record)) },
    ]),
  );
  const archive = await createWorkspaceArchive(
    manifest.value,
    new Map([[await documentDigest(document.value), document.value]]),
    raw,
    new Map(),
    codecs,
  );
  return { archive, codecs };
}

it("adds one compiled module entry and removes it while retaining exact admitted archive and source bytes", async () => {
  const { archive, codecs } = await originalArchive();
  const source = await admitWorkspaceArchive(archive.json, codecs);
  const original = requirePresent(
    source.members.find((member) => member.schema === "review_fixture.v1"),
  );
  const extended = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    sourceNotesModule,
  ]);
  const host = (_context: unknown, editors: ReactNode) => <section>Host overview{editors}</section>;
  const { rerender } = render(
    <Workbench registry={extended} rawCodecs={codecs}>
      {host}
    </Workbench>,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "No job requested",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Open source notes" }));
  await screen.findByRole("region", { name: "Source notes" });
  expect(window.location.hash).toBe("#/source-notes");
  fireEvent.click(screen.getByRole("button", { name: "Open Workspace" }));
  fireEvent.change(screen.getByLabelText("Workspace archive JSON"), {
    target: { value: archive.json },
  });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  const nodes = await screen.findAllByRole("button", { name: /^review_fixture\.v1 ·/ });
  fireEvent.click(
    requirePresent(nodes.find((node) => node.textContent?.includes(original.sha256.slice(0, 12)))),
  );
  expect(((await screen.findByLabelText("Source notes editor")) as HTMLTextAreaElement).value).toBe(
    original.content,
  );
  expect((screen.getByLabelText("Source notes inspector") as HTMLTextAreaElement).value).toBe(
    original.content,
  );
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(
    archive.json,
  );
  rerender(
    <Workbench registry={workbenchRegistry} rawCodecs={codecs}>
      {host}
    </Workbench>,
  );
  expect(await screen.findByText(/No editor is registered for review_fixture.v1/)).toBeTruthy();
  expect((screen.getByLabelText("Original document content") as HTMLTextAreaElement).value).toBe(
    original.content,
  );
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(
    archive.json,
  );
  expect(screen.queryByRole("button", { name: "Open source notes" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Open Build" }));
  await screen.findByRole("heading", { name: "Program editor" });
  expect(window.location.hash).toBe("#/build");
});

it("keeps a module visible with named missing requirements and does not execute its stage renderer", async () => {
  const unavailableModule = {
    ...sourceNotesModule,
    requires: { verbs: ["compile"], schemas: ["review_fixture.v1"] },
  };
  const extended = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    unavailableModule,
  ]);
  window.history.replaceState(null, "", "#/source-notes");
  const declarations = declaredBy(studioManifest);
  render(
    <Workbench registry={extended} declared={{ ...declarations, verbs: [] }}>
      {() => <p>Host overview</p>}
    </Workbench>,
  );
  expect(screen.getByRole("link", { name: "Source notes" })).toBeTruthy();
  expect(await screen.findByRole("heading", { name: "Source notes unavailable" })).toBeTruthy();
  expect(screen.getAllByText(/schema.*review_fixture\.v1/).length).toBeGreaterThan(0);
  expect(screen.queryByRole("region", { name: "Source notes" })).toBeNull();
  expect(
    (screen.getByRole("button", { name: "Open source notes" }) as HTMLButtonElement).disabled,
  ).toBe(true);
});

it("keeps unowned compiled slot contributions unavailable without invoking their callbacks or rewriting admitted source", async () => {
  const { archive, codecs } = await originalArchive();
  const compiled = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    unownedContributionsModule,
  ]);
  const empty = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([]);
  const mismatched = { ...compiled, contributionOwner: empty.contributionOwner };
  render(
    <Workbench registry={mismatched} rawCodecs={codecs}>
      {(_context, editors) => <section>Host overview{editors}</section>}
    </Workbench>,
  );
  expect(screen.getByRole("heading", { name: "Unowned stage editor unavailable" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Unowned explorer unavailable" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Open unowned probe" })).toMatchObject({
    disabled: true,
  });
  fireEvent.click(screen.getByRole("button", { name: "Open unowned probe" }));
  expect(window.location.hash).toBe("");
  fireEvent.change(screen.getByLabelText("Workspace archive JSON"), {
    target: { value: archive.json },
  });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await screen.findByRole("heading", { name: "Unowned settings unavailable" });
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(
    archive.json,
  );
  expect(screen.queryByText("An unowned contribution was invoked")).toBeNull();
});
