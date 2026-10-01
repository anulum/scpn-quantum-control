// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned settings behaviour tests

import { webcrypto } from "node:crypto";
import { afterAll, beforeAll, expect, it } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import fixtureText from "../../../../tests/data/studio_workspace/settings.json?raw";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { documentDigest, parseResolvedSettings, parseWorkspaceManifest } from "../../shared/contracts/workspace";
import { readJson, writeJson } from "../../shared/contracts/jsonTransport";
import { createWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { SettingsInspector } from "./SettingsInspector";
import { WorkspacePanel } from "../workspace/WorkspacePanel";

interface Fixture {
  readonly manifest: unknown;
  readonly document: { readonly schema: string; readonly extensions: unknown; readonly body: Record<string, unknown> };
  readonly raw: readonly { readonly digest: string; readonly record: {
    readonly schema: string; readonly body: { readonly role: string; readonly synthetic: boolean };
  } }[];
}
const fixture = readJson(fixtureText) as Fixture;
const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeAll(() => Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto }));
afterAll(() => {
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Real codec for the explicit synthetic conformance schema; never a built-in provider. */
function fixtureCodecs() {
  return new Map([["review_fixture.v1", async (content: Uint8Array) => {
    const record = readJson(new TextDecoder().decode(content)) as Fixture["raw"][number]["record"];
    if (record.schema !== "review_fixture.v1" || record.body.synthetic !== true) throw new Error("Synthetic owner required");
    return { schema: record.schema, kind: record.body.role, digest: await canonicalDigest(record.schema, record) };
  }]]);
}

/** Admit literal source settings through the original archive API and actual Node WebCrypto. */
async function admitted(body: Record<string, unknown> = fixture.document.body): Promise<WorkspaceArchivePreview> {
  const manifest = parseWorkspaceManifest(fixture.manifest);
  const document = parseResolvedSettings({ ...fixture.document, body });
  if (!manifest.ok || !document.ok) throw new Error("Shared fixture refused");
  const raw = new Map(fixture.raw.map(({ digest, record }) => [
    digest, { schema: record.schema, content: new TextEncoder().encode(writeJson(record)) },
  ]));
  return createWorkspaceArchive(manifest.value,
    new Map([[await documentDigest(document.value), document.value]]), raw, new Map(),
    fixtureCodecs());
}

it("renders exact source-owned values and winning origins through the original inspector", async () => {
  const preview = await admitted();
  const before = preview.json;
  render(<SettingsInspector preview={preview} />);
  const table = screen.getByRole("table", { name: "Requested and effective settings" });
  const shots = within(table).getByRole("rowheader", { name: "shots" }).parentElement!;
  expect(within(shots).getAllByRole("cell").map(cell => cell.textContent)).toEqual(["7", "7", '"run"']);
  const seed = within(table).getByRole("rowheader", { name: "seed" }).parentElement!;
  expect(within(seed).getAllByRole("cell").map(cell => cell.textContent)).toEqual([
    "9007199254740993", "9007199254740993", '"defaults"',
  ]);
  expect(within(table).getAllByRole("rowheader").map(row => row.textContent)).toEqual([
    "precision", "seed", "shots", "theme",
  ]);
  expect(screen.getByText(/Policy reference/).textContent).toContain("review_fixture.v1");
  expect(screen.getByText(/Environment reference/).textContent).toContain("review_fixture.v1");
  expect(screen.getByText(/Settings digest/).textContent).toContain(preview.documentHashes[0]);
  expect(preview.json).toBe(before);
});

it("retains recorded effective values when a legacy source omitted requested values", async () => {
  render(<SettingsInspector preview={await admitted({ ...fixture.document.body, requested: {} })} />);
  expect(screen.getAllByText("Not recorded")).toHaveLength(4);
  const shots = screen.getByRole("rowheader", { name: "shots" }).parentElement!;
  expect(within(shots).getAllByRole("cell").map(cell => cell.textContent)).toEqual(["Not recorded", "7", '"run"']);
});

it("shows absence honestly for a real admitted project without settings", async () => {
  const manifest = parseWorkspaceManifest(fixture.manifest);
  if (!manifest.ok) throw new Error(manifest.message);
  const preview = await createWorkspaceArchive(manifest.value, new Map(), new Map(), new Map());
  render(<SettingsInspector preview={preview} />);
  expect(screen.getByText("No resolved settings records in this workspace.")).toBeTruthy();
  expect(screen.queryByRole("table")).toBeNull();
});

it("shows an authored refusal for damaged text without a partial settings table", async () => {
  const preview = await admitted();
  render(<SettingsInspector preview={{ ...preview, json: "{" }} />);
  expect(screen.getByRole("alert").textContent).toBe("Settings records could not be inspected.");
  expect(screen.queryByRole("table")).toBeNull();
});

it("reaches settings through the original workspace archive preview without claiming a saved draft", async () => {
  const preview = await admitted();
  render(<WorkspacePanel rawCodecs={fixtureCodecs()} />);
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("IndexedDB unavailable"));
  const input = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  fireEvent.change(input, { target: { value: preview.json } });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await screen.findByRole("region", { name: "Resolved settings provenance" });
  expect(screen.getByRole("rowheader", { name: "shots" }).parentElement!.textContent).toContain('7"run"');
  expect(screen.getByText(/Settings digest/).textContent).toContain(preview.documentHashes[0]);
  expect(input.value).toBe(preview.json);
  expect((screen.getByRole("button", { name: "Save draft and revision references" }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.queryByRole("definition", { name: "Saved workspace digest" })).toBeNull();
});
