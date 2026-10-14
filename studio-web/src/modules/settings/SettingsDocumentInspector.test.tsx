// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original settings document identity and lifecycle

import { webcrypto } from "node:crypto";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import fixtureText from "../../../../tests/data/studio_workspace/settings.json?raw";
import { documentDigest, parseResolvedSettings } from "../../shared/contracts/workspace";
import { readJson, writeJson } from "../../shared/contracts/jsonTransport";
import type { ModuleDocument } from "../../shared/contracts/module";
import { SettingsDocumentInspector } from "./SettingsDocumentInspector";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Bind the literal original resolver fixture with its actual native document digest. */
async function sourceDocument(): Promise<ModuleDocument> {
  const fixture = readJson(fixtureText) as { readonly document: unknown };
  const parsed = parseResolvedSettings(fixture.document);
  if (!parsed.ok) throw new Error("Original resolver fixture required");
  return {
    schema: parsed.value.schema,
    content: writeJson(parsed.value),
    sha256: await documentDigest(parsed.value),
  };
}

it("displays original resolved values only after their actual document identity is verified", async () => {
  const document = await sourceDocument();
  const before = { ...document };
  render(<SettingsDocumentInspector document={document} />);
  expect(screen.getByRole("status").textContent).toContain("Reading original settings");
  await screen.findByRole("region", { name: "Settings document inspector" });
  expect(screen.getByRole("table", { name: "Requested and effective settings" })).toBeTruthy();
  expect(screen.getByText(/Settings digest/).textContent).toContain(document.sha256);
  expect(screen.getAllByText("9007199254740993").length).toBeGreaterThan(0);
  expect(document).toEqual(before);
});

it.each(["malformed transport", "unsupported schema", "different digest"])(
  "refuses %s visibly without rewriting the supplied original document",
  async (kind) => {
    const original = await sourceDocument();
    const document = {
      ...original,
      ...(kind === "malformed transport" ? { content: "{" } : {}),
      ...(kind === "unsupported schema" ? { content: writeJson({ schema: "unknown.v1" }) } : {}),
      ...(kind === "different digest" ? { sha256: "0".repeat(64) } : {}),
    };
    const before = { ...document };
    render(<SettingsDocumentInspector document={document} />);
    expect((await screen.findByRole("alert")).textContent).toContain("identity refused");
    expect(screen.queryByRole("table")).toBeNull();
    expect(document).toEqual(before);
  },
);

it.each([false, true])(
  "discarding a settings view during actual digest verification retains source and leaves no late view (refused=%s)",
  async (refused) => {
    const original = await sourceDocument();
    const document = { ...original, sha256: refused ? "0".repeat(64) : original.sha256 };
    const view = render(<SettingsDocumentInspector document={document} />);
    expect(screen.getByRole("status").textContent).toContain("Reading original settings");
    view.unmount();
    await act(async () => {
      await documentDigest(readJson(original.content) as Parameters<typeof documentDigest>[0]);
    });
    expect(view.container.childElementCount).toBe(0);
    expect(document.content).toBe(original.content);
  },
);
