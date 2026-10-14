// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered settings views reject stale host source bindings

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { requirePresent } from "../../test-support/requirePresent";
import type { ModuleDocument } from "../shared/contracts/module";
import type { WorkbenchServices } from "../shared/contracts/workbench";
import { ModuleDocumentViews } from "../shared/moduleViews/ModuleDocumentViews";
import { declaredBy, workbenchRegistry } from "./moduleRegistry";
import { studioManifest } from "../shared/browser/data";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(["settings", "editors"] as const)(
  "keeps the original registered %s source when a host retains an older document through edit and re-admission",
  async (slot) => {
    const plan = await stationaryPlan();
    const observed: { services: WorkbenchServices | null; document: ModuleDocument | null } = {
      services: null,
      document: null,
    };
    render(
      <RegisteredExperimentRuntime
        plan={plan}
        observeServices={(services, document) => {
          observed.services = services;
          observed.document = document;
        }}
      />,
    );
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Original workspace state" }).textContent,
      ).not.toContain("Opening browser"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
    fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
    await waitFor(() => expect(observed.document).not.toBeNull());
    const original = requirePresent(observed.document);
    const before = { ...original };
    const view = (document: ModuleDocument) => (
      <ModuleDocumentViews
        registry={workbenchRegistry}
        slot={slot}
        services={requirePresent(observed.services)}
        document={document}
        declared={declaredBy(studioManifest)}
      />
    );
    const settings = render(view(original));
    if (slot === "settings")
      await screen.findByRole("table", { name: "Requested and effective settings" });
    else await screen.findByRole("region", { name: "Linked parameter editor" });
    const current = `${plan.sourceJson}\n`;
    fireEvent.change(screen.getByLabelText("Original archive bytes"), {
      target: { value: current },
    });
    settings.rerender(view(original));
    expect(screen.getByRole("alert").textContent).toContain("source changed");
    expect(screen.queryByRole("table")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
    await waitFor(() => expect(requirePresent(observed.document).content).toBe(current));
    settings.rerender(view(original));
    expect(screen.getByRole("alert").textContent).toContain("source changed");
    const latest = requirePresent(observed.document);
    expect(latest.sha256).not.toBe(original.sha256);
    // A host that updates only the digest must not relabel the older content.
    settings.rerender(view({ ...original, sha256: latest.sha256 }));
    expect(screen.getByRole("alert").textContent).toContain("source changed");
    expect(original).toEqual(before);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      current,
    );
  },
);
