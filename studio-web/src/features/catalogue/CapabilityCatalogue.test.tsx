// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — CapabilityCatalogue.test

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import raw from "../../../../docs/_generated/studio_manifest.json";
import { parseManifest } from "../../panel/data";
import { CapabilityCatalogue } from "./CapabilityCatalogue";
import { parseCatalogue } from "./catalogue";

const manifest = parseManifest(raw);
if (!manifest.ok) throw new Error(manifest.reason);
const source = manifest.value;
const catalogue = parseCatalogue(raw);
if (!catalogue.ok) throw new Error(catalogue.reason);
const projection = catalogue.value;
const ready = { compile: { available: true, reason: "Kernel loaded" }, differentiate: { available: true, reason: "Kernel loaded" } };
afterEach(cleanup);

describe("capability catalogue", () => {
  it("disables missing runtimes with a reason and recovers from fresh availability", () => {
    const { rerender } = render(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={{}} />);
    expect(screen.queryByRole("link", { name: /Open XY/ })).toBeNull();
    expect(screen.getAllByText(/Backend availability unknown/)).toHaveLength(2);
    rerender(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={ready} />);
    expect(screen.getByRole("link", { name: /Open XY/ }).getAttribute("href")).toBe("#/build/compile-recompute");
    expect(within(screen.getByTestId("capability-execute")).queryByRole("link")).toBeNull();
  });
  it("intersects task, runtime and backend filters and recovers from no matches", () => {
    render(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={ready} />);
    fireEvent.change(screen.getByLabelText("Capability task"), { target: { value: "COMPILE" } });
    fireEvent.change(screen.getByLabelText("Capability runtime"), { target: { value: "browser-wasm" } });
    fireEvent.change(screen.getByLabelText("Capability backend"), { target: { value: "rust" } });
    expect(screen.getAllByTestId(/^capability-/)).toHaveLength(1);
    fireEvent.change(screen.getByLabelText("Capability backend"), { target: { value: "numpy" } });
    expect(screen.getByText("No capability matches these filters.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getAllByTestId(/^capability-/)).toHaveLength(9);
  });
  it("invalidates routes immediately when installed source identity changes", () => {
    const { rerender } = render(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={ready} />);
    rerender(<CapabilityCatalogue catalogue={projection} manifest={{ ...source, studioVersion: "0+other" }} runtimes={ready} />);
    expect(screen.getByRole("alert").textContent).toContain("Source mismatch");
    expect(screen.queryByRole("link", { name: /Open/ })).toBeNull();
    rerender(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={{ compile: { available: false, reason: "kernel fetch failed: 404" } }} />);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("kernel fetch failed: 404")).toBeTruthy();
  });
  it("keeps a valid instrument link usable when its target is not mounted yet", () => {
    render(<CapabilityCatalogue catalogue={projection} manifest={source} runtimes={ready} />);
    fireEvent.click(screen.getByRole("link", { name: /Open XY/ }));
    expect(screen.getByRole("link", { name: /Open XY/ }).getAttribute("href")).toBe("#/build/compile-recompute");
  });
  it("opens the actual mounted route and moves focus on keyboard activation", () => {
    render(<><CapabilityCatalogue catalogue={projection} manifest={source} runtimes={ready} /><section id="/build/compile-recompute" tabIndex={-1}>Compile recompute</section></>);
    const link = screen.getByRole("link", { name: /Open XY/ });
    link.focus();
    fireEvent.click(link);
    expect(document.activeElement?.id).toBe("/build/compile-recompute");
  });
});
