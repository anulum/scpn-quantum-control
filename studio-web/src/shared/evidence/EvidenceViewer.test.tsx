// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — public evidence viewer tests

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import committedText from "../../../../data/studio/program_ad_replay_rational_20260714.json?raw";
import { EvidenceViewer } from "./index";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function inspect(text: string) {
  fireEvent.change(screen.getByLabelText("Evidence JSON"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect snapshot" }));
}
it("inspects source metadata, refuses bad JSON and recovers without network access", () => {
  render(<EvidenceViewer />);
  expect(screen.queryByRole("region", { name: "Evidence inspector" })).toBeNull();
  inspect('{"schema":"studio.evidence-replay.v1","evidence_kind":"falsified","claim_boundary":{"status":"refuted"}}');
  expect(screen.getByText("refuted")).toBeTruthy();
  inspect('{"schema":');
  expect(screen.getByRole("alert").textContent).toContain("Cannot inspect evidence");
  expect(screen.queryByText("refuted")).toBeNull();
  inspect('{}');
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByText("Missing schema")).toBeTruthy();
});
it("replays the committed source through the original parser and real WASM", async () => {
  const bytes = readFileSync(resolve("..", "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm"));
  const content = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  vi.stubGlobal("fetch", async () => new Response(content, { status: 200 }));
  render(<EvidenceViewer />);
  inspect(committedText);
  const inspector = screen.getByRole("region", { name: "Evidence inspector" });
  fireEvent.click(within(inspector).getByRole("button"));
  await waitFor(() => expect(within(inspector).getByRole("status").getAttribute("data-verdict")).toBe("match"));
  inspect('{}');
  expect(screen.queryByRole("status")).toBeNull();
});
it("routes malformed supported replay records to their owning refusal", () => {
  render(<EvidenceViewer />);
  inspect('{"schema":"scpn_qc_studio_program_ad_replay_v2"}');
  expect(screen.getByRole("alert").textContent).toContain("Cannot replay evidence");
});
