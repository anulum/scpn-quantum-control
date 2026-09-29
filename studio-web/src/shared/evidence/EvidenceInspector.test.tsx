// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence inspector lifecycle tests

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeAll, expect, it } from "vitest";
import { EvidenceInspector } from "./EvidenceInspector";
import type { EvidenceVerification } from "./EvidenceInspector";
import { projectEvidenceBundle, projectProgramAdEvidence } from "./evidence";
import { instantiateProgramAd, programAdUnit, verifyProgramAdUnit } from "../../panel/programAd";
import type { KernelReplay } from "../../panel/programAd";

let kernel: KernelReplay;
function unit() {
  if (!programAdUnit.ok) throw new Error(programAdUnit.reason);
  return programAdUnit.value;
}
beforeAll(async () => {
  const bytes = readFileSync(resolve("..", "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm"));
  kernel = await instantiateProgramAd(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
});
afterEach(cleanup);
async function verify(): Promise<EvidenceVerification> {
  const verdict = await verifyProgramAdUnit(unit(), kernel);
  return { display: verdict.display, detail: "Original bounded replay: " + verdict.display };
}
it("drops a real result when only the input revision changes while pending", async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const pendingResult = gate.then(verify);
  const pending = () => pendingResult;
  const view = projectProgramAdEvidence(unit());
  const { rerender } = render(<EvidenceInspector view={view} inputRevision="A" verify={pending} />);
  fireEvent.click(screen.getByRole("button"));
  expect(screen.getByRole("region").getAttribute("data-verification-phase")).toBe("running");
  rerender(<EvidenceInspector view={view} inputRevision="B" verify={verify} />);
  await act(async () => { release(); await pendingResult; });
  expect(screen.queryByRole("status")).toBeNull();
  fireEvent.click(screen.getByRole("button"));
  await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Original bounded replay: match"));
});
it("clears a completed verdict on a same-ID source or presentation change", async () => {
  const view = projectProgramAdEvidence(unit());
  const { rerender } = render(<EvidenceInspector view={view} inputRevision="A" verify={verify} />);
  fireEvent.click(screen.getByRole("button"));
  await screen.findByRole("status");
  rerender(<EvidenceInspector view={{ ...view, freshness: "changed declaration" }} inputRevision="A" verify={verify} />);
  expect(screen.queryByRole("status")).toBeNull();
});
it("keeps sealed falsification and freshness unchanged after a real bounded replay", async () => {
  const source = { schema: "studio.evidence-replay.v1", evidence_kind: "falsified",
    prov: { entity: { id: "synthetic-falsification-metadata", digest: "sha256:" + "a".repeat(64) } },
    claim_boundary: { status: "refuted", admission: "rejected" }, freshness: "traceable-unchecked",
    attestation: { signature: "synthetic-unverified-signature" } };
  render(<EvidenceInspector view={projectEvidenceBundle(source)} inputRevision="A" verify={verify} />);
  fireEvent.click(screen.getByRole("button"));
  await screen.findByRole("status");
  expect(screen.getByText("falsified")).toBeTruthy();
  expect(screen.getByText("refuted")).toBeTruthy();
  expect(screen.getByText("traceable-unchecked")).toBeTruthy();
  expect(screen.getByText("Seal present — not verified")).toBeTruthy();
  expect(screen.getByRole("status").className).not.toContain("validated");
});
it.each([null, {}, { attestation: [] }, { schema: "unknown" }])("shows unsupported evidence without a success verdict: %j", source => {
  render(<EvidenceInspector view={projectEvidenceBundle(source)} inputRevision="A" />);
  expect(screen.getByText("Partial or unsupported evidence")).toBeTruthy();
  expect(screen.getByText("Verification is unavailable for this evidence format.")).toBeTruthy();
  expect(screen.queryByRole("button")).toBeNull();
  expect(screen.queryByRole("status")).toBeNull();
});
it("disables verification of an unrepresentable snapshot", () => {
  render(<EvidenceInspector view={projectEvidenceBundle({ value: Infinity })} inputRevision="A" verify={verify} />);
  expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(true);
  expect(within(screen.getByRole("list")).getByText(/cannot be represented/)).toBeTruthy();
});
it.each([new Error("source unavailable"), "offline"])("renders verifier failure without changing source axes: %s", error => {
  render(<EvidenceInspector view={projectProgramAdEvidence(unit())} inputRevision="A" verify={async () => { throw error; }} />);
  fireEvent.click(screen.getByRole("button"));
  return waitFor(() => expect(screen.getByRole("alert").textContent).toContain("unverifiable"));
});
it("shows a real numerical mismatch as verification disagreement", async () => {
  const changed = { ...unit(), expectedGradient: [6, 99] };
  const verifyChanged = async (): Promise<EvidenceVerification> => {
    const verdict = await verifyProgramAdUnit(changed, kernel);
    return { display: verdict.display, detail: verdict.reason ?? verdict.display };
  };
  render(<EvidenceInspector view={projectProgramAdEvidence(changed)} inputRevision="changed" verify={verifyChanged} />);
  fireEvent.click(screen.getByRole("button"));
  await waitFor(() => expect(screen.getByRole("status").getAttribute("data-verdict")).toBe("mismatch"));
});
