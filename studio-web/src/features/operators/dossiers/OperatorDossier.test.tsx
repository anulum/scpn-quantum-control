// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — reachable immutable operator review dossier
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import native from "../../../../../data/studio/operator_review_dossier.json?raw";
import OperationsView from "../../../app/routes/OperationsView";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { OperatorDossier } from "./OperatorDossier";
import * as api from "./operatorDossier";
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });
it("test_operator_review_dossiers_04: the public operations view exposes a review without a submit action", () => {
  render(<OperationsView />);
  expect(screen.getByRole("region", { name: "Operator review dossier" })).toBeTruthy();
  expect(screen.queryByRole("button", { name: /submit/i })).toBeNull();
});

function fixedTime(): void { vi.useFakeTimers({ toFake: ["Date"] }); vi.setSystemTime(new Date("2026-10-04T12:00:00Z")); }
async function changedExample(change: (body: Record<string, unknown>) => void): Promise<string> {
  type Wire = { schema: string; body: Record<string, unknown>; extensions: object; sha256: string };
  const wire = readJson(native) as Wire, source = readJson(String(wire.body["dossier_text"])) as Wire; change(source.body);
  for (const [field, identity, domain] of [["plan", "plan_sha256", "execution_plan.v1"], ["profile", "profile_sha256", "backend_profile.v1"], ["settings", "settings_sha256", "resolved_settings.v1"], ["semantic_settings", "semantic_settings_sha256", "operator_review_settings.v1"], ["policy_decision", "policy_decision_sha256", "operator_policy_decision.v1"]]) source.body[identity!] = await canonicalDigest(domain!, source.body[field!]);
  const keys = ["plan_sha256", "profile_sha256", "workload_sha256", "payload", "semantic_settings_sha256", "policy_decision_sha256", "calibration", "created_at", "expires_at"];
  source.body["execution_sha256"] = await canonicalDigest("studio.operator-review-execution.v1", Object.fromEntries(keys.map(key => [key, source.body[key]])));
  source.sha256 = await canonicalDigest(source.schema, { schema: source.schema, body: source.body, extensions: source.extensions });
  wire.body["dossier_text"] = writeJson(source); wire.body["dossier_sha256"] = source.sha256;
  wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions }); return writeJson(wire);
}
async function inspect(raw: string): Promise<void> {
  fireEvent.change(screen.getByLabelText("Operator dossier JSON"), { target: { value: raw } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect operator dossier" }));
  await screen.findByLabelText("Admitted dossier identity", {}, { timeout: 15000 });
  await waitFor(() => expect((screen.getByRole("button", { name: "Inspect operator dossier" }) as HTMLButtonElement).disabled).toBe(false), { timeout: 15000 });
}
const status = () => screen.getByLabelText("Operator review status").textContent;
const disabled = (name: string) => (screen.getByRole("button", { name }) as HTMLButtonElement).disabled;

it("test_operator_review_dossiers_01: changed execution invalidates review while malformed drafts retain original evidence", async () => {
  fixedTime(); render(<OperatorDossier />); await inspect(native);
  const identity = screen.getByLabelText("Admitted dossier identity").textContent;
  fireEvent.click(screen.getByRole("button", { name: "Approve human review" }));
  await waitFor(() => expect(status()).toBe("approved"));
  expect(screen.getByLabelText("Original human review reference").textContent).toContain(identity);
  fireEvent.change(screen.getByLabelText("Operator dossier JSON"), { target: { value: "{}" } });
  expect(status()).toBe("draft changed"); expect(disabled("Approve human review")).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Inspect operator dossier" }));
  await screen.findByRole("alert"); expect(screen.getByLabelText("Admitted dossier identity").textContent).toBe(identity);
  await inspect(await changedExample(body => { const calibration = body["calibration"] as Record<string, unknown>; calibration["sha256"] = "d".repeat(64); }));
  expect(status()).toBe("invalidated"); expect(screen.getByLabelText("Original human review reference").textContent).toContain(identity);
});

it("test_operator_review_dossiers_02: denial, expiry and refused price remain distinct and observable", async () => {
  fixedTime(); render(<OperatorDossier />); await inspect(native);
  expect(status()).toBe("pending"); fireEvent.click(screen.getByRole("button", { name: "Deny human review" }));
  await waitFor(() => expect(status()).toBe("denied"));
  fireEvent.click(screen.getByRole("button", { name: "Approve human review" })); await waitFor(() => expect(status()).toBe("approved"));
  vi.setSystemTime(new Date("2026-10-05T00:00:00Z")); await waitFor(() => expect(status()).toBe("expired"));
  expect(disabled("Approve human review")).toBe(true);
  vi.setSystemTime(new Date("2026-10-04T12:00:00Z"));
  await inspect(await changedExample(body => { const decision = body["policy_decision"] as Record<string, unknown>; decision["allowed"] = false; decision["reasons"] = ["price_unknown"]; decision["estimate"] = null; body["calibration"] = null; delete ((body["plan"] as Record<string, unknown>)["parameters"] as Record<string, unknown>)["calibration_ref"]; }));
  expect(screen.getByLabelText("Original price estimate").textContent).toBe("unknown"); expect(screen.getByLabelText("Original calibration").textContent).toBe("unknown");
  expect(disabled("Approve human review")).toBe(true);
});

it("test_operator_review_dossiers_05: a display-only import preserves the original human source reference", async () => {
  fixedTime(); render(<OperatorDossier />); fireEvent.click(screen.getByRole("button", { name: "Open dossier example" }));
  await screen.findByLabelText("Admitted dossier identity", {}, { timeout: 15000 });
  fireEvent.click(screen.getByRole("button", { name: "Approve human review" })); await waitFor(() => expect(status()).toBe("approved"));
  const original = screen.getByLabelText("Original human review reference").textContent;
  await inspect(await changedExample(body => { const settings = (body["settings"] as Record<string, unknown>)["body"] as Record<string, unknown>; for (const name of ["requested", "effective"]) (settings[name] as Record<string, unknown>)["theme"] = "light"; }));
  expect(status()).toBe("approved"); expect(screen.getByLabelText("Original human review reference").textContent).toBe(original);
});

it("pending genuine hash operations cannot accept a changed draft or update an unmounted view", async () => {
  fixedTime(); const admission = vi.spyOn(api, "parseOperatorDossier"); const record = vi.spyOn(api, "reviewDocument");
  const view = render(<OperatorDossier />); await inspect(native);
  fireEvent.click(screen.getByRole("button", { name: "Open dossier example" }));
  const pending = admission.mock.results.at(-1)!.value as ReturnType<typeof api.parseOperatorDossier>;
  expect(disabled("Inspecting dossier…")).toBe(true);
  fireEvent.change(screen.getByLabelText("Operator dossier JSON"), { target: { value: "newer draft" } });
  await act(async () => { expect((await pending).ok).toBe(true); }); expect(status()).toBe("draft changed");
  await inspect(native); fireEvent.click(screen.getByRole("button", { name: "Approve human review" }));
  const reviewing = record.mock.results.at(-1)!.value as ReturnType<typeof api.reviewDocument>;
  fireEvent.change(screen.getByLabelText("Operator dossier JSON"), { target: { value: "changed while reviewing" } });
  await act(async () => { await reviewing; }); expect(screen.queryByLabelText("Original human review reference")).toBeNull();
  await inspect(native); fireEvent.click(screen.getByRole("button", { name: "Approve human review" }));
  const afterUnmount = record.mock.results.at(-1)!.value as ReturnType<typeof api.reviewDocument>;
  view.unmount(); await act(async () => { await afterUnmount; });
  const other = render(<OperatorDossier />); fireEvent.click(screen.getByRole("button", { name: "Open dossier example" }));
  const importing = admission.mock.results.at(-1)!.value as ReturnType<typeof api.parseOperatorDossier>;
  other.unmount(); await act(async () => { expect((await importing).ok).toBe(true); });
});

it("a clock crossing expiry during an actual review hash cannot record approval", async () => {
  fixedTime(); const records = vi.spyOn(api, "reviewDocument"); render(<OperatorDossier />); await inspect(native);
  fireEvent.click(screen.getByRole("button", { name: "Approve human review" }));
  const pending = records.mock.results.at(-1)!.value as ReturnType<typeof api.reviewDocument>;
  vi.setSystemTime(new Date("2026-10-05T00:00:00Z")); await act(async () => { await pending; });
  expect(screen.queryByLabelText("Original human review reference")).toBeNull();
  await inspect(native); expect(status()).toBe("expired");
});

it("current expiry refuses a click before the periodic display timer catches up", async () => {
  fixedTime(); render(<OperatorDossier />); await inspect(native);
  vi.setSystemTime(new Date("2026-10-05T00:00:00Z"));
  fireEvent.click(screen.getByRole("button", { name: "Approve human review" }));
  expect(screen.queryByLabelText("Original human review reference")).toBeNull();
});

it("test_operator_review_dossiers_03: downloads original native source, verifier and separately referenced review", async () => {
  fixedTime(); const exports: Blob[] = []; const revoked: string[] = [];
  vi.stubGlobal("URL", class extends URL { static override createObjectURL(blob: Blob) { exports.push(blob); return "blob:review"; } static override revokeObjectURL(url: string) { revoked.push(url); } });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  try {
    render(<OperatorDossier />); expect(disabled("Export admitted dossier")).toBe(true); expect(disabled("Export native verifier")).toBe(true); expect(disabled("Export human review")).toBe(true);
    await inspect(native); fireEvent.click(screen.getByRole("button", { name: "Approve human review" })); await waitFor(() => expect(status()).toBe("approved"));
    const wire = readJson(native) as { body: { dossier_text: string; script: { source: string } } };
    for (const name of ["Export admitted dossier", "Export native verifier", "Export human review"]) fireEvent.click(screen.getByRole("button", { name }));
    expect(exports[0]!.size).toBe(new TextEncoder().encode(wire.body.dossier_text).length); expect(exports[1]!.size).toBe(new TextEncoder().encode(wire.body.script.source).length);
    expect(exports[2]!.size).toBeGreaterThan(new TextEncoder().encode(writeJson({ original_export: native })).length);
    expect(click).toHaveBeenCalledTimes(3); expect(revoked).toEqual(["blob:review", "blob:review", "blob:review"]);
  } finally { vi.unstubAllGlobals(); }
});
