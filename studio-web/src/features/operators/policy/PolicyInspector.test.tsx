// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native operator verdict inspection tests

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import native from "../../../../../data/studio/operator_policy_decisions.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import * as policyApi from "./policyDecision";
import { PolicyInspector } from "./PolicyInspector";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

type Wire = {
  schema: string; body: {
    decision: { allowed: boolean; reasons: string[]; rejected_substitutions: string[]; assessed_at: string;
      request: Record<string, unknown>; estimate: Record<string, unknown> | null };
    settings: { body: { requested: Record<string, unknown>; effective: Record<string, unknown> } };
    settings_sha256: string;
  }; extensions: unknown; sha256: string;
};

async function wireCase(field?: string, value?: unknown, reasons: string[] = []): Promise<string> {
  const wire = readJson(native) as Wire;
  wire.body.decision.allowed = reasons.length === 0;
  wire.body.decision.reasons = reasons;
  wire.body.decision.estimate!["amount"] = field === "cost" ? value : "12.50";
  if (field !== undefined && field !== "cost") {
    wire.body.decision.request[field === "device" ? "target" : field] = value;
    wire.body.settings.body.requested[field] = value;
    wire.body.settings.body.effective[field] = value;
  }
  wire.body.settings_sha256 = await canonicalDigest("resolved_settings.v1", wire.body.settings);
  wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions });
  return writeJson(wire);
}

async function inspect(raw: string): Promise<void> {
  fireEvent.change(screen.getByLabelText("Operator policy JSON"), { target: { value: raw } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect policy decision" }));
  await screen.findByRole("button", { name: "Inspect policy decision" });
}

describe("native operator policy decisions", () => {
  it("test_operator_policy_decisions_04: displays the core refusal and preserves it after malformed import", async () => {
    render(<PolicyInspector />);
    fireEvent.click(screen.getByRole("button", { name: "Open policy example" }));
    expect((await screen.findByLabelText("Core policy verdict")).textContent).toContain("refused");
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toContain("price_unknown");
    fireEvent.change(screen.getByLabelText("Operator policy JSON"), { target: { value: "{}" } });
    fireEvent.click(screen.getByRole("button", { name: "Inspect policy decision" }));
    expect((await screen.findByRole("alert")).textContent).toContain("refused");
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toContain("price_unknown");
  });

  it("test_operator_policy_decisions_01: exact requested values and shot/cost refusals stay observable", async () => {
    render(<PolicyInspector />);
    await inspect(await wireCase());
    expect(screen.getByLabelText("Core policy verdict").textContent).toBe("allowed plan");
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toBe("none");
    expect(screen.getByLabelText("Rejected substitutions").textContent).toBe("none");
    expect(screen.getByLabelText("Estimated cost").textContent).toBe("12.50");
    await inspect(await wireCase("shots", 1025n, ["shots_ceiling"]));
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toBe("shots_ceiling");
    expect(screen.getByRole("table", { name: "Operator requested and effective settings" }).textContent).toContain("1025");
    await inspect(await wireCase("cost", "12.500000001", ["cost_ceiling"]));
    expect(screen.getByLabelText("Estimated cost").textContent).toBe("12.500000001");
    expect(screen.getByLabelText("Core policy verdict").textContent).toBe("refused plan");
  });

  it("test_operator_policy_decisions_02: the original region and rejected binding remain visible", async () => {
    render(<PolicyInspector />);
    const wire = readJson(await wireCase("region", "us-east1", ["profile_region_mismatch", "region_forbidden"])) as Wire;
    wire.body.decision.rejected_substitutions = ["profile_region_mismatch"];
    wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions });
    await inspect(writeJson(wire));
    expect(screen.getByLabelText("Rejected substitutions").textContent).toBe("profile_region_mismatch");
    expect(screen.getByRole("table", { name: "Operator requested and effective settings" }).textContent).toContain("us-east1");
  });

  it("test_operator_policy_decisions_03: an expired source verdict never becomes an execute action", async () => {
    render(<PolicyInspector />);
    await inspect(await wireCase(undefined, undefined, ["policy_expired"]));
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toBe("policy_expired");
    expect(screen.getByText(/HAL rechecks current policy/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /submit|execute/i })).toBeNull();
    const wire = readJson(native) as Wire;
    wire.body.decision.estimate = null;
    wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions });
    await inspect(writeJson(wire));
    expect(screen.getByLabelText("Estimated cost").textContent).toBe("unknown");
  });

  it("discarding an actual pending admission after edit or unmount preserves prior state", async () => {
    const admission = vi.spyOn(policyApi, "parseOperatorDecision");
    const view = render(<PolicyInspector />);
    await inspect(native);
    fireEvent.click(screen.getByRole("button", { name: "Open policy example" }));
    const pending = admission.mock.results.at(-1)!.value as ReturnType<typeof policyApi.parseOperatorDecision>;
    expect((screen.getByRole("button", { name: "Inspecting policy…" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Operator policy JSON"), { target: { value: "newer draft" } });
    await act(async () => { expect((await pending).ok).toBe(true); });
    expect(screen.getByLabelText("Policy refusal reasons").textContent).toBe("price_unknown");
    expect((screen.getByLabelText("Operator policy JSON") as HTMLTextAreaElement).value).toBe("newer draft");
    fireEvent.click(screen.getByRole("button", { name: "Open policy example" }));
    const unmounted = admission.mock.results.at(-1)!.value as ReturnType<typeof policyApi.parseOperatorDecision>;
    view.unmount();
    await act(async () => { expect((await unmounted).ok).toBe(true); });
  });

  it("exports admitted bytes even while a different draft is refused, then releases its URL", async () => {
    const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
    const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
    let exported: Blob | undefined;
    const create = vi.fn((blob: Blob) => { exported = blob; return "blob:owned-policy"; });
    const revoke = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { value: create, configurable: true });
    Object.defineProperty(URL, "revokeObjectURL", { value: revoke, configurable: true });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    try {
      render(<PolicyInspector />);
      expect((screen.getByRole("button", { name: "Export admitted policy" }) as HTMLButtonElement).disabled).toBe(true);
      await inspect(native);
      await inspect("{}");
      fireEvent.click(screen.getByRole("button", { name: "Export admitted policy" }));
      expect(create).toHaveBeenCalledOnce();
      expect(exported!.size).toBe(new TextEncoder().encode(native).length);
      expect(click).toHaveBeenCalledOnce();
      expect(revoke).toHaveBeenCalledWith("blob:owned-policy");
    } finally {
      if (originalCreate) Object.defineProperty(URL, "createObjectURL", originalCreate); else Reflect.deleteProperty(URL, "createObjectURL");
      if (originalRevoke) Object.defineProperty(URL, "revokeObjectURL", originalRevoke); else Reflect.deleteProperty(URL, "revokeObjectURL");
    }
  });
});
