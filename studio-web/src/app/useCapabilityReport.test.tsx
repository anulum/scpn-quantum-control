// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — on-demand runtime checks and the resolved capability report

import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";
import { createModuleRegistry } from "../shared/contracts/module";
import type { StudioModule } from "../shared/contracts/module";
import type { WorkspacePersistence } from "../shared/storage/workspacePersistence";
import { workbenchRegistry } from "./moduleRegistry";
import { hostRefusals, useCapabilityReport } from "./useCapabilityReport";
import type { KernelProbes } from "./useCapabilityReport";

const resolved = async () => {};
const state = (report: ReturnType<typeof useCapabilityReport>, id: string) =>
  report.capabilities.find((capability) => capability.id === id)?.state;

it("loads no kernel until asked, then resolves every declared capability", async () => {
  const calls: string[] = [];
  const probes: KernelProbes = {
    "browser-kernel": async () => void calls.push("browser"),
    "program-ad-kernel": async () => {
      calls.push("program-ad");
      throw new Error("program-AD runtime did not load");
    },
  };
  const { result } = renderHook(() =>
    useCapabilityReport(workbenchRegistry, [], "browser", probes),
  );
  expect(calls).toEqual([]);
  expect(state(result.current, "recompute-xy-compile")).toBe("degraded");
  expect(state(result.current, "save-projects")).toBe("stable");
  expect(state(result.current, "execute-on-hardware")).toBe("unavailable");
  expect(state(result.current, "durable-local-jobs")).toBe("unavailable");
  expect(state(result.current, "verify-attestations")).toBe("stable");
  act(() => result.current.checkRuntimes());
  expect(result.current.runtimes["browser-kernel"]).toEqual({
    status: "unknown",
    reason: "checking",
  });
  await waitFor(() => expect(state(result.current, "recompute-xy-compile")).toBe("stable"));
  await waitFor(() => expect(state(result.current, "replay-program-ad")).toBe("unavailable"));
  expect(
    result.current.capabilities.find((row) => row.id === "replay-program-ad")?.reasons,
  ).toEqual(["program-ad-kernel: program-AD runtime did not load"]);
  expect(calls).toEqual(["browser", "program-ad"]);
  expect(result.current.capabilities).toHaveLength(workbenchRegistry.capabilities.length);
});

it("reads storage as opening, browser cache or this page session only", () => {
  const probes: KernelProbes = { "browser-kernel": resolved, "program-ad-kernel": resolved };
  const storage = (persistence: WorkspacePersistence | null) =>
    renderHook(() => useCapabilityReport(workbenchRegistry, [], persistence, probes)).result.current
      .runtimes["browser-storage"].status;
  expect([storage(null), storage("browser"), storage("session")]).toEqual([
    "unknown",
    "available",
    "degraded",
  ]);
  expect(hostRefusals.get("hardware")).toContain("separate gated service integration");
});

it("names a missing manifest requirement and drops results after unmounting", async () => {
  let finish: () => void = () => {};
  const probes: KernelProbes = {
    "browser-kernel": () => new Promise<void>((resolve) => (finish = resolve)),
    "program-ad-kernel": resolved,
  };
  const missing = [{ module: "workspace", kind: "verb", name: "simulate" }] as const;
  const { result, unmount } = renderHook(() =>
    useCapabilityReport(workbenchRegistry, missing, "browser", probes),
  );
  expect(result.current.capabilities.find((row) => row.id === "save-projects")?.reasons).toEqual([
    "the installation does not declare the verb simulate",
  ]);
  act(() => result.current.checkRuntimes());
  unmount();
  finish();
  await new Promise((settle) => setTimeout(settle, 10));
  expect(result.current.runtimes["browser-kernel"].reason).toBe("checking");
});

it("labels a capability without a registered owner instead of dropping it", () => {
  const module = {
    id: "alpha",
    apiVersion: "2.2",
    requires: { verbs: [], schemas: [] },
    stages: [],
    capabilities: [{ id: "probe", label: "Probe", maturity: "stable", needs: [] }],
  } as const satisfies StudioModule<object>;
  const registry = { ...createModuleRegistry([module]), contributionOwner: () => undefined };
  const { result } = renderHook(() => useCapabilityReport(registry, [], "browser"));
  expect(result.current.capabilities[0]?.module).toBe("unknown");
});
