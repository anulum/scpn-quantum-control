// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — capability resolution truth table

import { expect, it } from "vitest";
import type { RuntimeObservation } from "../shared/contracts/capabilities";
import type { ModuleCapability, RuntimeNeed } from "../shared/contracts/module";
import { resolveCapabilities } from "./capabilityResolver";

const ok: RuntimeObservation = { status: "available", reason: "verified" };
const runtimes = (overrides: Partial<Record<RuntimeNeed, RuntimeObservation>> = {}) => ({
  "browser-kernel": ok,
  "program-ad-kernel": ok,
  "browser-storage": ok,
  "local-service": ok,
  hardware: ok,
  ...overrides,
});
const owned = (
  needs: readonly RuntimeNeed[],
  maturity: ModuleCapability["maturity"] = "stable",
) => [{ capability: { id: "probe", label: "Probe", maturity, needs }, module: "alpha" }];
const none = new Map<RuntimeNeed, string>();

it("keeps the declared maturity when every need is met, and with no needs", () => {
  expect(resolveCapabilities(owned(["browser-kernel"]), runtimes(), [], none)[0]).toEqual({
    id: "probe",
    label: "Probe",
    module: "alpha",
    state: "stable",
    reasons: [],
  });
  expect(resolveCapabilities(owned([], "experimental"), runtimes(), [], none)[0]?.state).toBe(
    "experimental",
  );
});

it("degrades on a degraded or unchecked need and names only those", () => {
  const resolved = resolveCapabilities(
    owned(["browser-kernel", "browser-storage", "program-ad-kernel"]),
    runtimes({
      "browser-storage": { status: "degraded", reason: "session only" },
      "program-ad-kernel": { status: "unknown", reason: "not checked" },
    }),
    [],
    none,
  )[0];
  expect(resolved?.state).toBe("degraded");
  expect(resolved?.reasons).toEqual([
    "browser-storage: session only",
    "program-ad-kernel: not checked",
  ]);
});

it("makes it unavailable on a missing requirement, a host refusal or an unavailable need", () => {
  const missing = [
    { module: "alpha", kind: "verb", name: "simulate" },
    { module: "other", kind: "schema", name: "x.v1" },
  ] as const;
  const refused = new Map<RuntimeNeed, string>([["hardware", "not offered"]]);
  const resolved = resolveCapabilities(
    owned(["hardware", "local-service", "browser-storage"]),
    runtimes({
      hardware: { status: "unavailable", reason: "ignored, the host refuses it first" },
      "local-service": { status: "unavailable", reason: "not connected" },
      "browser-storage": { status: "degraded", reason: "session only" },
    }),
    missing,
    refused,
  )[0];
  expect(resolved?.state).toBe("unavailable");
  expect(resolved?.reasons).toEqual([
    "the installation does not declare the verb simulate",
    "hardware: not offered",
    "local-service: not connected",
  ]);
});

it("returns one row per declaration in registry order, so none is hidden", () => {
  const two = [
    ...owned(["hardware"]),
    {
      capability: { id: "second", label: "Second", maturity: "stable", needs: [] },
      module: "beta",
    },
  ] as const;
  const refused = new Map<RuntimeNeed, string>([["hardware", "not offered"]]);
  expect(
    resolveCapabilities(two, runtimes(), [], refused).map((row) => [row.id, row.state]),
  ).toEqual([
    ["probe", "unavailable"],
    ["second", "stable"],
  ]);
});
