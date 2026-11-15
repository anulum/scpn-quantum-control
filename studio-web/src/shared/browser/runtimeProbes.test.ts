// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — refusals of the shipped runtime checks

import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.doUnmock("./recompute");
  vi.doUnmock("./programAd");
  vi.resetModules();
});

/** Load the probes against stand-ins for the two committed inputs and their verifiers. */
async function probesWith(unit: { ok: boolean; reason?: string }, display: string) {
  vi.resetModules();
  vi.doMock("./recompute", () => ({
    recomputeUnit: { ...unit, value: {} },
    fetchKernel: async () => ({}),
    verifyRecomputeUnit: () => ({ display }),
  }));
  vi.doMock("./programAd", () => ({
    programAdUnit: { ...unit, value: {} },
    fetchProgramAd: async () => ({}),
    verifyProgramAdUnit: async () => ({ display }),
  }));
  return import("./runtimeProbes");
}

it("refuses a committed input that failed its guard", async () => {
  const probes = await probesWith({ ok: false, reason: "guard failed" }, "match");
  await expect(probes.probeBrowserKernel()).rejects.toThrow("guard failed");
  await expect(probes.probeProgramAdKernel()).rejects.toThrow("guard failed");
});

it("refuses a kernel whose result differs from the committed input", async () => {
  const probes = await probesWith({ ok: true }, "mismatch");
  await expect(probes.probeBrowserKernel()).rejects.toThrow("Committed XY input did not verify");
  await expect(probes.probeProgramAdKernel()).rejects.toThrow(
    "Committed program-AD input did not verify",
  );
});

it("accepts a kernel that reproduces the committed input", async () => {
  const probes = await probesWith({ ok: true }, "match");
  await expect(probes.probeBrowserKernel()).resolves.toBeUndefined();
  await expect(probes.probeProgramAdKernel()).resolves.toBeUndefined();
});
