// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — useCatalogueRuntimes.test

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { instantiateKernel } from "../../panel/recompute";
import { useCatalogueRuntimes } from "./useCatalogueRuntimes";

const fixture = vi.hoisted(() => ({ mode: "valid" }));
vi.mock("../../panel/recompute", async importOriginal => {
  const actual = await importOriginal<typeof import("../../panel/recompute")>();
  return { ...actual, get recomputeUnit() {
    if (fixture.mode === "malformed") return { ok: false, reason: "XY input missing" };
    if (fixture.mode === "tampered" && actual.recomputeUnit.ok) {
      return { ok: true, value: { ...actual.recomputeUnit.value, claimedDigest: `sha256:${"0".repeat(64)}` } };
    }
    return actual.recomputeUnit;
  } };
});
vi.mock("../../panel/programAd", async importOriginal => {
  const actual = await importOriginal<typeof import("../../panel/programAd")>();
  return { ...actual, get programAdUnit() {
    if (fixture.mode === "malformed") return { ok: false, reason: "Program input missing" };
    if (fixture.mode === "tampered" && actual.programAdUnit.ok) {
      return { ok: true, value: { ...actual.programAdUnit.value, expectedValue: actual.programAdUnit.value.expectedValue + 1 } };
    }
    return actual.programAdUnit;
  } };
});
async function shippedKernel(url: string): Promise<Response> {
  const crate = url.includes("program_ad") ? "studio_program_ad_wasm" : "studio_wasm_kernel";
  const name = url.includes("program_ad") ? "scpn_quantum_studio_program_ad_wasm.wasm" : "scpn_quantum_studio_wasm_kernel.wasm";
  const bytes = readFileSync(resolve(`../scpn_quantum_engine/${crate}/target/wasm32-unknown-unknown/release/${name}`));
  return new Response(Uint8Array.from(bytes), { status: 200 });
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); fixture.mode = "valid"; });
function realKernel() {
  const bytes = readFileSync(resolve("../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));
  return instantiateKernel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}
describe("catalogue runtime observation", () => {
  it.each(["malformed", "tampered"])("refuses %s evidence even when real kernels are present", async mode => {
    fixture.mode = mode;
    vi.stubGlobal("fetch", shippedKernel);
    const { result } = renderHook(() => useCatalogueRuntimes());
    await waitFor(() => expect(Object.keys(result.current)).toHaveLength(2));
    expect(result.current["compile"]?.available).toBe(false);
    expect(result.current["differentiate"]?.available).toBe(false);
  });
  it("verifies both shipped inputs through the default loaders and real WASM", async () => {
    vi.stubGlobal("fetch", shippedKernel);
    const { result } = renderHook(() => useCatalogueRuntimes());
    await waitFor(() => expect(result.current["compile"]?.available).toBe(true));
    await waitFor(() => expect(result.current["differentiate"]?.available).toBe(true));
  });
  it("reports missing shipped kernels through the default public loaders", async () => {
    const { result } = renderHook(() => useCatalogueRuntimes());
    await waitFor(() => expect(Object.keys(result.current)).toHaveLength(2));
    expect(result.current["compile"]?.available).toBe(false);
    expect(result.current["differentiate"]?.available).toBe(false);
  });
  it("admits a real compiled kernel and retains missing-backend reasons", async () => {
    const probes = { compile: realKernel, differentiate: () => Promise.reject(new Error("missing optional backend")), unknown: () => Promise.reject("missing") };
    const { result } = renderHook(() => useCatalogueRuntimes(probes));
    expect(result.current).toEqual({});
    await waitFor(() => expect(result.current["compile"]?.available).toBe(true));
    expect(result.current["differentiate"]).toEqual({ available: false, reason: "missing optional backend" });
    expect(result.current["unknown"]?.reason).toBe("Kernel load failed");
  });
  it("ignores an old probe after replacement and disposal", async () => {
    let resolveOld: (value: unknown) => void = () => { throw new Error("not started"); };
    const old = { compile: () => new Promise(resolve => { resolveOld = resolve; }) };
    const replacement = { compile: realKernel };
    const { result, rerender, unmount } = renderHook(({ probes }) => useCatalogueRuntimes(probes), { initialProps: { probes: old } });
    await act(async () => { await Promise.resolve(); });
    rerender({ probes: replacement });
    expect(result.current).toEqual({});
    await waitFor(() => expect(result.current["compile"]?.available).toBe(true));
    unmount();
    await act(async () => { resolveOld(null); });
  });
});
