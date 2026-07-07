// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — browser recompute + strip-detection tests (WS-1)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import committedArtifact from "../../../data/studio/xy_compile_recompute_unit_20260708.json";
import {
  type KernelExports,
  type KernelRecompute,
  type RecomputeUnit,
  bindKernel,
  fetchKernel,
  hexToBytes,
  instantiateKernel,
  parseRecomputeUnit,
  recomputeUnit,
  verifyRecomputeUnit,
} from "./recompute";

// Resolved from the studio-web working directory (vitest cwd).
const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
);

let wasmBytes: ArrayBuffer;
let kernel: KernelRecompute;

function committedUnit(): RecomputeUnit {
  const parsed = parseRecomputeUnit(committedArtifact);
  if (!parsed.ok) {
    throw new Error(`committed unit failed to parse: ${parsed.reason}`);
  }
  return parsed.value;
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  wasmBytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  kernel = await instantiateKernel(wasmBytes);
});

describe("committed recompute unit", () => {
  it("parses and recomputes to a matching digest through the real kernel", () => {
    expect(recomputeUnit.ok).toBe(true);
    const verdict = verifyRecomputeUnit(committedUnit(), kernel);
    expect(verdict.display).toBe("match");
    expect(verdict.recomputed).toBe(committedUnit().claimedDigest);
  });
});

describe("strip detection (a tampered unit never renders match)", () => {
  const base = committedUnit();

  const tampers: ReadonlyArray<{
    readonly name: string;
    readonly unit: RecomputeUnit;
    readonly display: "mismatch" | "unverifiable";
  }> = [
    {
      name: "forged claimed digest",
      unit: { ...base, claimedDigest: `sha256:${"0".repeat(64)}` },
      display: "mismatch",
    },
    {
      name: "stripped schema",
      unit: { ...base, schema: "studio.something-else.v1" },
      display: "unverifiable",
    },
    {
      name: "downgraded verifiability mode",
      unit: { ...base, verifiabilityMode: "attestation" },
      display: "unverifiable",
    },
    {
      name: "loosened exactness grade",
      unit: { ...base, exactnessClass: "tolerance" },
      display: "unverifiable",
    },
    {
      name: "non-hex input payload",
      unit: { ...base, inputHex: "not-hex!!" },
      display: "unverifiable",
    },
    {
      name: "odd-length input payload",
      unit: { ...base, inputHex: "abc" },
      display: "unverifiable",
    },
    {
      name: "structurally invalid input the kernel rejects",
      unit: { ...base, inputHex: "0102030405060708" },
      display: "unverifiable",
    },
  ];

  it.each(tampers)("flags $name as $display, never match", ({ unit, display }) => {
    const verdict = verifyRecomputeUnit(unit, kernel);
    expect(verdict.display).toBe(display);
    expect(verdict.display).not.toBe("match");
  });

  it("detects 100% of the tamper cases", () => {
    const detected = tampers.filter(
      ({ unit }) => verifyRecomputeUnit(unit, kernel).display !== "match",
    );
    expect(detected).toHaveLength(tampers.length);
  });
});

describe("hexToBytes", () => {
  it("decodes valid lowercase hex", () => {
    expect(Array.from(hexToBytes("00ff10") ?? [])).toEqual([0, 255, 16]);
  });

  it("rejects empty, odd, and non-hex strings", () => {
    expect(hexToBytes("")).toBeNull();
    expect(hexToBytes("abc")).toBeNull();
    expect(hexToBytes("zz")).toBeNull();
  });
});

describe("parseRecomputeUnit", () => {
  it("rejects a payload without a unit block", () => {
    expect(parseRecomputeUnit({ nope: true }).ok).toBe(false);
  });

  it("rejects a unit with malformed fields", () => {
    expect(parseRecomputeUnit({ unit: { schema: 1 } }).ok).toBe(false);
  });
});

describe("bindKernel allocation failures", () => {
  function fakeExports(allocs: number[]): KernelExports {
    let call = 0;
    return {
      memory: new WebAssembly.Memory({ initial: 1 }),
      scpn_alloc: () => allocs[call++] ?? 0,
      scpn_free: () => undefined,
      scpn_xy_compile_digest: () => 0,
    };
  }

  it("fails closed when the input allocation returns null", () => {
    const recompute = bindKernel(fakeExports([0]));
    expect(recompute(new Uint8Array([1, 2]))).toEqual({ ok: false, code: -1 });
  });

  it("fails closed and frees the input when the output allocation returns null", () => {
    const freed: Array<[number, number]> = [];
    let call = 0;
    const exports: KernelExports = {
      memory: new WebAssembly.Memory({ initial: 1 }),
      scpn_alloc: () => (call++ === 0 ? 64 : 0),
      scpn_free: (ptr: number, len: number) => {
        freed.push([ptr, len]);
      },
      scpn_xy_compile_digest: () => 0,
    };
    const recompute = bindKernel(exports);
    expect(recompute(new Uint8Array([1, 2]))).toEqual({ ok: false, code: -1 });
    expect(freed).toEqual([[64, 2]]);
  });
});

describe("fetchKernel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("instantiates the kernel from a successful fetch", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(wasmBytes, { status: 200 })),
    );
    const fetched = await fetchKernel();
    const verdict = verifyRecomputeUnit(committedUnit(), fetched);
    expect(verdict.display).toBe("match");
  });

  it("throws a loud error on a failed fetch", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 404 })),
    );
    await expect(fetchKernel("wasm/missing.wasm")).rejects.toThrow(/kernel fetch failed: 404/);
  });
});
