// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — program-AD replay against the real WASM (ST-12)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import committedUnit from "../../../data/studio/program_ad_replay_rational_20260708.json";
import {
  type KernelExports,
  type KernelReplay,
  type ProgramAdUnit,
  bindProgramAd,
  fetchProgramAd,
  hexToBytes,
  instantiateProgramAd,
  parseProgramAdUnit,
  programAdUnit,
  verifyProgramAdUnit,
} from "./programAd";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm",
);

let wasmBytes: ArrayBuffer;
let replay: KernelReplay;

function committed(): ProgramAdUnit {
  const parsed = parseProgramAdUnit(committedUnit);
  if (!parsed.ok) throw new Error(`committed unit failed to parse: ${parsed.reason}`);
  return parsed.value;
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  wasmBytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  replay = await instantiateProgramAd(wasmBytes);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("committed program-AD unit", () => {
  it("recomputes value 19 and gradient [6, 2] bit-exactly through the real kernel", () => {
    expect(programAdUnit.ok).toBe(true);
    const verdict = verifyProgramAdUnit(committed(), replay);
    expect(verdict.display).toBe("match");
    expect(verdict.recomputed).toEqual({ value: 19, gradient: [6, 2] });
    expect(committed().expectedValue).toBe(19);
    expect(committed().expectedGradient).toEqual([6, 2]);
  });

  it("renders mismatch when the claimed gradient is forged", () => {
    const forged: ProgramAdUnit = { ...committed(), expectedGradient: [6, 99] };
    const verdict = verifyProgramAdUnit(forged, replay);
    expect(verdict.display).toBe("mismatch");
  });

  it("renders mismatch when the claimed value is forged", () => {
    const forged: ProgramAdUnit = { ...committed(), expectedValue: 42 };
    expect(verifyProgramAdUnit(forged, replay).display).toBe("mismatch");
  });
});

describe("fail-closed verification", () => {
  it("rejects an unknown schema before the kernel runs", () => {
    const unit: ProgramAdUnit = { ...committed(), schema: "studio.other.v1" };
    const verdict = verifyProgramAdUnit(unit, replay);
    expect(verdict.display).toBe("unverifiable");
    expect(verdict.reason).toContain("unknown program-AD schema");
  });

  it("rejects a non-hex input payload", () => {
    const unit: ProgramAdUnit = { ...committed(), inputHex: "nothex" };
    expect(verifyProgramAdUnit(unit, replay).display).toBe("unverifiable");
  });

  it("surfaces a kernel rejection of a malformed frozen input", () => {
    const unit: ProgramAdUnit = { ...committed(), inputHex: "00010203" };
    const verdict = verifyProgramAdUnit(unit, replay);
    expect(verdict.display).toBe("unverifiable");
    expect(verdict.reason).toContain("kernel rejected");
  });
});

describe("binding and loading edge cases", () => {
  function stubExports(overrides: Partial<KernelExports> = {}): KernelExports {
    return {
      memory: new WebAssembly.Memory({ initial: 1 }),
      scpn_alloc: () => 16,
      scpn_free: () => undefined,
      scpn_program_ad_replay: () => 0,
      ...overrides,
    };
  }

  it("surfaces an input allocation failure", () => {
    const result = bindProgramAd(stubExports({ scpn_alloc: () => 0 }))(new Uint8Array([1]), 2);
    expect(result).toEqual({ ok: false, code: -1 });
  });

  it("surfaces an output allocation failure", () => {
    let calls = 0;
    const result = bindProgramAd(
      stubExports({
        scpn_alloc: () => {
          calls += 1;
          return calls === 1 ? 16 : 0;
        },
      }),
    )(new Uint8Array([1]), 2);
    expect(result).toEqual({ ok: false, code: -1 });
  });

  it("surfaces a negative kernel status", () => {
    const result = bindProgramAd(stubExports({ scpn_program_ad_replay: () => -5 }))(
      new Uint8Array([1]),
      2,
    );
    expect(result).toEqual({ ok: false, code: -5 });
  });

  it("fetches and instantiates the deployed kernel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, arrayBuffer: async () => wasmBytes }) as Response),
    );
    const loaded = await fetchProgramAd("wasm/kernel.wasm");
    const verdict = verifyProgramAdUnit(committed(), loaded);
    expect(verdict.display).toBe("match");
  });

  it("throws when the kernel fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 502 }) as Response));
    await expect(fetchProgramAd()).rejects.toThrow("kernel fetch failed: 502");
  });
});

describe("parsing and hex helpers", () => {
  it("fails closed on a malformed artefact", () => {
    expect(parseProgramAdUnit(null).ok).toBe(false);
    expect(parseProgramAdUnit({ program: {}, expected: {} }).ok).toBe(false);
    const good = committedUnit as Record<string, unknown>;
    expect(parseProgramAdUnit({ ...good, input_hex: 123 }).ok).toBe(false);
  });

  it("rejects odd-length and non-hex strings", () => {
    expect(hexToBytes("")).toBeNull();
    expect(hexToBytes("abc")).toBeNull();
    expect(hexToBytes("zz")).toBeNull();
    expect(hexToBytes("00ff")).toEqual(new Uint8Array([0, 255]));
  });
});
