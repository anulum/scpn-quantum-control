// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — SHA-256-bound program-AD replay against real WASM

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import committedUnit from "../../../data/studio/program_ad_replay_rational_20260714.json";
import {
  MAX_PROGRAM_AD_REPLAY_INPUT_BYTES,
  MAX_PROGRAM_AD_REPLAY_INPUTS,
  PROGRAM_AD_ARTIFACT_ID,
  PROGRAM_AD_CLAIM_BOUNDARY,
  PROGRAM_AD_INPUT_SHA256,
  PROGRAM_AD_SCHEMA,
  type KernelExports,
  type KernelReplay,
  type ProgramAdUnit,
  bindProgramAd,
  digestProgramAdInput,
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
let server: ReturnType<typeof createServer>;
let serverUrl: string;

function committed(): ProgramAdUnit {
  const parsed = parseProgramAdUnit(committedUnit);
  if (!parsed.ok) throw new Error(`committed unit failed to parse: ${parsed.reason}`);
  return parsed.value;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("test fixture is not an object");
  }
  return value as Record<string, unknown>;
}

function mutateCommitted(mutator: (root: Record<string, unknown>) => void): unknown {
  const clone = structuredClone(committedUnit) as unknown;
  const root = asRecord(clone);
  mutator(root);
  return root;
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  wasmBytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  replay = await instantiateProgramAd(wasmBytes);
  server = createServer((request, response) => {
    if (request.url === "/kernel.wasm") {
      response.writeHead(200, { "content-type": "application/wasm" });
      response.end(Buffer.from(wasmBytes));
      return;
    }
    response.writeHead(404);
    response.end("missing");
  });
  await new Promise<void>((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("test server has no port");
  serverUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await new Promise<void>((resolveClose, rejectClose) => {
    server.close((error) => (error === undefined ? resolveClose() : rejectClose(error)));
  });
});

describe("committed program-AD v2 unit", () => {
  it("recomputes value 19 and gradient [6, 2] bit-exactly through the real kernel", async () => {
    expect(programAdUnit.ok).toBe(true);
    const verdict = await verifyProgramAdUnit(committed(), replay);
    expect(verdict.display).toBe("match");
    expect(verdict.recomputed).toEqual({ value: 19, gradient: [6, 2] });
    expect(committed().expectedValue).toBe(19);
    expect(committed().expectedGradient).toEqual([6, 2]);
  });

  it("binds the exact replay bytes to the committed SHA-256", async () => {
    const bytes = hexToBytes(committed().inputHex);
    if (bytes === null) throw new Error("committed input is not hex");
    expect(await digestProgramAdInput(bytes)).toBe(PROGRAM_AD_INPUT_SHA256);
    expect(committed().inputSha256).toBe(PROGRAM_AD_INPUT_SHA256);
  });

  it("fails closed when Web Crypto SHA-256 is unavailable", async () => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
    if (descriptor === undefined || descriptor.configurable !== true) {
      throw new Error("test runtime crypto property is not configurable");
    }
    Object.defineProperty(globalThis, "crypto", { configurable: true, value: undefined });
    try {
      await expect(digestProgramAdInput(new Uint8Array([0]))).rejects.toThrow(
        "Web Crypto SHA-256 is unavailable",
      );
      expect((await verifyProgramAdUnit(committed(), replay)).reason).toBe(
        "replay input digest could not be verified",
      );
    } finally {
      Object.defineProperty(globalThis, "crypto", descriptor);
    }
  });

  it("renders mismatch when the claimed gradient or value is forged", async () => {
    const forgedGradient: ProgramAdUnit = { ...committed(), expectedGradient: [6, 99] };
    const forgedValue: ProgramAdUnit = { ...committed(), expectedValue: 42 };
    expect((await verifyProgramAdUnit(forgedGradient, replay)).display).toBe("mismatch");
    expect((await verifyProgramAdUnit(forgedValue, replay)).display).toBe("mismatch");
  });

  it("refuses a changed input even when the expected claim changes with it", async () => {
    const base = committed();
    const last = base.inputHex.endsWith("00") ? "01" : "00";
    const forged: ProgramAdUnit = {
      ...base,
      inputHex: `${base.inputHex.slice(0, -2)}${last}`,
      expectedValue: 0,
      expectedGradient: [0, 0],
    };
    const verdict = await verifyProgramAdUnit(forged, replay);
    expect(verdict.display).toBe("unverifiable");
    expect(verdict.reason).toContain("SHA-256 binding");
  });
});

describe("fail-closed canonical metadata", () => {
  const metadataTampers: ReadonlyArray<{
    readonly name: string;
    readonly unit: ProgramAdUnit;
    readonly reason: string;
  }> = [
    {
      name: "schema",
      unit: { ...committed(), schema: "studio.other.v1" },
      reason: "unknown program-AD schema",
    },
    {
      name: "artifact id",
      unit: { ...committed(), artifactId: "other-artifact" },
      reason: "unknown program-AD artifact id",
    },
    {
      name: "claim boundary",
      unit: { ...committed(), claimBoundary: "unbounded" },
      reason: "claim boundary is not canonical",
    },
    {
      name: "declared input digest",
      unit: { ...committed(), inputSha256: `sha256:${"0".repeat(64)}` },
      reason: "input digest is not canonical",
    },
    {
      name: "parameter targets",
      unit: { ...committed(), parameterTargets: ["%1", "%0"] },
      reason: "parameter targets are not canonical",
    },
    {
      name: "gradient arity",
      unit: { ...committed(), expectedGradient: [6] },
      reason: "gradient arity",
    },
    {
      name: "non-finite value",
      unit: { ...committed(), expectedValue: Number.NaN },
      reason: "expected value is not finite",
    },
    {
      name: "non-finite gradient",
      unit: { ...committed(), expectedGradient: [6, Number.POSITIVE_INFINITY] },
      reason: "expected gradient is not a finite vector",
    },
    {
      name: "non-hex input",
      unit: { ...committed(), inputHex: "nothex" },
      reason: "input payload is not valid hex",
    },
  ];

  it.each(metadataTampers)("flags $name before replay", async ({ unit, reason }) => {
    const verdict = await verifyProgramAdUnit(unit, replay);
    expect(verdict.display).toBe("unverifiable");
    expect(verdict.reason).toContain(reason);
  });

  it("uses the exact cross-language v2 constants", () => {
    expect(committed()).toMatchObject({
      schema: PROGRAM_AD_SCHEMA,
      artifactId: PROGRAM_AD_ARTIFACT_ID,
      claimBoundary: PROGRAM_AD_CLAIM_BOUNDARY,
      inputSha256: PROGRAM_AD_INPUT_SHA256,
    });
  });
});

describe("real WASM and loading boundaries", () => {
  it("rejects empty, oversized, invalid-arity, and malformed inputs", () => {
    expect(replay(new Uint8Array(), 2)).toEqual({ ok: false, code: -2 });
    expect(replay(new Uint8Array(MAX_PROGRAM_AD_REPLAY_INPUT_BYTES + 1), 2)).toEqual({
      ok: false,
      code: -2,
    });
    expect(replay(new Uint8Array([1]), 0)).toEqual({ ok: false, code: -2 });
    expect(replay(new Uint8Array([1]), 1.5)).toEqual({ ok: false, code: -2 });
    expect(replay(new Uint8Array([1]), MAX_PROGRAM_AD_REPLAY_INPUTS + 1)).toEqual({
      ok: false,
      code: -2,
    });
    const malformed = replay(new Uint8Array([0, 1, 2, 3]), 2);
    expect(malformed.ok).toBe(false);
    if (!malformed.ok) expect(malformed.code).toBeLessThan(0);
  });

  it("fetches and instantiates the deployed kernel through a real HTTP server", async () => {
    const loaded = await fetchProgramAd(`${serverUrl}/kernel.wasm`);
    expect((await verifyProgramAdUnit(committed(), loaded)).display).toBe("match");
  });

  it("throws on a real HTTP 404", async () => {
    await expect(fetchProgramAd(`${serverUrl}/missing.wasm`)).rejects.toThrow(
      "kernel fetch failed: 404",
    );
  });

  it("classifies kernel rejection, exceptions, and non-finite results", async () => {
    const rejected: KernelReplay = () => ({ ok: false, code: -5 });
    const throwing: KernelReplay = () => {
      throw new Error("kernel trap");
    };
    const nonFinite: KernelReplay = () => ({
      ok: true,
      value: Number.NaN,
      gradient: [6, 2],
    });
    expect((await verifyProgramAdUnit(committed(), rejected)).reason).toContain(
      "kernel rejected",
    );
    expect((await verifyProgramAdUnit(committed(), throwing)).reason).toBe(
      "kernel replay failed",
    );
    expect((await verifyProgramAdUnit(committed(), nonFinite)).reason).toContain("non-finite");
  });
});

describe("binding allocation boundaries", () => {
  function guardedExports(overrides: Partial<KernelExports> = {}): KernelExports {
    return {
      memory: new WebAssembly.Memory({ initial: 1 }),
      scpn_alloc: () => 16,
      scpn_free: () => undefined,
      scpn_program_ad_replay: () => 0,
      ...overrides,
    };
  }

  it("surfaces input and output allocation failures", () => {
    const inputFailure = bindProgramAd(guardedExports({ scpn_alloc: () => 0 }))(
      new Uint8Array([1]),
      2,
    );
    expect(inputFailure).toEqual({ ok: false, code: -1 });

    let calls = 0;
    const outputFailure = bindProgramAd(
      guardedExports({
        scpn_alloc: () => {
          calls += 1;
          return calls === 1 ? 16 : 0;
        },
      }),
    )(new Uint8Array([1]), 2);
    expect(outputFailure).toEqual({ ok: false, code: -1 });
  });

  it("surfaces a kernel status and frees both buffers", () => {
    const freed: Array<[number, number]> = [];
    const result = bindProgramAd(
      guardedExports({
        scpn_free: (pointer, length) => freed.push([pointer, length]),
        scpn_program_ad_replay: () => -5,
      }),
    )(new Uint8Array([1]), 2);
    expect(result).toEqual({ ok: false, code: -5 });
    expect(freed).toEqual([
      [16, 1],
      [16, 24],
    ]);
  });
});

describe("parsing and hex helpers", () => {
  it("fails closed on missing blocks and malformed scalar fields", () => {
    expect(parseProgramAdUnit(null).ok).toBe(false);
    expect(parseProgramAdUnit({ program: {}, expected: {} }).ok).toBe(false);
    expect(parseProgramAdUnit(mutateCommitted((root) => (root["input_hex"] = 123))).ok).toBe(
      false,
    );
    expect(
      parseProgramAdUnit(
        mutateCommitted((root) => (asRecord(root["expected"])["value"] = Number.NaN)),
      ).ok,
    ).toBe(false);
  });

  it("rejects empty/non-finite gradients and malformed/duplicate targets", () => {
    const mutations: Array<(root: Record<string, unknown>) => void> = [
      (root) => (asRecord(root["expected"])["gradient"] = []),
      (root) => (asRecord(root["expected"])["gradient"] = [6, Number.POSITIVE_INFINITY]),
      (root) => (asRecord(root["program"])["parameter_targets"] = []),
      (root) => (asRecord(root["program"])["parameter_targets"] = ["%0", 1]),
      (root) => (asRecord(root["program"])["parameter_targets"] = ["%0", "%0"]),
    ];
    for (const mutate of mutations) {
      expect(parseProgramAdUnit(mutateCommitted(mutate)).ok).toBe(false);
    }
  });

  it("rejects noncanonical metadata, arity, and input hex", () => {
    const mutations: Array<(root: Record<string, unknown>) => void> = [
      (root) => (root["schema"] = "other"),
      (root) => (root["artifact_id"] = "other"),
      (root) => (root["claim_boundary"] = "unbounded"),
      (root) => (root["input_sha256"] = `sha256:${"0".repeat(64)}`),
      (root) => (asRecord(root["expected"])["gradient"] = [6]),
      (root) => (root["input_hex"] = "abc"),
    ];
    for (const mutate of mutations) {
      expect(parseProgramAdUnit(mutateCommitted(mutate)).ok).toBe(false);
    }
  });

  it("rejects empty, oversized, odd-length, and non-hex strings", () => {
    expect(hexToBytes("")).toBeNull();
    expect(hexToBytes("00".repeat(MAX_PROGRAM_AD_REPLAY_INPUT_BYTES + 1))).toBeNull();
    expect(hexToBytes("abc")).toBeNull();
    expect(hexToBytes("zz")).toBeNull();
    expect(hexToBytes("00ff")).toEqual(new Uint8Array([0, 255]));
  });
});
