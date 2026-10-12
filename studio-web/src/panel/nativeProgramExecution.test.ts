// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — real Program AD WASM execution contracts

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

import committedUnit from "../../../data/studio/program_ad_replay_rational_20260714.json";
import {
  type KernelExports,
  type KernelReplay,
  hexToBytes,
  instantiateProgramAd,
} from "./programAd";

let wasmBytes: ArrayBuffer;
let replay: KernelReplay;

/** Copy the unchanged committed envelope, replacing only its two scalar inputs. */
function rationalInput(x: number, y: number): Uint8Array {
  const input = hexToBytes(committedUnit.input_hex);
  if (input === null) throw new Error("committed rational input is invalid");
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
  const irLength = view.getUint32(0, true);
  expect(view.getUint32(4 + irLength, true)).toBe(2);
  view.setFloat64(8 + irLength, x, true);
  view.setFloat64(16 + irLength, y, true);
  return input;
}

beforeAll(async () => {
  const buffer = readFileSync(
    resolve(
      "..",
      "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm",
    ),
  );
  wasmBytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  replay = await instantiateProgramAd(wasmBytes);
});

describe("executed Program AD WASM ABI", () => {
  it.each([
    [3, 5, 19, 6],
    [-2, 1, 6, -4],
    [0.5, -4, -7.75, 1],
  ])("executes f(%s,%s) with independent rational value and gradient", (x, y, value, dx) => {
    const input = rationalInput(x, y);
    const before = input.slice();
    expect(replay(input, 2)).toEqual({ ok: true, value, gradient: [dx, 2] });
    expect(input).toEqual(before);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 1e308])(
    "refuses nonfinite input or overflowing output %s and then recovers",
    (bad) => {
      const input = rationalInput(bad, 5);
      const before = input.slice();
      expect(replay(input, 2).ok).toBe(false);
      expect(input).toEqual(before);
      expect(replay(rationalInput(3, 5), 2)).toEqual({ ok: true, value: 19, gradient: [6, 2] });
    },
  );

  it.each([0, 1, 3, -1, 1.5, Number.MAX_SAFE_INTEGER])(
    "rejects incorrect output arity %s without corrupting subsequent execution",
    (arity) => {
      const input = rationalInput(3, 5);
      const before = input.slice();
      expect(replay(input, arity).ok).toBe(false);
      expect(input).toEqual(before);
      expect(replay(input, 2)).toEqual({ ok: true, value: 19, gradient: [6, 2] });
    },
  );

  it("rejects truncated and inconsistent input envelopes with no caller-buffer writes", () => {
    const original = rationalInput(3, 5);
    const inconsistent = original.slice();
    const view = new DataView(inconsistent.buffer);
    view.setUint32(4 + view.getUint32(0, true), 3, true);
    for (const input of [new Uint8Array(), original.slice(0, -1), inconsistent]) {
      const before = input.slice();
      expect(replay(input, 2).ok).toBe(false);
      expect(input).toEqual(before);
    }
    expect(replay(original, 2)).toEqual({ ok: true, value: 19, gradient: [6, 2] });
  });

  it("the real raw export leaves a sentinel output untouched on malformed input", async () => {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    const exports = instance.exports as unknown as KernelExports;
    const input = rationalInput(3, 5);
    const inputPtr = exports.scpn_alloc(input.length);
    const outputPtr = exports.scpn_alloc(24);
    expect(inputPtr).not.toBe(0);
    expect(outputPtr).not.toBe(0);
    try {
      new Uint8Array(exports.memory.buffer, inputPtr, input.length).set(input);
      new Uint8Array(exports.memory.buffer, outputPtr, 24).fill(0xa5);
      const status = exports.scpn_program_ad_replay(inputPtr, input.length - 1, outputPtr, 24);
      expect(status).not.toBe(0);
      expect(Array.from(new Uint8Array(exports.memory.buffer, outputPtr, 24))).toEqual(
        Array(24).fill(0xa5),
      );
      expect(exports.scpn_program_ad_replay(inputPtr, input.length, outputPtr, 24)).toBe(0);
      const output = new DataView(exports.memory.buffer, outputPtr, 24);
      expect([
        output.getFloat64(0, true),
        output.getFloat64(8, true),
        output.getFloat64(16, true),
      ]).toEqual([19, 6, 2]);
    } finally {
      exports.scpn_free(inputPtr, input.length);
      exports.scpn_free(outputPtr, 24);
    }
  });

  it("retained execution closures survive repeated allocate/free cycles", async () => {
    const retained = await instantiateProgramAd(wasmBytes.slice(0));
    for (let index = 0; index < 32; index += 1) {
      expect(retained(rationalInput(-2, 1), 2)).toEqual({ ok: true, value: 6, gradient: [-4, 2] });
      expect(retained(rationalInput(0.5, -4), 2)).toEqual({
        ok: true,
        value: -7.75,
        gradient: [1, 2],
      });
    }
  });

  it("absent compiled bytes and compiler text cannot qualify as an executed WASM module", async () => {
    await expect(instantiateProgramAd(new ArrayBuffer(0))).rejects.toThrow();
    await expect(
      instantiateProgramAd(new TextEncoder().encode("module { func.func @value() }")),
    ).rejects.toThrow();
    expect(replay(rationalInput(3, 5), 2)).toEqual({ ok: true, value: 19, gradient: [6, 2] });
  });
});
