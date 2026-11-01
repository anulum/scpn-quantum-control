// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — complete polynomial and linear-solve WASM replay

/// <reference types="vitest/importMeta" />

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { type KernelReplay, instantiateProgramAd } from "../src/shared/browser/programAd";

const runtime = import.meta.vitest;
assert(runtime, "whole-program replay qualification requires the Vitest runtime");
const { beforeAll, describe, expect, it } = runtime;

interface CapturedCase {
  inputs: [number, number];
  effect_ir: string;
  effect_ir_sha256: string;
  input_hex: string;
  input_sha256: string;
  value: number;
  gradient: number[];
  record: { record: { program_ir_sha256: string; parameter_values: string[] } };
}

let cases: [CapturedCase, CapturedCase, CapturedCase];
let replay: KernelReplay;

function analytic([p, q]: readonly [number, number]): {
  value: number;
  gradient: [number, number];
} {
  const b0 = p * p + 2 * p + 0.5;
  const b1 = q * q + 2 * q - 1;
  const x0 = (2 * b0 - b1) / 5;
  const x1 = (3 * b1 - b0) / 5;
  const outer0 = 2 * x0 + 3;
  const outer1 = 2 * x1 + 3;
  return {
    value: x0 * x0 + x1 * x1 + 3 * (x0 + x1),
    gradient: [
      ((2 * p + 2) * (2 * outer0 - outer1)) / 5,
      ((2 * q + 2) * (3 * outer1 - outer0)) / 5,
    ],
  };
}

function close(actual: number | undefined, expected: number): void {
  assert(actual !== undefined, "replay omitted a required scalar");
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-12 + 1e-12 * Math.abs(expected));
}

function inputOf(item: CapturedCase): Uint8Array {
  return Uint8Array.from(Buffer.from(item.input_hex, "hex"));
}

function replacedIr(item: CapturedCase, ir: string): Uint8Array {
  const text = new TextEncoder().encode(ir);
  const result = new Uint8Array(8 + text.length + item.inputs.length * 8);
  const view = new DataView(result.buffer);
  view.setUint32(0, text.length, true);
  result.set(text, 4);
  view.setUint32(4 + text.length, item.inputs.length, true);
  item.inputs.forEach((value, index) => {
    view.setFloat64(8 + text.length + index * 8, value, true);
  });
  return result;
}

beforeAll(async () => {
  const output = execFileSync("python", ["examples/whole_program_replay.py"], {
    cwd: resolve(".."),
    encoding: "utf8",
    timeout: 30_000,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", OPENBLAS_NUM_THREADS: "1" },
  });
  const captured = JSON.parse(output) as { schema: string; cases: typeof cases };
  expect(captured.schema).toBe("whole_program_replay_example.v1");
  cases = captured.cases;
  expect(cases.map((item) => item.inputs)).toEqual([
    [0.7, -0.4],
    [-1.3, 0.25],
    [2, 1.5],
  ]);
  const bytes = readFileSync(
    resolve(
      "..",
      "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm",
    ),
  );
  replay = await instantiateProgramAd(Uint8Array.from(bytes).buffer);
}, 30_000);

describe("the captured complete polynomial-solve-objective program", () => {
  it("replays the exact Python IR and bindings through the public WASM host", () => {
    for (const item of cases) {
      const input = inputOf(item);
      const view = new DataView(input.buffer);
      const irSize = view.getUint32(0, true);
      expect(new TextDecoder().decode(input.slice(4, 4 + irSize))).toBe(item.effect_ir);
      expect(view.getUint32(4 + irSize, true)).toBe(item.inputs.length);
      expect(item.inputs.map((_, index) => view.getFloat64(8 + irSize + index * 8, true))).toEqual(
        item.inputs,
      );
      expect(createHash("sha256").update(item.effect_ir).digest("hex")).toBe(item.effect_ir_sha256);
      expect(item.record.record.program_ir_sha256).toBe(item.effect_ir_sha256);
      expect(
        item.record.record.parameter_values.map((hex) => Buffer.from(hex, "hex").readDoubleBE()),
      ).toEqual(item.inputs);
      expect(createHash("sha256").update(input).digest("hex")).toBe(item.input_sha256);
      expect(item.effect_ir).toContain("linalg:solve:2x2:rhs:2:");
      const original = input.slice();
      const result = replay(input, item.inputs.length);
      assert(result.ok, "complete program refused");
      const reference = analytic(item.inputs);
      close(result.value, reference.value);
      close(item.value, reference.value);
      expect(result.gradient).toHaveLength(item.inputs.length);
      for (const index of [0, 1] as const) {
        close(result.gradient[index], reference.gradient[index]);
        close(item.gradient[index], reference.gradient[index]);
      }
      expect(input).toEqual(original);
    }
  });

  it("refuses unknown IR versions and unsupported solve stages, then recovers", () => {
    const item = cases[0];
    const unknown = replacedIr(
      item,
      item.effect_ir.replace("program_ad_effect_ir.v1", "program_ad_effect_ir.v999"),
    );
    expect(replay(unknown, 2)).toEqual({ ok: false, code: -4 });
    const unsupported = replacedIr(
      item,
      item.effect_ir.replaceAll("linalg:solve:2x2:rhs:2:", "linalg:unqualified:2x2:rhs:2:"),
    );
    expect(replay(unsupported, 2)).toEqual({ ok: false, code: -5 });
    const recovered = replay(inputOf(item), 2);
    assert(recovered.ok, "recovery refused");
    close(recovered.value, analytic(item.inputs).value);
  });

  it("rejects truncated, nonfinite and wrong-arity envelopes without poisoning the instance", () => {
    const item = cases[0];
    const input = inputOf(item);
    expect(replay(input.slice(0, -1), 2)).toEqual({ ok: false, code: -2 });
    expect(replay(input, 1)).toEqual({ ok: false, code: -6 });
    const nonfinite = input.slice();
    new DataView(nonfinite.buffer).setFloat64(nonfinite.length - 8, Number.NaN, true);
    expect(replay(nonfinite, 2)).toEqual({ ok: false, code: -7 });
    const recovered = replay(input, 2);
    assert(recovered.ok, "recovery refused");
    for (const index of [0, 1] as const)
      close(recovered.gradient[index], analytic(item.inputs).gradient[index]);
  });
});
