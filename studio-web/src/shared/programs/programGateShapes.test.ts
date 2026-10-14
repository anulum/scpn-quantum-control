// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public operation arity metadata immutability

import { expect, expectTypeOf, it } from "vitest";
import { PROGRAM_GATE_SHAPES } from "./programSource";
import type { ProgramGateName } from "./programSource";

const original = {
  h: [0, 1],
  x: [0, 1],
  y: [0, 1],
  z: [0, 1],
  s: [0, 1],
  sdg: [0, 1],
  t: [0, 1],
  tdg: [0, 1],
  id: [0, 1],
  sx: [0, 1],
  sxdg: [0, 1],
  rx: [1, 1],
  ry: [1, 1],
  rz: [1, 1],
  p: [1, 1],
  u1: [1, 1],
  u2: [2, 1],
  u: [3, 1],
  u3: [3, 1],
  cx: [0, 2],
  cz: [0, 2],
  swap: [0, 2],
  rxx: [1, 2],
  ryy: [1, 2],
  rzz: [1, 2],
  measure: [0, 1],
  reset: [0, 1],
  barrier: [0, 0],
} as const;

it("retains every original operation name, arity and readonly literal type", () => {
  expect(PROGRAM_GATE_SHAPES).toEqual(original);
  expect(Object.keys(PROGRAM_GATE_SHAPES)).toEqual(Object.keys(original));
  expect(Object.isFrozen(PROGRAM_GATE_SHAPES)).toBe(true);
  expectTypeOf<typeof PROGRAM_GATE_SHAPES>().toEqualTypeOf<typeof original>();
});

it.each(Object.keys(original) as ProgramGateName[])(
  "refuses mutation of the original published %s tuple and retains its alias",
  (name) => {
    const alias = PROGRAM_GATE_SHAPES[name];
    const before = [alias[0], alias[1]] as const;
    try {
      expect(Reflect.set(alias, "0", before[0] + 1)).toBe(false);
      expect(Reflect.set(alias, "1", before[1] + 1)).toBe(false);
      expect(Reflect.set(alias, "length", 0)).toBe(false);
      expect(Reflect.set(alias, "extra", true)).toBe(false);
      expect(Reflect.defineProperty(alias, "0", { value: 99 })).toBe(false);
      expect(Object.isFrozen(alias)).toBe(true);
      expect(PROGRAM_GATE_SHAPES[name]).toBe(alias);
      expect(alias).toEqual(original[name]);
    } finally {
      // Restore the original public table if an unfrozen implementation admits the counterexample.
      Reflect.set(alias, "0", before[0]);
      Reflect.set(alias, "1", before[1]);
    }
  },
);

it("retains the public table's original tuple after an attempted outer replacement", () => {
  const alias = PROGRAM_GATE_SHAPES.h;
  expect(Reflect.set(PROGRAM_GATE_SHAPES, "h", [1, 1])).toBe(false);
  expect(PROGRAM_GATE_SHAPES.h).toBe(alias);
  expect(alias).toEqual([0, 1]);
});
