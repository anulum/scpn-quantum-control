// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace lossless JSON tests

// @vitest-environment node
import { describe, expect, it } from "vitest";
import corpus from "../../../../tests/data/studio_workspace/transport.json";
import { canonicalBytes, canonicalDigest } from "./canonical";
import { readJson, writeJson } from "./jsonTransport";

describe("lossless workspace JSON boundary", () => {
  it.each(corpus.cases)("preserves the $id transport contract", async row => {
    if (row.expectation === "reject") {
      expect(() => readJson(row.input_json)).toThrow();
      return;
    }
    const value = readJson(row.input_json);
    expect(canonicalBytes("test.v1", readJson(writeJson(value)))).toEqual(canonicalBytes("test.v1", value));
    if ("expected_digest" in row && row.expected_digest !== undefined) {
      expect(await canonicalDigest(row.schema!, value)).toBe(row.expected_digest);
    }
    if (row.id === "negative_zero_token") expect(Object.is(value, -0)).toBe(true);
    if (row.id === "prototype_named_key") expect(Object.getPrototypeOf(value)).toBe(null);
  });
  it.each(["", "NaN", "Infinity", "[", "{", "{1:2}", '{"a" 1}', '{"a":1,}', "true false", '"\\q"', '"unterminated', "01", "1.", "--1", "\u00a0null"])("refuses invalid JSON %s", input => {
    expect(() => readJson(input)).toThrow();
  });
  it("preserves primitive, nested and escaped values without caller aliasing", () => {
    const value = [null, true, false, [], {}, 1n, 1, -0, 1e-20, "quote\" slash\\\n\t😀"];
    const encoded = writeJson(value);
    value.push("later");
    expect(writeJson(readJson(encoded))).toBe(encoded);
    expect(readJson(" \n\t[1, 2.0]\r ")).toEqual([1n, 2]);
  });
  it("bounds depth and numeric token size", () => {
    expect(() => readJson("[".repeat(65) + "0" + "]".repeat(65))).toThrow(/depth/);
    expect(() => readJson("1".repeat(4097))).toThrow(/integer/);
    expect(() => readJson('"\ud800"')).toThrow(/Unicode/);
    expect(() => writeJson({ bad: undefined })).toThrow();
  });
});

it.each(['[1,]', '{"key":1,}', '{1:2}', '"unterminated', '"bad\\x"', '01', '1e9999', '\u00a0true'])("does not extend JSON grammar: %s", text => {
  expect(() => readJson(text)).toThrow();
});
it("bounds integer tokens before conversion", () => {
  const decimal = "9".repeat(4096);
  expect(writeJson(readJson(decimal))).toBe(decimal);
  expect(() => readJson(decimal + "9")).toThrow("integer scalar too large");
});
it.each([["x", 128 * 1024 * 1024 + 1], ["é", 64 * 1024 * 1024 + 1]] as const)("bounds UTF-8 input for %s", (character, count) => {
  expect(() => readJson(character.repeat(count))).toThrow("byte limit");
}, 30_000);
