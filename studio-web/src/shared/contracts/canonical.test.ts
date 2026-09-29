// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace canonical conformance tests

// @vitest-environment node
import { describe, expect, it } from "vitest";
import corpus from "../../../../tests/data/studio_workspace/canonical.json";
import { canonicalBytes, canonicalDigest } from "./canonical";

function materialise(raw: unknown): unknown {
  const descriptor = raw as Record<string, unknown>;
  switch (descriptor["kind"]) {
    case "null": return null;
    case "integer": return BigInt(String(descriptor["decimal"]));
    case "float64": {
      const bytes = Uint8Array.from(String(descriptor["bits"]).match(/../g) ?? [], x => parseInt(x, 16));
      return new DataView(bytes.buffer).getFloat64(0, false);
    }
    case "array": return (descriptor["items"] as unknown[]).map(materialise);
    case "object": return Object.fromEntries((descriptor["entries"] as [string, unknown][]).map(([key, value]) => [key, materialise(value)]));
    case "string_codepoints": return String.fromCodePoint(...(descriptor["hex"] as string[]).map(x => parseInt(x, 16)));
    default: return descriptor["value"];
  }
}

describe("workspace canonical public codec", () => {
  it.each(corpus.cases)("matches the explicit $id oracle", async raw => {
    const value = materialise(raw.input_descriptor);
    if ("expected" in raw) expect(() => canonicalBytes(raw.schema, value)).toThrow();
    else {
      const bytes = canonicalBytes(raw.schema, value);
      expect(Array.from(bytes, x => x.toString(16).padStart(2, "0")).join("")).toBe(raw.expected_canonical_utf8_hex);
      expect(await canonicalDigest(raw.schema, value)).toBe(raw.expected_sha256);
    }
  });
  it("allows repeated aliases but refuses cycles and excessive depth", () => {
    const shared: unknown[] = [1n];
    expect(canonicalBytes("test.v1", [shared, shared])).toEqual(canonicalBytes("test.v1", [[1n], [1n]]));
    shared.push(shared);
    expect(() => canonicalBytes("test.v1", shared)).toThrow(/cycle/);
    let nested: unknown = 1n;
    for (let index = 0; index < 65; index++) nested = [nested];
    expect(() => canonicalBytes("test.v1", nested)).toThrow(/depth/);
  });
  it.each([undefined, NaN, Infinity, -Infinity, Symbol("x"), () => 1, new Date(), new Map(), { x: undefined }])("refuses unsupported value %s", value => {
    expect(() => canonicalBytes("test.v1", value)).toThrow();
  });
  it.each(["", "line\nbreak", "line\rreturn", "\ud800"])("refuses invalid domain %s", schema => {
    expect(() => canonicalBytes(schema, null)).toThrow();
  });
  it("refuses accessors, symbol keys, sparse arrays and invalid Unicode keys", () => {
    let invoked = false;
    const accessor = { get x() { invoked = true; return 1; } };
    for (const value of [accessor, { [Symbol("key")]: 1 }, new Array(2), { "\udfff": null }]) {
      expect(() => canonicalBytes("test.v1", value)).toThrow();
    }
    expect(invoked).toBe(false);
  });
  it("retains domain separation and refuses integers beyond the scalar budget", async () => {
    expect(await canonicalDigest("one.v1", {})).not.toBe(await canonicalDigest("two.v1", {}));
    expect(() => canonicalBytes("test.v1", 10n ** 4096n)).toThrow(/integer/);
  });
});

it("bounds integer scalars without rounding their admitted digits", () => {
  const decimal = "9".repeat(4096);
  expect(new TextDecoder().decode(canonicalBytes("example.v1", BigInt(decimal)))).toBe('example.v1\n["integer","' + decimal + '"]');
  expect(() => canonicalBytes("example.v1", 10n ** 4096n)).toThrow("integer scalar too large");
});
it("rejects custom array prototypes before a serializer can invoke their methods", () => {
  class Decorated extends Array<unknown> {}
  expect(() => canonicalBytes("example.v1", new Decorated(1, 2))).toThrow("unsupported array prototype");
});
