// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the browser canonical form against the Python owner's text

import { expect, it } from "vitest";
import { readJson } from "../../shared/contracts/jsonTransport";
import { attestationInstrument } from "./attestationInstrument";
import { pythonCanonicalText, pythonFloat, pythonStrip, sha256Hex } from "./pythonCanonical";

const committed = () => {
  if (!attestationInstrument.ok) throw new Error(attestationInstrument.reason);
  return attestationInstrument.value;
};

it("spells floats as Python's repr does, in both notations", () => {
  const spelled = [0, -0, 1, -7.25, 0.1, 0.0001, 1e-5, 1.5e-7, 1e15, 1e16, 2.5e22, 5e-324];
  expect(spelled.map(pythonFloat)).toEqual([
    "0.0",
    "-0.0",
    "1.0",
    "-7.25",
    "0.1",
    "0.0001",
    "1e-05",
    "1.5e-07",
    "1000000000000000.0",
    "1e+16",
    "2.5e+22",
    "5e-324",
  ]);
  expect(pythonFloat(123456789.125)).toBe("123456789.125");
  expect(pythonFloat(1.7976931348623157e308)).toBe("1.7976931348623157e+308");
  expect(() => pythonFloat(Number.NaN)).toThrow("non-finite float");
  expect(() => pythonFloat(Number.POSITIVE_INFINITY)).toThrow("non-finite float");
});

it("writes the owner's exact text and digest for every committed oracle sample", async () => {
  const rows = committed().oracle;
  expect(rows.map((row) => row.label)).toEqual([
    "edge floats",
    "seeded random floats",
    "integers",
    "strings",
    "key order",
    "nesting and literals",
  ]);
  for (const row of rows) {
    const text = pythonCanonicalText({ claim_axes: {}, content: row.content });
    expect(text).toBe(row.pythonText);
    expect(await sha256Hex(text)).toBe(row.digest);
  }
});

it("keeps integers and floats apart and sorts keys by code point", () => {
  expect(
    pythonCanonicalText(readJson('{"b":[1,1.0,-0.0,null,true,false],"a":10000000000000000000000}')),
  ).toBe('{"a":10000000000000000000000,"b":[1,1.0,-0.0,null,true,false]}');
  expect(pythonCanonicalText({ "\u{1d538}": 2n, "￿": 1n })).toBe('{"￿":1,"\u{1d538}":2}');
  expect(pythonCanonicalText({ q: 'a"b\\c\n\u0001\u007f ' })).toBe(
    '{"q":"a\\"b\\\\c\\n\\u0001\u007f "}',
  );
});

it("refuses what the owner cannot serialise", () => {
  expect(() => pythonCanonicalText({ value: "\ud800" })).toThrow("invalid Unicode scalar");
  expect(() => pythonCanonicalText({ value: Number.NaN })).toThrow("non-finite float");
  expect(() => pythonCanonicalText({ value: undefined })).toThrow(
    "$.value: unsupported value type",
  );
  expect(() => pythonCanonicalText({ value: new Date(0) })).toThrow("unsupported object");
  let deep: unknown = 1n;
  for (let level = 0; level < 66; level++) deep = [deep];
  expect(() => pythonCanonicalText(deep)).toThrow("depth exceeded");
});

it("strips exactly the code points Python strips", () => {
  const whitespace = committed().whitespace;
  expect(whitespace.has(0x1c) && whitespace.has(0x85) && whitespace.has(0x3000)).toBe(true);
  expect(whitespace.has(0x200b) || whitespace.has(0xfeff)).toBe(false);
  expect(pythonStrip("\u001c　 claim \u0085\t", whitespace)).toBe("claim");
  expect(pythonStrip("​claim﻿", whitespace)).toBe("​claim﻿");
  expect(pythonStrip("　　", whitespace)).toBe("");
  expect(pythonStrip("", whitespace)).toBe("");
});
