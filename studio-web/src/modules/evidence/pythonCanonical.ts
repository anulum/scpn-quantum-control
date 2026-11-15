// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the attestation owner's canonical JSON, written in the browser

import { dataEntries, scalarString } from "../../shared/contracts/canonical";

/**
 * Spell a binary64 float as Python's `repr` does.
 *
 * Both languages print the shortest digits that read back to the same value;
 * Python writes them in fixed notation while the decimal point falls after
 * fewer than 17 digits and no more than four zeros follow it, adds `.0` to an
 * integral value and gives the exponent a sign and at least two digits.
 *
 * @param value - Finite float read from the document.
 * @returns The Python spelling, for example `0.0`, `0.0001`, `1e-05` or `1e+16`.
 * @throws Error When the value is not finite, which the owner refuses as well.
 */
export function pythonFloat(value: number): string {
  if (!Number.isFinite(value)) throw new Error("non-finite float");
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";
  const sign = value < 0 ? "-" : "";
  const [mantissa = "", exponent = ""] = Math.abs(value).toExponential().split("e");
  const digits = mantissa.replace(".", "");
  const point = Number(exponent) + 1;
  if (point > -4 && point <= 16) {
    if (point <= 0) return `${sign}0.${"0".repeat(-point)}${digits}`;
    if (point >= digits.length) return `${sign}${digits}${"0".repeat(point - digits.length)}.0`;
    return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
  }
  const body = digits.length === 1 ? digits : `${digits.slice(0, 1)}.${digits.slice(1)}`;
  const power = point - 1;
  return `${sign}${body}e${power < 0 ? "-" : "+"}${String(Math.abs(power)).padStart(2, "0")}`;
}

/** Order keys by code point, as Python sorts `str` keys; UTF-8 bytes sort the same way. */
function codePointOrder(left: string, right: string): number {
  const encoder = new TextEncoder();
  const a = encoder.encode(left);
  const b = encoder.encode(right);
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    const difference = (a[index] as number) - (b[index] as number);
    if (difference !== 0) return difference;
  }
  return a.length - b.length;
}

function encode(value: unknown, path: string, depth: number): string {
  if (depth > 64) throw new Error(`${path}: depth exceeded`);
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number") return pythonFloat(value);
  // Python escapes the same characters as JSON.stringify once no lone surrogate remains.
  if (typeof value === "string") return JSON.stringify(scalarString(value, path));
  if (Array.isArray(value)) {
    return `[${value.map((item, at) => encode(item, `${path}[${at}]`, depth + 1)).join(",")}]`;
  }
  if (typeof value !== "object") throw new Error(`${path}: unsupported value type`);
  const members = dataEntries(value, path)
    .sort(([a], [b]) => codePointOrder(a, b))
    .map(([key, item]) => `${JSON.stringify(key)}:${encode(item, `${path}.${key}`, depth + 1)}`);
  return `{${members.join(",")}}`;
}

/**
 * Write the canonical text the attestation owner hashes.
 *
 * It matches `json.dumps(value, sort_keys=True, separators=(",", ":"),
 * ensure_ascii=False, allow_nan=False)` for values read by `readJson`:
 * a `bigint` is a Python `int` and a `number` is a Python `float`.
 *
 * @param value - Value read from an attested envelope.
 * @returns Compact text with code-point ordered keys and Python float spelling.
 * @throws Error On a lone surrogate, a non-finite float or an unsupported object, where the owner refuses too.
 */
export function pythonCanonicalText(value: unknown): string {
  return encode(value, "$", 0);
}

/**
 * Strip the characters Python's `str.strip()` removes.
 *
 * @param value - Field text.
 * @param whitespace - Code points for which Python's `str.isspace()` holds.
 * @returns The text without leading and trailing Python whitespace.
 */
export function pythonStrip(value: string, whitespace: ReadonlySet<number>): string {
  const characters = Array.from(value);
  const space = (character: string | undefined) =>
    character !== undefined && whitespace.has(character.codePointAt(0) as number);
  let start = 0;
  let end = characters.length;
  while (start < end && space(characters[start])) start++;
  while (end > start && space(characters[end - 1])) end--;
  return characters.slice(start, end).join("");
}

/**
 * SHA-256 of UTF-8 text as lowercase hexadecimal.
 *
 * @param text - Canonical text.
 * @returns The 64-character digest.
 */
export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
