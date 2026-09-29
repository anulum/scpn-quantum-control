// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace lossless JSON transport

import { canonicalBytes, dataEntries, maxDepth, maxIntegerDigits, scalarString } from "./canonical";

/** Expanded JSON capacity; compressed archive admission is a separate boundary. */
export const maxJsonBytes = 128 * 1024 * 1024;

function boundedText(text: string): void {
  if (text.length > maxJsonBytes) throw new Error("$: JSON byte limit exceeded");
  scalarString(text);
  if (new TextEncoder().encode(text).length > maxJsonBytes) throw new Error("$: JSON byte limit exceeded");
}

/** Read exact integer and float tokens, refusing duplicate keys and invalid syntax. */
export function readJson(text: string): unknown {
  boundedText(text);
  let position = 0;
  const fail = (reason: string): never => { throw new Error(`$ at ${position}: ${reason}`); };
  const whitespace = () => { while (/[ \t\r\n]/.test(text[position] ?? "")) position++; };
  const expect = (character: string) => {
    whitespace();
    if (text[position] !== character) fail(`expected ${character}`);
    position++;
  };
  const string = (): string => {
    whitespace();
    if (text[position] !== '"') fail("string required");
    const start = position++;
    let escaped = false;
    while (position < text.length) {
      const character = text[position++];
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') {
        // JSON.parse sees only a string literal, never numeric tokens or object keys.
        const decoded: unknown = JSON.parse(text.slice(start, position));
        return scalarString(decoded as string);
      }
    }
    return fail("unterminated string");
  };
  const value = (depth: number): unknown => {
    whitespace();
    const first = text[position];
    if (depth > maxDepth || (depth === maxDepth && (first === "[" || first === "{"))) fail("depth exceeded");
    if (first === '"') return string();
    if (first === "[") {
      position++;
      const items: unknown[] = [];
      whitespace();
      if (text[position] !== "]") {
        while (true) {
          items.push(value(depth + 1));
          whitespace();
          if (text[position] !== ",") break;
          position++;
        }
      }
      expect("]");
      return items;
    }
    if (first === "{") {
      position++;
      const record = Object.create(null) as Record<string, unknown>;
      whitespace();
      if (text[position] !== "}") {
        while (true) {
          const key = string();
          if (Object.hasOwn(record, key)) fail("duplicate key");
          expect(":");
          record[key] = value(depth + 1);
          whitespace();
          if (text[position] !== ",") break;
          position++;
        }
      }
      expect("}");
      return record;
    }
    for (const [token, result] of [["true", true], ["false", false], ["null", null]] as const) {
      if (text.startsWith(token, position)) { position += token.length; return result; }
    }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(position));
    if (!match) return fail("value required");
    const token = match[0];
    position += token.length;
    if (token === "-0") return -0;
    if (!/[.eE]/.test(token)) {
      if (token.replace(/^-/, "").length > maxIntegerDigits) fail("integer scalar too large");
      return BigInt(token);
    }
    const number = Number(token);
    if (!Number.isFinite(number)) fail("non-finite float");
    return number;
  };
  const parsed = value(0);
  whitespace();
  if (position !== text.length) fail("trailing input");
  return parsed;
}

function encode(value: unknown): string {
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number") {
    if (Object.is(value, -0)) return "-0.0";
    const decimal = value.toString();
    return /[.eE]/.test(decimal) ? decimal : decimal + ".0";
  }
  if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(encode).join(",") + "]";
  return "{" + dataEntries(value as object).map(([key, item]) => JSON.stringify(key) + ":" + encode(item)).join(",") + "}";
}

/** Serialize admitted values with float markers, retaining negative zero and bigint precision. */
export function writeJson(value: unknown): string {
  canonicalBytes("workspace_json.v1", value);
  const text = encode(value);
  boundedText(text);
  return text;
}
