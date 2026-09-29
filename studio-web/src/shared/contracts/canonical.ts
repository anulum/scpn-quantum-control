// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace canonical encoding

/** Maximum nested workspace container depth. */
export const maxDepth = 64;
/** Decimal scalar capacity shared with the Python codec. */
export const maxIntegerDigits = 4096;

/** Validate Unicode scalar sequences without normalising their spelling. */
export function scalarString(value: string, path = "$"): string {
  for (const character of value) {
    const code = character.codePointAt(0)!;
    if (code >= 0xd800 && code <= 0xdfff) throw new Error(`${path}: invalid Unicode scalar`);
  }
  return value;
}

/** Inspect plain data members without invoking getters or prototype methods. */
export function dataEntries(value: object, path = "$"): [string, unknown][] {
  const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) throw new Error(`${path}: unsupported object`);
  return Reflect.ownKeys(value).map(key => {
    if (typeof key !== "string") throw new Error(`${path}: symbol key unsupported`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!("value" in descriptor) || !descriptor.enumerable) throw new Error(`${path}: data member required`);
    return [scalarString(key, path), descriptor.value as unknown];
  });
}

function byteOrder(left: string, right: string): number {
  const encoder = new TextEncoder();
  const a = encoder.encode(left);
  const b = encoder.encode(right);
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    const difference = a[index]! - b[index]!;
    if (difference !== 0) return difference;
  }
  return a.length - b.length;
}

function tag(value: unknown, path: string, depth: number, ancestors: Set<object>): unknown {
  if (depth > maxDepth || (depth === maxDepth && typeof value === "object" && value !== null)) {
    throw new Error(`${path}: depth exceeded`);
  }
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") return scalarString(value, path);
  if (typeof value === "bigint") {
    const decimal = value.toString();
    if (decimal.replace(/^-/, "").length > maxIntegerDigits) throw new Error(`${path}: integer scalar too large`);
    return ["integer", decimal];
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path}: non-finite float`);
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, false);
    return ["float64", Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("")];
  }
  if (typeof value !== "object") throw new Error(`${path}: unsupported value type`);
  if (ancestors.has(value)) throw new Error(`${path}: ancestor cycle`);
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) throw new Error(`${path}: unsupported array prototype`);
      if (Reflect.ownKeys(value).length !== value.length + 1) throw new Error(`${path}: sparse or decorated array`);
      const items: unknown[] = [];
      for (let index = 0; index < value.length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor || !("value" in descriptor)) throw new Error(`${path}: array data element required`);
        items.push(tag(descriptor.value as unknown, `${path}[${index}]`, depth + 1, ancestors));
      }
      return ["array", items];
    }
    return ["object", dataEntries(value, path).sort(([a], [b]) => byteOrder(a, b))
      .map(([key, item]) => [key, tag(item, `${path}.${key}`, depth + 1, ancestors)])];
  } finally {
    ancestors.delete(value);
  }
}

/** Encode schema + LF + compact typed JSON, preserving integer/float identity. */
export function canonicalBytes(schema: string, body: unknown): Uint8Array<ArrayBuffer> {
  if (!schema || /[\r\n]/.test(schema)) throw new Error("schema: invalid domain prefix");
  scalarString(schema, "schema");
  return new TextEncoder().encode(schema + "\n" + JSON.stringify(tag(body, "$", 0, new Set())));
}

/** Hash the complete canonical byte sequence; existing raw evidence uses its own codec. */
export async function canonicalDigest(schema: string, body: unknown): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", canonicalBytes(schema, body));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}
