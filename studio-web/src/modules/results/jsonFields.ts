// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — strict field readers shared by the committed result instruments

/**
 * Read a JSON object.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The object.
 * @throws Error when the value is not a plain object.
 */
export function record(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${field} must be an object`);
  return value as Record<string, unknown>;
}

/**
 * Read a non-empty string.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The string.
 * @throws Error when the value is not a non-empty string.
 */
export function text(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0)
    throw new Error(`${field} must be a non-empty string`);
  return value;
}

/**
 * Read a finite number.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The number.
 * @throws Error when the value is not a finite number.
 */
export function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Error(`${field} must be a finite number`);
  return value;
}

/**
 * Read a non-negative integer.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The integer.
 * @throws Error when the value is not a non-negative integer.
 */
export function count(value: unknown, field: string): number {
  const number = finite(value, field);
  if (!Number.isInteger(number) || number < 0)
    throw new Error(`${field} must be a non-negative integer`);
  return number;
}

/**
 * Read a boolean.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The boolean.
 * @throws Error when the value is not a boolean.
 */
export function flag(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${field} must be true or false`);
  return value;
}

/**
 * Read a JSON array.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The array.
 * @throws Error when the value is not an array.
 */
export function list(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${field} must be a list`);
  return value;
}

/**
 * Read an array of non-empty strings.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The strings.
 * @throws Error when the value or an item is malformed.
 */
export function texts(value: unknown, field: string): string[] {
  return list(value, field).map((item, index) => text(item, `${field}[${index}]`));
}

/**
 * Read an array of finite numbers.
 *
 * @param value - Decoded value.
 * @param field - Location used in the error.
 * @returns The numbers.
 * @throws Error when the value or an item is malformed.
 */
export function numbers(value: unknown, field: string): number[] {
  return list(value, field).map((item, index) => finite(item, `${field}[${index}]`));
}
