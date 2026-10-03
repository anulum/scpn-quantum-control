// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — frozen kernel envelope and actual binary digest tests

// @vitest-environment node
import { createHash } from "node:crypto";
import { runInNewContext } from "node:vm";
import { expect, it } from "vitest";
import { kernelBinaryFingerprint, ownedWorkerBinary, ownedWorkerVector, readWorkerEvent, readWorkerRequest, workerData, workerDigest, workerRunId } from "./kernelProtocol";

const request = { version: 1, run_id: "source-run", revision_hash: "a".repeat(64), plan_hash: "b".repeat(64), command: "run", payload: null };
const event = { version: 1, run_id: "source-run", sequence: 1, kind: "progress", payload: { revision_hash: request.revision_hash, plan_hash: request.plan_hash } };

it("preserves every frozen outer field and hashes the exact original binary bytes", async () => {
  expect(readWorkerRequest(request)).toEqual(request);
  expect(readWorkerEvent(event)).toEqual(event);
  const bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
  expect(await kernelBinaryFingerprint(bytes)).toBe(createHash("sha256").update(bytes).digest("hex"));
  expect(bytes).toEqual(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
  expect(workerDigest("0".repeat(64))).toBe("0".repeat(64));
  expect(workerRunId("α".repeat(128))).toHaveLength(128);
});

it("refuses unknown versions, command kinds, fields and malformed bounded identities", () => {
  for (const value of [null, [], 1, { ...request, extra: true }, { ...request, version: 2 }, { ...request, command: 1 }, { ...request, command: "launch" }, { ...request, run_id: "" }, { ...request, revision_hash: "A".repeat(64) }, { ...request, plan_hash: 1 }]) expect(() => readWorkerRequest(value)).toThrow();
  for (const value of [null, [], { ...event, version: 2 }, { ...event, kind: 1 }, { ...event, kind: "success" }, { ...event, sequence: 0 }, { ...event, sequence: 1.5 }, { ...event, sequence: "1" }, { ...event, payload: null }, { ...event, payload: [] }, { ...event, payload: 1 }]) expect(() => readWorkerEvent(value)).toThrow();
  for (const value of [null, [], 1, {}, { value: 1, other: 2 }]) expect(() => workerData(value, ["value"])).toThrow();
  for (const value of [null, 1, "", "a".repeat(63), "A".repeat(64)]) expect(() => workerDigest(value)).toThrow();
  for (const value of [null, 1, "", "x".repeat(129), "x\n", "x\u007f"]) expect(() => workerRunId(value)).toThrow();
});

it("rejects accessor payloads and unsupported prototypes without executing a getter", () => {
  let reads = 0;
  const payload = Object.defineProperty({}, "revision_hash", { enumerable: true, get() { reads++; return request.revision_hash; } });
  expect(() => readWorkerEvent({ ...event, payload })).toThrow("data member");
  expect(() => workerData(new Date(), [])).toThrow("unsupported object");
  expect(readWorkerEvent({ ...event, payload: Object.assign(Object.create(null) as object, event.payload) })).toEqual(event);
  expect(reads).toBe(0);
});

it("preserves native cross-realm dtype while refusing shared buffers and tag spoofing", () => {
  expect(ownedWorkerVector(runInNewContext("new Float64Array([0.2, 0.3])") as unknown)).toBe(true);
  expect(ownedWorkerBinary(runInNewContext("new Uint8Array([0, 1])") as unknown)).toBe(true);
  expect(ownedWorkerVector(new Float64Array(new SharedArrayBuffer(16)))).toBe(false);
  expect(ownedWorkerBinary(new Uint8Array(new SharedArrayBuffer(8)))).toBe(false);
  expect(ownedWorkerVector(new Float32Array(2))).toBe(false);
  expect(ownedWorkerVector({ [Symbol.toStringTag]: "Float64Array", length: 2 })).toBe(false);
  const decorated = new Uint8Array(8);
  Object.defineProperty(decorated, Symbol.toStringTag, { value: "Float64Array" });
  expect(ownedWorkerVector(decorated)).toBe(false);
});
