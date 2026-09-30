// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser coverage input boundary tests

// @vitest-environment node
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { coverageObject, qualifyBrowserRecord } from "./browserRecord";

async function actualInput(): Promise<{ record: Record<string, unknown>; origin: string }> {
  const filename = process.env["STUDIO_WORKSPACE_COVERAGE"];
  if (!filename) throw new Error("Run native workspace_recovery first and supply its actual evidence path");
  const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
  if (evidence["scenario"] !== "workspace_recovery" || evidence["passed"] !== true || !Array.isArray(evidence["native_v8_coverage"]) || typeof evidence["source_url"] !== "string") throw new Error("Successful actual workspace native evidence required");
  const record = evidence["native_v8_coverage"].find(value => {
    const entry = coverageObject(value);
    const native = coverageObject(entry["coverage"]);
    return typeof native["url"] === "string" && new URL(native["url"]).pathname === "/src/shared/storage/workspaceStore.ts";
  });
  await qualifyBrowserRecord(record, process.cwd(), evidence["source_url"]);
  return { record: coverageObject(structuredClone(record)), origin: evidence["source_url"] };
}

describe("browser coverage evidence qualification", () => {
  it.each([null, [], 0, false, "counter"])("refuses non-object metadata %j", value => {
    expect(() => coverageObject(value)).toThrow("Coverage object required");
  });
  it.each([
    "https://127.0.0.1:4174/", "http://example.com:4174/", "http://127.0.0.1/",
    "http://user@127.0.0.1:4174/", "http://127.0.0.1:4174/?q=1", "http://127.0.0.1:4174/nested/",
  ])("refuses unowned source origin %s before reading any source", async origin => {
    await expect(qualifyBrowserRecord({ coverage: { url: "http://127.0.0.1:4174/src/shared/storage/workspaceStore.ts" } }, process.cwd(), origin)).rejects.toThrow("Owned root loopback source origin required");
  });
  it.each([
    "http://example.com:4174/src/shared/storage/workspaceStore.ts",
    "http://127.0.0.1:4174/src/shared/storage/workspaceStore.ts?replace=1",
    "http://user@127.0.0.1:4174/src/shared/storage/workspaceStore.ts",
    "http://127.0.0.1:4174/src/unowned.ts",
  ])("refuses redirected or unowned script %s", async url => {
    await expect(qualifyBrowserRecord({ coverage: { url } }, process.cwd(), "http://127.0.0.1:4174/")).rejects.toThrow("Unowned browser coverage source");
  });
  it("refuses a replaced executed script before trusting its source map or counters", async () => {
    await expect(qualifyBrowserRecord({ coverage: { url: "http://127.0.0.1:4174/src/shared/storage/workspaceStore.ts" }, code: "untrusted replacement", code_sha256: "invalid" }, process.cwd(), "http://127.0.0.1:4174/")).rejects.toThrow("Executed script hash mismatch");
  });
  it("refuses an actual script rebound to a different checkout source hash", async () => {
    const input = await actualInput();
    input.record["source_sha256"] = "invalid-source-identity";
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Original source hash mismatch");
  });
  it("refuses duplicate inline mappings on an otherwise actual script", async () => {
    const input = await actualInput();
    const code = input.record["code"];
    if (typeof code !== "string") throw new Error("Actual script text missing");
    const mapping = code.match(/\/\/# sourceMappingURL=data:application\/json(?:;charset=utf-8)?;base64,[A-Za-z0-9+/=]+/);
    if (!mapping) throw new Error("Actual source mapping missing");
    const changed = code + "\n" + mapping[0];
    input.record["code"] = changed;
    input.record["code_sha256"] = createHash("sha256").update(changed, "utf8").digest("hex");
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("One inline original source map required");
  });
  it("refuses a changed original source embedded in an actual script mapping", async () => {
    const input = await actualInput();
    const code = input.record["code"];
    if (typeof code !== "string") throw new Error("Actual script text missing");
    const changed = code.replace(/(\/\/# sourceMappingURL=data:application\/json(?:;charset=utf-8)?;base64,)([A-Za-z0-9+/=]+)/, (_match: string, prefix: string, encoded: string) => {
      const map = coverageObject(JSON.parse(Buffer.from(encoded, "base64").toString("utf8")) as unknown);
      map["sourcesContent"] = ["substituted source"];
      return prefix + Buffer.from(JSON.stringify(map), "utf8").toString("base64");
    });
    input.record["code"] = changed;
    input.record["code_sha256"] = createHash("sha256").update(changed, "utf8").digest("hex");
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Source map does not identify the exact original owner");
  });
  it.each(["negative-count", "outside-script", "reversed-range"])("refuses %s ranges in an actual native record", async fault => {
    const input = await actualInput();
    const native = coverageObject(input.record["coverage"]);
    if (!Array.isArray(native["functions"])) throw new Error("Actual native functions missing");
    const first = coverageObject(native["functions"][0]);
    if (!Array.isArray(first["ranges"])) throw new Error("Actual native ranges missing");
    const range = coverageObject(first["ranges"][0]);
    const code = input.record["code"];
    if (typeof code !== "string") throw new Error("Actual script text missing");
    if (fault === "negative-count") range["count"] = -1;
    else if (fault === "outside-script") range["endOffset"] = code.length + 1;
    else range["endOffset"] = range["startOffset"];
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Native coverage range outside actual script");
  });

  it("refuses missing executed text instead of admitting counters without code", async () => {
    const input = await actualInput();
    input.record["code"] = "";
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Nonempty coverage text required");
  });

  it("refuses an actual script whose function-counter inventory was removed", async () => {
    const input = await actualInput();
    coverageObject(input.record["coverage"])["functions"] = [];
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Native function counters required");
  });

  it.each(["missing-name", "missing-block-status", "missing-ranges"])("refuses %s in the actual function inventory", async fault => {
    const input = await actualInput();
    const native = coverageObject(input.record["coverage"]);
    if (!Array.isArray(native["functions"])) throw new Error("Actual native functions missing");
    const entry = coverageObject(native["functions"][0]);
    if (fault === "missing-name") delete entry["functionName"];
    else if (fault === "missing-block-status") delete entry["isBlockCoverage"];
    else entry["ranges"] = [];
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Malformed native function coverage");
  });

  it("refuses a non-string source-map symbol in an otherwise actual mapping", async () => {
    const input = await actualInput();
    const code = input.record["code"];
    if (typeof code !== "string") throw new Error("Actual script text missing");
    const changed = code.replace(/(\/\/# sourceMappingURL=data:application\/json(?:;charset=utf-8)?;base64,)([A-Za-z0-9+/=]+)/, (_match: string, prefix: string, encoded: string) => {
      const map = coverageObject(JSON.parse(Buffer.from(encoded, "base64").toString("utf8")) as unknown);
      map["names"] = [1];
      return prefix + Buffer.from(JSON.stringify(map), "utf8").toString("base64");
    });
    input.record["code"] = changed;
    input.record["code_sha256"] = createHash("sha256").update(changed, "utf8").digest("hex");
    await expect(qualifyBrowserRecord(input.record, process.cwd(), input.origin)).rejects.toThrow("Source map does not identify the exact original owner");
  });

  it("refuses a checkout that added coverage exclusions to the measured owner", async () => {
    const input = await actualInput();
    const native = coverageObject(input.record["coverage"]);
    if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
    const relative = new URL(native["url"]).pathname.slice(1);
    const directory = await mkdtemp(join(tmpdir(), "studio-excluded-owner-"));
    try {
      const original = await readFile(join(process.cwd(), relative), "utf8");
      const rejectedSource = original + "\n/* v8 ignore next */\n";
      const target = join(directory, relative);
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, rejectedSource, "utf8");
      input.record["source_sha256"] = createHash("sha256").update(rejectedSource, "utf8").digest("hex");
      await expect(qualifyBrowserRecord(input.record, directory, input.origin)).rejects.toThrow("Browser owner coverage exclusions are refused");
    } finally { await rm(directory, { recursive: true }); }
  });
});
