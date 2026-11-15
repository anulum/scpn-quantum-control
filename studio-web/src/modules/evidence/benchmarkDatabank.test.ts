// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tier benchmark databank verification and its refusals

import { expect, it } from "vitest";
import pinText from "../../../../data/studio/benchmark_databank_20261115.json?raw";
import ciText from "../../../../docs/benchmarks/tiers/kuramoto_tiers.ci.json?raw";
import localText from "../../../../docs/benchmarks/tiers/kuramoto_tiers.local.json?raw";
import { readJson } from "../../shared/contracts/jsonTransport";
import { lowestMedian, verifyCommittedDatabank, verifyDatabank } from "./benchmarkDatabank";
import { pythonCanonicalText, sha256Hex } from "./pythonCanonical";

type Row = Record<string, unknown>;

/** Change the CI artefact, seal it again as its owner would, and pin the new file. */
async function resealed(change: (artifact: Row) => void) {
  const artifact = readJson(ciText) as Row;
  change(artifact);
  const { payload_sha256: _old, ...body } = artifact;
  const payload = await sha256Hex(pythonCanonicalText(body, true));
  const source = pythonCanonicalText({ ...body, payload_sha256: payload }, true);
  const pin = readJson(pinText) as { sources: { ci: Row } };
  pin.sources.ci = { ...pin.sources.ci, sha256: await sha256Hex(source), payload_sha256: payload };
  return verifyDatabank(pythonCanonicalText(pin), { ci: source, local: localText });
}
const firstResult = (artifact: Row) => (artifact["results"] as Row[])[0] as Row;
const firstRow = (artifact: Row) => (firstResult(artifact)["rows"] as Row[])[0] as Row;

it("test_benchmark_databank_01: verifies both artefacts and recomputes every fastest backend", async () => {
  const loaded = await verifyCommittedDatabank();
  if (!loaded.ok) throw new Error(loaded.reason);
  const { ci, local } = loaded.value.environments;
  expect(loaded.value.operations).toHaveLength(35);
  expect(ci.results).toHaveLength(140);
  expect(new Set(ci.results.map((item) => item.size))).toEqual(new Set([8, 32, 128, 512]));
  expect(ci.provenance["cpu_model"]).toBe("AMD EPYC 7763 64-Core Processor");
  expect(ci.provenance["python"]).toBe("3.12.13");
  expect(
    local.results.flatMap((item) => item.rows).some((row) => row.status === "unavailable"),
  ).toBe(true);
  expect(ci.repeats).toBe(5);
  expect(loaded.value.boundary).toContain("re-baseline is pending");
});

it("escapes text exactly as the tier owner's default serialisation does", async () => {
  const pin = readJson(pinText) as { ascii_oracle: { content: unknown; python_text: string }[] };
  expect(pin.ascii_oracle).toHaveLength(2);
  for (const row of pin.ascii_oracle)
    expect(pythonCanonicalText(row.content, true)).toBe(row.python_text);
  expect(await sha256Hex(pythonCanonicalText({ a: "é" }, true))).not.toBe(
    await sha256Hex(pythonCanonicalText({ a: "é" })),
  );
});

it("keeps the first of two equal medians and returns none without a measurement", () => {
  const row = (backend: string, p50: number | null) =>
    ({ backend, status: "measured", reason: null, p50, p95: p50, samples: 5 }) as const;
  expect(lowestMedian([row("rust", 2), row("julia", 1), row("python", 1)])).toBe("julia");
  expect(lowestMedian([row("rust", null)])).toBeNull();
});

it("refuses a file, payload or pin that does not match", async () => {
  const pin = readJson(pinText) as Row;
  expect(await verifyDatabank(pinText, { ci: `${ciText} `, local: localText })).toEqual({
    ok: false,
    reason: "docs/benchmarks/tiers/kuramoto_tiers.ci.json: file digest differs from the pin",
  });
  const unsealed = readJson(pinText) as { sources: { ci: Row } };
  const edited = ciText.replace(
    '"generated_utc": "2026-06-26T15:47:00Z"',
    '"generated_utc": "2026-06-27T15:47:00Z"',
  );
  unsealed.sources.ci = { ...unsealed.sources.ci, sha256: await sha256Hex(edited) };
  expect(
    await verifyDatabank(pythonCanonicalText(unsealed), { ci: edited, local: localText }),
  ).toMatchObject({
    reason: "docs/benchmarks/tiers/kuramoto_tiers.ci.json: payload digest differs from its content",
  });
  expect(
    await verifyDatabank(pythonCanonicalText({ ...pin, schema: "x" }), {
      ci: ciText,
      local: localText,
    }),
  ).toEqual({
    ok: false,
    reason: "Unsupported databank pin",
  });
});

it("refuses resealed artefacts whose content contradicts the databank rules", async () => {
  const refusals: [(artifact: Row) => void, string][] = [
    [(a) => (a["schema_version"] = "other"), "unsupported benchmark schema"],
    [(a) => (a["production_claim_allowed"] = true), "a production claim is not shown"],
    [(a) => (firstResult(a)["fastest_backend"] = "python"), "recorded fastest backend differs"],
    [(a) => (firstResult(a)["parity_max_abs_diff"] = -1), "parity must be a non-negative number"],
    [(a) => (firstRow(a)["status"] = "skipped"), "rust: unknown status"],
    [(a) => (firstRow(a)["status"] = "unavailable"), "rust: an unavailable row carries no timing"],
    [(a) => ((firstRow(a)["stats"] as Row)["p95_us"] = 0.1), "rust: P95 below P50"],
    [(a) => ((firstRow(a)["stats"] as Row)["p50_us"] = 0), "p50_us must be a positive time"],
    [
      (a) => ((firstRow(a)["stats"] as Row)["samples"] = -1n),
      "samples must be a non-negative integer",
    ],
    [
      (a) => (a["results"] = (a["results"] as Row[]).slice(4)),
      "The environments measure different operations",
    ],
  ];
  for (const [change, reason] of refusals)
    expect(await resealed(change)).toMatchObject({
      ok: false,
      reason: expect.stringContaining(reason),
    });
  const unmeasured = await resealed((a) => {
    const first = firstResult(a);
    first["fastest_backend"] = null;
    first["rows"] = (first["rows"] as Row[]).map((row) => ({
      ...row,
      status: "unavailable",
      reason: "tier excluded",
      stats: null,
    }));
  });
  expect(unmeasured.ok && unmeasured.value.environments.ci.results[0]?.fastest).toBeNull();
  const nullParity = await resealed((a) => (firstResult(a)["parity_max_abs_diff"] = null));
  expect(nullParity.ok && nullParity.value.environments.ci.results[0]?.parity).toBeNull();
});
