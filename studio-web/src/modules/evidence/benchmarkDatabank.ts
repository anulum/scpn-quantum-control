// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — canonical tier benchmarks, verified in the browser

import pinText from "../../../../data/studio/benchmark_databank_20261115.json?raw";
import ciText from "../../../../docs/benchmarks/tiers/kuramoto_tiers.ci.json?raw";
import localText from "../../../../docs/benchmarks/tiers/kuramoto_tiers.local.json?raw";
import type { Loaded } from "../../shared/browser/data";
import { list, record, text } from "../../shared/browser/jsonFields";
import { readJson } from "../../shared/contracts/jsonTransport";
import { pythonCanonicalText, sha256Hex } from "./pythonCanonical";

/** The two measured environments. */
export type BenchmarkEnvironment = "ci" | "local";
/** Environments in display order. */
export const benchmarkEnvironments: readonly BenchmarkEnvironment[] = ["ci", "local"];

/** One backend's outcome for an operation and size. */
export interface BackendTiming {
  /** Backend tier. */ readonly backend: string;
  /** Whether it was measured. */ readonly status: "measured" | "unavailable";
  /** Why it was not measured, or null. */ readonly reason: string | null;
  /** Median latency in microseconds, or null. */ readonly p50: number | null;
  /** 95th percentile latency in microseconds, or null. */ readonly p95: number | null;
  /** Number of timed repeats, or null. */ readonly samples: number | null;
}
/** One operation at one size. */
export interface OperationResult {
  /** Operation name. */ readonly operation: string;
  /** Problem size. */ readonly size: number;
  /** Measured backend with the lowest median, recomputed here. */ readonly fastest: string | null;
  /** Largest cross-backend deviation of the results, or null. */ readonly parity: number | null;
  /** Backend rows in recorded order. */ readonly rows: readonly BackendTiming[];
}
/** Host and toolchain of one environment. */
export type Provenance = Readonly<Record<string, string>>;
/** One verified environment. */
export interface EnvironmentRecord {
  /** Environment key. */ readonly name: BenchmarkEnvironment;
  /** Repository path of the artefact. */ readonly path: string;
  /** File SHA-256, verified here. */ readonly sha256: string;
  /** Owner's payload digest, recomputed here. */ readonly payloadSha256: string;
  /** Generation time. */ readonly generatedUtc: string;
  /** Host and toolchain fields. */ readonly provenance: Provenance;
  /** Timed repeats per measurement. */ readonly repeats: number;
  /** Results in recorded order. */ readonly results: readonly OperationResult[];
}
/** The verified databank. */
export interface BenchmarkDatabank {
  /** Pin identifier. */ readonly id: string;
  /** What the timings do and do not show. */ readonly boundary: string;
  /** Both environments. */ readonly environments: Readonly<
    Record<BenchmarkEnvironment, EnvironmentRecord>
  >;
  /** Operations present in both environments, sorted. */ readonly operations: readonly string[];
}

/** Provenance fields shown, in display order. */
export const provenanceFields = [
  "commit",
  "cpu_model",
  "cpu_count",
  "platform",
  "python",
  "numpy",
  "rustc",
  "juliacall",
  "engine",
] as const;

function integer(value: unknown, field: string): number {
  if (typeof value !== "bigint" || value < 0n)
    throw new Error(`${field} must be a non-negative integer`);
  return Number(value);
}

function latency(value: unknown, field: string): number {
  if (typeof value !== "number" || !(value > 0))
    throw new Error(`${field} must be a positive time`);
  return value;
}

function timing(raw: unknown): BackendTiming {
  const row = record(raw, "row");
  const backend = text(row["backend"], "backend");
  if (row["status"] === "unavailable") {
    if (row["stats"] !== null) throw new Error(`${backend}: an unavailable row carries no timing`);
    const reason = text(row["reason"], "reason");
    return { backend, status: "unavailable", reason, p50: null, p95: null, samples: null };
  }
  if (row["status"] !== "measured") throw new Error(`${backend}: unknown status`);
  const stats = record(row["stats"], `${backend} stats`);
  const p50 = latency(stats["p50_us"], "p50_us");
  const p95 = latency(stats["p95_us"], "p95_us");
  if (p95 < p50) throw new Error(`${backend}: P95 below P50`);
  return {
    backend,
    status: "measured",
    reason: null,
    p50,
    p95,
    samples: integer(stats["samples"], "samples"),
  };
}

/** The measured backend with the lowest median; the first one wins a tie, as in the owner. */
export function lowestMedian(rows: readonly BackendTiming[]): string | null {
  let best: BackendTiming | null = null;
  for (const row of rows)
    if (row.p50 !== null && (best === null || row.p50 < (best.p50 as number))) best = row;
  return best === null ? null : best.backend;
}

function result(raw: unknown): OperationResult {
  const row = record(raw, "result");
  const operation = text(row["operation"], "operation");
  const size = integer(row["size"], "size");
  const rows = list(row["rows"], "rows").map(timing);
  const fastest = lowestMedian(rows);
  if ((row["fastest_backend"] ?? null) !== fastest)
    throw new Error(
      `${operation} N=${size}: recorded fastest backend differs from the lowest median`,
    );
  const parity = row["parity_max_abs_diff"];
  if (parity !== null && (typeof parity !== "number" || !(parity >= 0)))
    throw new Error(`${operation} N=${size}: parity must be a non-negative number`);
  return { operation, size, fastest, parity: parity as number | null, rows };
}

async function environment(
  name: BenchmarkEnvironment,
  pin: Record<string, unknown>,
  source: string,
): Promise<EnvironmentRecord> {
  const path = text(pin["path"], `${name}.path`);
  const sha256 = text(pin["sha256"], `${name}.sha256`);
  if ((await sha256Hex(source)) !== sha256)
    throw new Error(`${path}: file digest differs from the pin`);
  const artifact = record(readJson(source), path);
  if (artifact["schema_version"] !== "scpn-quantum-control.tier-benchmark.v1")
    throw new Error(`${path}: unsupported benchmark schema`);
  if (artifact["production_claim_allowed"] !== false)
    throw new Error(`${path}: a production claim is not shown`);
  const { payload_sha256: carried, ...body } = artifact;
  const payload = await sha256Hex(pythonCanonicalText(body, true));
  if (payload !== carried || payload !== pin["payload_sha256"])
    throw new Error(`${path}: payload digest differs from its content`);
  const host = record(artifact["provenance"], "provenance");
  return {
    name,
    path,
    sha256,
    payloadSha256: payload,
    generatedUtc: text(artifact["generated_utc"], "generated_utc"),
    provenance: Object.fromEntries(provenanceFields.map((field) => [field, String(host[field])])),
    repeats: integer(record(artifact["parameters"], "parameters")["repeats"], "repeats"),
    results: list(artifact["results"], "results").map(result),
  };
}

/**
 * Verify both tier artefacts against the committed pin and read them.
 *
 * @param pinSource - Committed pin text.
 * @param sources - Artefact text per environment.
 * @returns The databank, or the reason it is unverifiable.
 */
export async function verifyDatabank(
  pinSource: string,
  sources: Readonly<Record<BenchmarkEnvironment, string>>,
): Promise<Loaded<BenchmarkDatabank>> {
  try {
    const pin = record(readJson(pinSource), "pin");
    if (pin["schema"] !== "scpn_qc_studio_benchmark_databank_v1")
      throw new Error("Unsupported databank pin");
    const pins = record(pin["sources"], "sources");
    const ci = await environment("ci", record(pins["ci"], "ci"), sources.ci);
    const local = await environment("local", record(pins["local"], "local"), sources.local);
    const operations = [...new Set(ci.results.map((item) => item.operation))].sort();
    const elsewhere = [...new Set(local.results.map((item) => item.operation))].sort();
    if (operations.join() !== elsewhere.join())
      throw new Error("The environments measure different operations");
    const value = {
      id: text(pin["artifact_id"], "artifact_id"),
      boundary: text(pin["boundary"], "boundary"),
    };
    return { ok: true, value: { ...value, environments: { ci, local }, operations } };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** Verify the committed pin and artefacts. */
export const verifyCommittedDatabank = () =>
  verifyDatabank(pinText, { ci: ciText, local: localText });
