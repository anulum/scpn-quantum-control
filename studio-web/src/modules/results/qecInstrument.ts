// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — QEC and future-resource instrument model over source-owned data

import committedInstrumentJson from "../../../../data/studio/qec_resource_instrument_20261031.json";
import type { Loaded } from "../../shared/browser/data";
import { count, finite, flag, list, record, text, texts } from "./jsonFields";

/** Schema of the committed instrument data. */
export const QEC_INSTRUMENT_SCHEMA = "scpn_qc_studio_qec_resource_instrument_v1";

/** One seeded toric-code decoder Monte Carlo at one distance and noise rate. */
export interface DecoderSample {
  /** Toric code distance. */
  readonly distance: number;
  /** Independent bit-flip probability per data qubit. */
  readonly physicalErrorRate: number;
  /** Trials sampled. */
  readonly shots: number;
  /** Generator seed that repeats the sample. */
  readonly seed: number;
  /** Logical failures observed. */
  readonly failures: number;
  /** Observed failure fraction. */
  readonly rate: number;
  /** Lower bound of the exact interval. */
  readonly lower: number;
  /** Upper bound of the exact interval. */
  readonly upper: number;
  /** Two-sided confidence level. */
  readonly confidenceLevel: number;
  /** Interval method named by the source. */
  readonly method: string;
}

/** The syndrome and decoding outcome of one data-qubit flip. */
export interface SyndromeSample {
  /** Toric code distance. */
  readonly distance: number;
  /** Flipped data qubit. */
  readonly dataQubit: number;
  /** Vertex checks that fire. */
  readonly defects: readonly number[];
  /** Whether matching returns the code to the trivial class. */
  readonly corrected: boolean;
}

/** One offline decoder aggregate of the readiness decision. */
export interface ReadinessAggregate {
  /** Circuit family. */
  readonly family: string;
  /** Circuit label. */
  readonly label: string;
  /** Decoder name. */
  readonly decoder: string;
  /** Noise model name. */
  readonly noiseModel: string;
  /** Logical failures observed. */
  readonly failures: number;
  /** Trials sampled. */
  readonly trials: number;
  /** Observed failure fraction. */
  readonly rate: number;
  /** Lower bound of the exact interval. */
  readonly lower: number;
  /** Upper bound of the exact interval. */
  readonly upper: number;
  /** Interval method. */
  readonly method: string;
}

/** The committed offline readiness decision. */
export interface Readiness {
  /** Repository path of the readiness artefact. */
  readonly source: string;
  /** Federated bundle schema. */
  readonly bundleSchema: string;
  /** Federated bundle digest. */
  readonly bundleDigest: string;
  /** Whether the federation gate admitted the bundle. */
  readonly admitted: boolean;
  /** Code distance of the offline comparison. */
  readonly distance: number;
  /** Decision taken before any live submission. */
  readonly decision: string;
  /** Claims the artefact supports. */
  readonly supported: readonly string[];
  /** Claims the artefact blocks. */
  readonly blocked: readonly string[];
  /** Decoder aggregates with recomputed intervals. */
  readonly aggregates: readonly ReadinessAggregate[];
  /** Always "not measured": a simulation is not hardware evidence. */
  readonly hardwareLogicalErrorReduction: "not measured";
}

/** Space and time floors of one stated overhead scenario. */
export interface OverheadScenario {
  /** Scenario label. */
  readonly label: string;
  /** Routing space factor on the data patches. */
  readonly routingOverheadFactor: number;
  /** Magic-state factories. */
  readonly factoryCount: number;
  /** Physical qubits of all factories. */
  readonly factoryPhysicalQubits: number;
  /** Routed patches plus factories. */
  readonly totalPhysicalQubits: number;
  /** Time floor of T-state production in seconds. */
  readonly tStateTimeFloorSeconds: number;
  /** Time floor of decoding latency in seconds. */
  readonly decodingTimeFloorSeconds: number;
  /** Scenario time floor in seconds. */
  readonly scenarioTimeFloorSeconds: number;
}

/** A resource estimate at one physical error rate, or the source's refusal. */
export type ResourceRow =
  | {
      /** The source estimated this rate. */
      readonly status: "estimated";
      /** Physical error rate assumed. */
      readonly physicalErrorRate: number;
      /** Digest of the base resource product. */
      readonly payloadSha256: string;
      /** Digest of the overhead qualification. */
      readonly qualificationSha256: string;
      /** Selected surface-code distance. */
      readonly codeDistance: number;
      /** Physical qubits of the data patches alone. */
      readonly dataPatchPhysicalQubits: number;
      /** T gates counted. */
      readonly totalTCount: number;
      /** Syndrome-extraction rounds. */
      readonly qecRounds: number;
      /** Syndrome time floor in seconds. */
      readonly syndromeTimeFloorSeconds: number;
      /** Union bound on logical failure. */
      readonly logicalFailureUnionBound: number;
      /** Overhead scenarios. */
      readonly scenarios: readonly OverheadScenario[];
    }
  | {
      /** The source refused this rate. */
      readonly status: "refused";
      /** Physical error rate requested. */
      readonly physicalErrorRate: number;
      /** The source's reason. */
      readonly reason: string;
    };

/** The resource section: one request across physical error rates. */
export interface Resources {
  /** The bounded problem every row estimates. */
  readonly request: Readonly<Record<string, number>>;
  /** Overheads the base estimate leaves out. */
  readonly excluded: readonly string[];
  /** Claim boundary of the base estimate. */
  readonly claimBoundary: string;
  /** Claim boundary of the overhead scenarios. */
  readonly scenarioClaimBoundary: string;
  /** One row per physical error rate. */
  readonly rows: readonly ResourceRow[];
}

/** The whole committed instrument data. */
export interface QecInstrument {
  /** Artefact identity. */
  readonly artifactId: string;
  /** What the data may be cited for. */
  readonly claimBoundary: string;
  /** Decoder simulation section. */
  readonly decoder: {
    /** Claim boundary of the decoder samples. */
    readonly claimBoundary: string;
    /** Trials per grid point. */
    readonly shots: number;
    /** Grid samples. */
    readonly experiments: readonly DecoderSample[];
    /** Single-error syndromes per distance. */
    readonly syndromes: readonly SyndromeSample[];
  };
  /** Offline readiness section. */
  readonly readiness: Readiness;
  /** Future resources section. */
  readonly resources: Resources;
}

function interval(row: Record<string, unknown>, field: string) {
  const failures = count(row["failures"], `${field}.failures`);
  const trials = count(row["trials"] ?? row["shots"], `${field}.trials`);
  const lower = finite(row["lower"], `${field}.lower`);
  const upper = finite(row["upper"], `${field}.upper`);
  if (trials < 1 || failures > trials) throw new Error(`${field} counts are inconsistent`);
  if (!(0 <= lower && lower <= upper && upper <= 1))
    throw new Error(`${field} interval must satisfy 0 <= lower <= upper <= 1`);
  if (failures === 0 && (lower !== 0 || upper === 0))
    throw new Error(`${field} with zero failures needs lower 0 and a positive upper bound`);
  return {
    failures,
    trials,
    rate: finite(row["rate"], `${field}.rate`),
    lower,
    upper,
    method: text(row["method"], `${field}.method`),
  };
}

function decoderSample(raw: unknown, index: number): DecoderSample {
  const field = `decoder.experiments[${index}]`;
  const row = record(raw, field);
  if (row["measured_logical_error_reduction_claim_allowed"] !== false)
    throw new Error(`${field} must not claim a measured logical-error reduction`);
  const counts = interval(row, field);
  return {
    distance: count(row["distance"], `${field}.distance`),
    physicalErrorRate: finite(row["physical_error_rate"], `${field}.physical_error_rate`),
    shots: counts.trials,
    seed: count(row["seed"], `${field}.seed`),
    failures: counts.failures,
    rate: counts.rate,
    lower: counts.lower,
    upper: counts.upper,
    confidenceLevel: finite(row["confidence_level"], `${field}.confidence_level`),
    method: counts.method,
  };
}

function syndrome(raw: unknown, index: number): SyndromeSample {
  const field = `decoder.single_error_syndromes[${index}]`;
  const row = record(raw, field);
  return {
    distance: count(row["distance"], `${field}.distance`),
    dataQubit: count(row["data_qubit"], `${field}.data_qubit`),
    defects: list(row["defects"], `${field}.defects`).map((item, at) =>
      count(item, `${field}.defects[${at}]`),
    ),
    corrected: flag(row["corrected"], `${field}.corrected`),
  };
}

function readiness(raw: unknown): Readiness {
  const row = record(raw, "readiness");
  if (row["hardware_logical_error_reduction"] !== "not measured")
    throw new Error("readiness must state that hardware logical-error reduction is not measured");
  return {
    source: text(row["source"], "readiness.source"),
    bundleSchema: text(row["bundle_schema"], "readiness.bundle_schema"),
    bundleDigest: text(row["bundle_digest"], "readiness.bundle_digest"),
    admitted: flag(row["admitted"], "readiness.admitted"),
    distance: count(row["distance"], "readiness.distance"),
    decision: text(row["decision"], "readiness.decision"),
    supported: texts(row["supported"], "readiness.supported"),
    blocked: texts(row["blocked"], "readiness.blocked"),
    aggregates: list(row["aggregates"], "readiness.aggregates").map((item, index) => {
      const field = `readiness.aggregates[${index}]`;
      const aggregate = record(item, field);
      return {
        family: text(aggregate["family"], `${field}.family`),
        label: text(aggregate["label"], `${field}.label`),
        decoder: text(aggregate["decoder"], `${field}.decoder`),
        noiseModel: text(aggregate["noise_model"], `${field}.noise_model`),
        ...interval(aggregate, field),
      };
    }),
    hardwareLogicalErrorReduction: "not measured",
  };
}

function scenario(raw: unknown, field: string): OverheadScenario {
  const row = record(raw, field);
  return {
    label: text(row["label"], `${field}.label`),
    routingOverheadFactor: finite(row["routing_overhead_factor"], `${field}.routing`),
    factoryCount: count(row["factory_count"], `${field}.factory_count`),
    factoryPhysicalQubits: count(row["factory_physical_qubits"], `${field}.factory_qubits`),
    totalPhysicalQubits: count(row["total_physical_qubits"], `${field}.total_qubits`),
    tStateTimeFloorSeconds: finite(row["t_state_time_floor_seconds"], `${field}.t_state`),
    decodingTimeFloorSeconds: finite(row["decoding_time_floor_seconds"], `${field}.decoding`),
    scenarioTimeFloorSeconds: finite(row["scenario_time_floor_seconds"], `${field}.scenario`),
  };
}

function resourceRow(raw: unknown, index: number): ResourceRow {
  const field = `resources.rows[${index}]`;
  const row = record(raw, field);
  const physicalErrorRate = finite(row["physical_error_rate"], `${field}.physical_error_rate`);
  if (row["status"] === "refused")
    return { status: "refused", physicalErrorRate, reason: text(row["reason"], `${field}.reason`) };
  if (row["status"] !== "estimated")
    throw new Error(`${field}.status must be estimated or refused`);
  return {
    status: "estimated",
    physicalErrorRate,
    payloadSha256: text(row["payload_sha256"], `${field}.payload_sha256`),
    qualificationSha256: text(row["qualification_sha256"], `${field}.qualification_sha256`),
    codeDistance: count(row["code_distance"], `${field}.code_distance`),
    dataPatchPhysicalQubits: count(row["data_patch_physical_qubits"], `${field}.data_qubits`),
    totalTCount: count(row["total_t_count"], `${field}.total_t_count`),
    qecRounds: count(row["qec_rounds"], `${field}.qec_rounds`),
    syndromeTimeFloorSeconds: finite(row["syndrome_time_floor_seconds"], `${field}.syndrome`),
    logicalFailureUnionBound: finite(row["logical_failure_union_bound"], `${field}.union_bound`),
    scenarios: list(row["scenarios"], `${field}.scenarios`).map((item, at) =>
      scenario(item, `${field}.scenarios[${at}]`),
    ),
  };
}

function resources(raw: unknown): Resources {
  const row = record(raw, "resources");
  const request = record(row["request"], "resources.request");
  return {
    request: Object.fromEntries(
      Object.entries(request).map(([name, value]) => [name, finite(value, `request.${name}`)]),
    ),
    excluded: texts(row["excluded_from_base_estimate"], "resources.excluded_from_base_estimate"),
    claimBoundary: text(row["claim_boundary"], "resources.claim_boundary"),
    scenarioClaimBoundary: text(row["scenario_claim_boundary"], "resources.scenario_claim"),
    rows: list(row["rows"], "resources.rows").map(resourceRow),
  };
}

/**
 * Parse the committed instrument data, refusing anything malformed.
 *
 * A sample that claims a measured logical-error reduction, a readiness section
 * that does not say the hardware reduction is unmeasured, or an interval
 * inconsistent with its counts refuses the whole artefact.
 *
 * @param raw - Decoded JSON.
 * @returns The parsed instrument or the reason it was refused.
 */
export function parseQecInstrument(raw: unknown): Loaded<QecInstrument> {
  try {
    const payload = record(raw, "artefact");
    if (payload["schema"] !== QEC_INSTRUMENT_SCHEMA)
      throw new Error(`schema must be ${QEC_INSTRUMENT_SCHEMA}`);
    const decoder = record(payload["decoder"], "decoder");
    return {
      ok: true,
      value: {
        artifactId: text(payload["artifact_id"], "artifact_id"),
        claimBoundary: text(payload["claim_boundary"], "claim_boundary"),
        decoder: {
          claimBoundary: text(decoder["claim_boundary"], "decoder.claim_boundary"),
          shots: count(decoder["shots"], "decoder.shots"),
          experiments: list(decoder["experiments"], "decoder.experiments").map(decoderSample),
          syndromes: list(decoder["single_error_syndromes"], "decoder.syndromes").map(syndrome),
        },
        readiness: readiness(payload["readiness"]),
        resources: resources(payload["resources"]),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/**
 * Distinct values of one numeric key, ascending.
 *
 * @param rows - Rows to read.
 * @param key - Numeric key.
 * @returns Sorted distinct values.
 */
export function distinct<T>(rows: readonly T[], key: (row: T) => number): number[] {
  return [...new Set(rows.map(key))].sort((left, right) => left - right);
}

/**
 * The source sample at a distance and noise rate, or null when none was run.
 *
 * @param instrument - Parsed instrument.
 * @param distance - Code distance.
 * @param rate - Physical error rate.
 * @returns The sample, unchanged.
 */
export function decoderSampleAt(
  instrument: QecInstrument,
  distance: number,
  rate: number,
): DecoderSample | null {
  return (
    instrument.decoder.experiments.find(
      (row) => row.distance === distance && row.physicalErrorRate === rate,
    ) ?? null
  );
}

/**
 * Words for an interval that state its method and the exact sample behind it.
 *
 * @param sample - Counts and bounds of one sample.
 * @param sample.failures - Logical failures observed.
 * @param sample.trials - Trials sampled.
 * @param sample.upper - Upper bound of the interval.
 * @param sample.method - Interval method named by the source.
 * @returns A sentence naming the method, the counts and, at zero failures,
 *   that the upper bound is still positive.
 */
export function intervalNote(sample: {
  readonly failures: number;
  readonly trials: number;
  readonly upper: number;
  readonly method: string;
}): string {
  const method =
    sample.method === "clopper_pearson_exact" ? "exact Clopper-Pearson" : sample.method;
  const base = `${method} interval from ${sample.failures} of ${sample.trials} trials`;
  return sample.failures === 0
    ? `${base}; with no failure observed the upper bound is still ${sample.upper.toPrecision(3)}, not zero.`
    : `${base}.`;
}

/** The committed instrument data, admitted or refused at load. */
export const qecInstrument: Loaded<QecInstrument> = parseQecInstrument(committedInstrumentJson);
