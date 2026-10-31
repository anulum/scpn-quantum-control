// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — mitigation and readout instrument model over source-owned data

import committedInstrumentJson from "../../../../data/studio/mitigation_instrument_20261031.json";
import type { Loaded } from "../../shared/browser/data";
import { count, finite, list, numbers, record, text, texts } from "./jsonFields";

/** Schema of the committed instrument data. */
export const MITIGATION_INSTRUMENT_SCHEMA = "scpn_qc_studio_mitigation_instrument_v1";

/** The physical qubit one classical bit reads. */
export interface BitMapping {
  /** Classical bit index; bit 0 is the rightmost label character. */
  readonly classicalBit: number;
  /** Physical qubit read into that bit. */
  readonly qubit: number;
}

/** Stated readout flip probabilities of one measured qubit. */
export interface QubitReadout {
  /** Physical qubit index. */
  readonly qubit: number;
  /** Probability that a prepared 0 reads as 1. */
  readonly zeroToOne: number;
  /** Probability that a prepared 1 reads as 0. */
  readonly oneToZero: number;
}

/** A calibration matrix with its shot accounting. */
export interface Calibration {
  /** Rows are observed labels, columns prepared labels, both in label order. */
  readonly matrix: readonly (readonly number[])[];
  /** Shots of each prepared state, in label order. */
  readonly shotsByPreparedState: readonly number[];
  /** Spectral condition number measured by the source. */
  readonly conditionNumber: number;
}

/** Signed corrected values with separate standard errors, in label order. */
export interface Corrected {
  /** Signed inverse estimates; they need not be probabilities. */
  readonly quasiprobabilities: readonly number[];
  /** Standard error from the measurement counts. */
  readonly samplingStandardError: readonly number[];
  /** Standard error from the calibration counts. */
  readonly calibrationStandardError: readonly number[];
  /** Both errors combined. */
  readonly totalStandardError: readonly number[];
  /** Sum of the signed estimates. */
  readonly sum: number;
  /** Labels whose estimate is negative. */
  readonly negativeLabels: readonly string[];
}

/** Fields every readout case carries, corrected or refused. */
interface ReadoutCaseBase {
  /** Stable identifier. */
  readonly caseId: string;
  /** What the case demonstrates. */
  readonly title: string;
  /** Label order of every vector in the case. */
  readonly labels: readonly string[];
  /** Physical qubit of each classical bit. */
  readonly measurementMap: readonly BitMapping[];
  /** Device qubits that are not measured. */
  readonly unmeasuredQubits: readonly number[];
  /** Stated flip probabilities of the measured qubits. */
  readonly readout: readonly QubitReadout[];
  /** Raw counts in label order; never replaced by a correction. */
  readonly rawCounts: readonly number[];
  /** Raw frequencies in label order. */
  readonly rawFrequencies: readonly number[];
  /** Measurement shots. */
  readonly shots: number;
  /** Calibration of the case. */
  readonly calibration: Calibration;
}

/** One readout case: corrected by the source, or refused with its reason. */
export type ReadoutCase =
  | (ReadoutCaseBase & {
      /** The source corrected the counts. */
      readonly status: "corrected";
      /** Signed corrected values. */
      readonly corrected: Corrected;
    })
  | (ReadoutCaseBase & {
      /** The source refused the calibration. */
      readonly status: "refused";
      /** The source's reason. */
      readonly reason: string;
    });

/** One noise scale of the extrapolation case. */
export interface ZnePoint {
  /** Noise amplification factor. */
  readonly noiseScale: number;
  /** Raw counts in label order. */
  readonly rawCounts: readonly number[];
  /** Measurement shots. */
  readonly shots: number;
  /** Observable from the raw frequencies. */
  readonly rawExpectation: number;
  /** Observable from the corrected values. */
  readonly correctedExpectation: number;
  /** Standard error of the corrected observable. */
  readonly correctedStandardError: number;
}

/** The zero-noise extrapolation case. */
export interface Zne {
  /** What the case may be cited for. */
  readonly claimBoundary: string;
  /** Observable estimated at each scale. */
  readonly observable: string;
  /** Label order of the counts. */
  readonly labels: readonly string[];
  /** Physical qubit of each classical bit. */
  readonly measurementMap: readonly BitMapping[];
  /** Shared calibration of every scale. */
  readonly calibration: Calibration;
  /** One point per noise scale. */
  readonly points: readonly ZnePoint[];
  /** The source's extrapolation. */
  readonly extrapolation: {
    /** Extrapolated zero-noise value. */
    readonly zeroNoiseEstimate: number;
    /** Its standard error. */
    readonly standardError: number;
    /** Interval lower bound. */
    readonly low: number;
    /** Interval upper bound. */
    readonly high: number;
    /** Nominal coverage of the interval. */
    readonly coverage: number;
    /** Polynomial degree. */
    readonly order: number;
    /** Root-mean-square fit residual. */
    readonly fitResidual: number;
    /** Uncertainty method named by the source. */
    readonly method: string;
  };
}

/** One measured raw-versus-corrected parity comparison. */
export interface HardwarePair {
  /** Dataset name. */
  readonly dataset: string;
  /** Compared sectors. */
  readonly comparison: string;
  /** Circuit depth. */
  readonly depth: number;
  /** Raw left value. */
  readonly leftRaw: number;
  /** Raw right value. */
  readonly rightRaw: number;
  /** Corrected left value. */
  readonly leftCorrected: number;
  /** Corrected right value. */
  readonly rightCorrected: number;
  /** Raw relative asymmetry. */
  readonly rawRelative: number;
  /** Corrected relative asymmetry. */
  readonly correctedRelative: number;
  /** Left samples. */
  readonly nLeft: number;
  /** Right samples. */
  readonly nRight: number;
}

/** The measured hardware section. */
export interface Hardware {
  /** Repository path of the summary. */
  readonly source: string;
  /** Federated bundle schema. */
  readonly bundleSchema: string;
  /** Federated bundle digest. */
  readonly bundleDigest: string;
  /** Whether the federation gate admitted the bundle. */
  readonly admitted: boolean;
  /** Correction method of the summary. */
  readonly method: string;
  /** Always false: the summary has no full confusion matrix. */
  readonly fullConfusionMatrixAvailable: false;
  /** The summary's own caveat. */
  readonly caveat: string;
  /** Measured pairs. */
  readonly pairs: readonly HardwarePair[];
}

/** The whole committed instrument data. */
export interface MitigationInstrument {
  /** Artefact identity. */
  readonly artifactId: string;
  /** What the data may be cited for. */
  readonly claimBoundary: string;
  /** How labels read classical bits. */
  readonly labelOrder: string;
  /** Readout correction section. */
  readonly readout: {
    /** Claim boundary of the stated cases. */
    readonly claimBoundary: string;
    /** Correction method. */
    readonly method: string;
    /** The cases. */
    readonly cases: readonly ReadoutCase[];
  };
  /** Extrapolation section. */
  readonly zne: Zne;
  /** Measured hardware section. */
  readonly hardware: Hardware;
}

function sized(value: unknown, field: string, length: number): number[] {
  const items = numbers(value, field);
  if (items.length !== length) throw new Error(`${field} must have ${length} entries`);
  return items;
}

function counts(value: unknown, field: string, length: number): number[] {
  return sized(value, field, length).map((item, index) => count(item, `${field}[${index}]`));
}

function labelsOf(value: unknown, field: string, width: number): string[] {
  const labels = texts(value, field);
  if (labels.length !== 2 ** width || new Set(labels).size !== labels.length)
    throw new Error(`${field} must list every one of the ${2 ** width} labels once`);
  if (labels.some((label) => label.length !== width || /[^01]/.test(label)))
    throw new Error(`${field} must be binary strings of width ${width}`);
  return labels;
}

function measurementMap(value: unknown, field: string): BitMapping[] {
  const map = list(value, field).map((item, index) => {
    const row = record(item, `${field}[${index}]`);
    const classicalBit = count(row["classical_bit"], `${field}[${index}].classical_bit`);
    if (classicalBit !== index) throw new Error(`${field} must list classical bits in order`);
    return { classicalBit, qubit: count(row["qubit"], `${field}[${index}].qubit`) };
  });
  if (map.length === 0 || new Set(map.map((item) => item.qubit)).size !== map.length)
    throw new Error(`${field} must read distinct qubits`);
  return map;
}

function calibration(value: unknown, field: string, size: number): Calibration {
  const row = record(value, field);
  const matrix = list(row["matrix"], `${field}.matrix`).map((item, index) =>
    sized(item, `${field}.matrix[${index}]`, size),
  );
  if (matrix.length !== size) throw new Error(`${field}.matrix must be ${size} square`);
  return {
    matrix,
    shotsByPreparedState: counts(row["shots_by_prepared_state"], `${field}.shots`, size),
    conditionNumber: finite(row["condition_number"], `${field}.condition_number`),
  };
}

function corrected(value: unknown, field: string, labels: readonly string[]): Corrected {
  const row = record(value, field);
  const size = labels.length;
  const quasiprobabilities = sized(row["quasiprobabilities"], `${field}.values`, size);
  const negativeLabels = texts(row["negative_labels"], `${field}.negative_labels`);
  const negative = labels.filter((_, index) => (quasiprobabilities[index] as number) < 0);
  if (negative.join() !== negativeLabels.join())
    throw new Error(`${field}.negative_labels must name exactly the negative values`);
  return {
    quasiprobabilities,
    samplingStandardError: sized(row["sampling_standard_error"], `${field}.sampling`, size),
    calibrationStandardError: sized(
      row["calibration_standard_error"],
      `${field}.calibration`,
      size,
    ),
    totalStandardError: sized(row["total_standard_error"], `${field}.total`, size),
    sum: finite(row["sum"], `${field}.sum`),
    negativeLabels,
  };
}

function readoutCase(raw: unknown, index: number): ReadoutCase {
  const field = `readout.cases[${index}]`;
  const row = record(raw, field);
  const map = measurementMap(row["measurement_map"], `${field}.measurement_map`);
  const labels = labelsOf(row["labels"], `${field}.labels`, map.length);
  const rawCounts = counts(row["raw_counts"], `${field}.raw_counts`, labels.length);
  const shots = count(row["shots"], `${field}.shots`);
  if (rawCounts.reduce((total, value) => total + value, 0) !== shots)
    throw new Error(`${field}.raw_counts must add up to the shots`);
  const base: ReadoutCaseBase = {
    caseId: text(row["case_id"], `${field}.case_id`),
    title: text(row["title"], `${field}.title`),
    labels,
    measurementMap: map,
    unmeasuredQubits: list(row["unmeasured_qubits"], `${field}.unmeasured`).map((item, at) =>
      count(item, `${field}.unmeasured[${at}]`),
    ),
    readout: list(row["readout"], `${field}.readout`).map((item, at) => {
      const flips = record(item, `${field}.readout[${at}]`);
      return {
        qubit: count(flips["qubit"], `${field}.readout[${at}].qubit`),
        zeroToOne: finite(flips["zero_to_one"], `${field}.readout[${at}].zero_to_one`),
        oneToZero: finite(flips["one_to_zero"], `${field}.readout[${at}].one_to_zero`),
      };
    }),
    rawCounts,
    rawFrequencies: sized(row["raw_frequencies"], `${field}.raw_frequencies`, labels.length),
    shots,
    calibration: calibration(row["calibration"], `${field}.calibration`, labels.length),
  };
  if (row["status"] === "refused")
    return { ...base, status: "refused", reason: text(row["reason"], `${field}.reason`) };
  if (row["status"] !== "corrected")
    throw new Error(`${field}.status must be corrected or refused`);
  return { ...base, status: "corrected", corrected: corrected(row["corrected"], field, labels) };
}

function zne(raw: unknown): Zne {
  const row = record(raw, "zne");
  const map = measurementMap(row["measurement_map"], "zne.measurement_map");
  const labels = labelsOf(row["labels"], "zne.labels", map.length);
  const fit = record(row["extrapolation"], "zne.extrapolation");
  const low = finite(fit["low"], "zne.extrapolation.low");
  const high = finite(fit["high"], "zne.extrapolation.high");
  const zeroNoiseEstimate = finite(fit["zero_noise_estimate"], "zne.extrapolation.estimate");
  if (!(low <= zeroNoiseEstimate && zeroNoiseEstimate <= high))
    throw new Error("zne.extrapolation interval must contain its estimate");
  return {
    claimBoundary: text(row["claim_boundary"], "zne.claim_boundary"),
    observable: text(row["observable"], "zne.observable"),
    labels,
    measurementMap: map,
    calibration: calibration(row["calibration"], "zne.calibration", labels.length),
    points: list(row["points"], "zne.points").map((item, index) => {
      const field = `zne.points[${index}]`;
      const point = record(item, field);
      return {
        noiseScale: finite(point["noise_scale"], `${field}.noise_scale`),
        rawCounts: counts(point["raw_counts"], `${field}.raw_counts`, labels.length),
        shots: count(point["shots"], `${field}.shots`),
        rawExpectation: finite(point["raw_expectation"], `${field}.raw_expectation`),
        correctedExpectation: finite(point["corrected_expectation"], `${field}.corrected`),
        correctedStandardError: finite(point["corrected_standard_error"], `${field}.error`),
      };
    }),
    extrapolation: {
      zeroNoiseEstimate,
      standardError: finite(fit["standard_error"], "zne.extrapolation.standard_error"),
      low,
      high,
      coverage: finite(fit["coverage"], "zne.extrapolation.coverage"),
      order: count(fit["order"], "zne.extrapolation.order"),
      fitResidual: finite(fit["fit_residual"], "zne.extrapolation.fit_residual"),
      method: text(fit["method"], "zne.extrapolation.method"),
    },
  };
}

function hardware(raw: unknown): Hardware {
  const row = record(raw, "hardware");
  if (row["full_confusion_matrix_available"] !== false)
    throw new Error("hardware must state that no full confusion matrix is available");
  if (row["admitted"] !== true) throw new Error("hardware bundle must be admitted");
  return {
    source: text(row["source"], "hardware.source"),
    bundleSchema: text(row["bundle_schema"], "hardware.bundle_schema"),
    bundleDigest: text(row["bundle_digest"], "hardware.bundle_digest"),
    admitted: true,
    method: text(row["method"], "hardware.method"),
    fullConfusionMatrixAvailable: false,
    caveat: text(row["caveat"], "hardware.caveat"),
    pairs: list(row["pairs"], "hardware.pairs").map((item, index) => {
      const field = `hardware.pairs[${index}]`;
      const pair = record(item, field);
      return {
        dataset: text(pair["dataset"], `${field}.dataset`),
        comparison: text(pair["comparison"], `${field}.comparison`),
        depth: count(pair["depth"], `${field}.depth`),
        leftRaw: finite(pair["left_raw"], `${field}.left_raw`),
        rightRaw: finite(pair["right_raw"], `${field}.right_raw`),
        leftCorrected: finite(pair["left_corrected"], `${field}.left_corrected`),
        rightCorrected: finite(pair["right_corrected"], `${field}.right_corrected`),
        rawRelative: finite(pair["raw_relative"], `${field}.raw_relative`),
        correctedRelative: finite(pair["corrected_relative"], `${field}.corrected_relative`),
        nLeft: count(pair["n_left"], `${field}.n_left`),
        nRight: count(pair["n_right"], `${field}.n_right`),
      };
    }),
  };
}

/**
 * Parse the committed instrument data, refusing anything malformed.
 *
 * Counts that do not add up to their shots, a label list that misses or
 * repeats a label, negative values that are not all named, or a hardware
 * section that claims a full confusion matrix refuse the whole artefact.
 *
 * @param raw - Decoded JSON.
 * @returns The parsed instrument or the reason it was refused.
 */
export function parseMitigationInstrument(raw: unknown): Loaded<MitigationInstrument> {
  try {
    const payload = record(raw, "artefact");
    if (payload["schema"] !== MITIGATION_INSTRUMENT_SCHEMA)
      throw new Error(`schema must be ${MITIGATION_INSTRUMENT_SCHEMA}`);
    const readout = record(payload["readout"], "readout");
    return {
      ok: true,
      value: {
        artifactId: text(payload["artifact_id"], "artifact_id"),
        claimBoundary: text(payload["claim_boundary"], "claim_boundary"),
        labelOrder: text(payload["label_order"], "label_order"),
        readout: {
          claimBoundary: text(readout["claim_boundary"], "readout.claim_boundary"),
          method: text(readout["method"], "readout.method"),
          cases: list(readout["cases"], "readout.cases").map(readoutCase),
        },
        zne: zne(payload["zne"]),
        hardware: hardware(payload["hardware"]),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/**
 * Describe a measurement map as text, classical bit first.
 *
 * @param map - Physical qubit of each classical bit.
 * @returns For example ``c0←q2, c1←q0``.
 */
export function describeMap(map: readonly BitMapping[]): string {
  return map.map((item) => `c${item.classicalBit}←q${item.qubit}`).join(", ");
}

function csvField(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

/**
 * Export one readout case as CSV, raw counts beside any correction.
 *
 * The measurement map and label order lead the file, so the columns can be read
 * back to their qubits. A refused case keeps its raw columns and states the
 * source's reason; its corrected columns stay empty.
 *
 * @param item - Readout case.
 * @returns CSV text with a trailing newline.
 */
export function readoutCaseCsv(item: ReadoutCase): string {
  const lines = [
    `# case,${csvField(item.caseId)}`,
    `# measurement map,${csvField(describeMap(item.measurementMap))}`,
    `# status,${item.status === "corrected" ? "corrected" : csvField(`refused: ${item.reason}`)}`,
    "label,raw_count,raw_frequency,corrected,sampling_standard_error,calibration_standard_error,total_standard_error",
  ];
  item.labels.forEach((label, index) => {
    const raw = [label, String(item.rawCounts[index]), String(item.rawFrequencies[index])];
    const fixed =
      item.status === "corrected"
        ? [
            item.corrected.quasiprobabilities,
            item.corrected.samplingStandardError,
            item.corrected.calibrationStandardError,
            item.corrected.totalStandardError,
          ].map((column) => String(column[index]))
        : ["", "", "", ""];
    lines.push([...raw, ...fixed].join(","));
  });
  return `${lines.join("\n")}\n`;
}

/** The committed instrument data, parsed once. */
export const mitigationInstrument = parseMitigationInstrument(committedInstrumentJson);
