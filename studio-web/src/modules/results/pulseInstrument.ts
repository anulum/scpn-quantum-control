// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — pulse and analog feasibility instrument model over source-owned data

import committedInstrumentJson from "../../../../data/studio/pulse_instrument_20261031.json";
import type { Loaded } from "../../shared/browser/data";
import {
  count,
  finite,
  flag,
  list,
  numbers,
  record,
  text,
  within,
} from "../../shared/browser/jsonFields";

/** Schema of the committed instrument data. */
export const PULSE_INSTRUMENT_SCHEMA = "scpn_qc_studio_pulse_instrument_v1";

/** One design pulse in seconds and rad/s. */
export interface DesignPulse {
  /** Position in the design, as violations name it. */
  readonly index: number;
  /** Logical qubits: one for a drive, two for a coupler. */
  readonly qubits: readonly number[];
  /** Start in seconds. */
  readonly startSeconds: number;
  /** Duration in seconds. */
  readonly durationSeconds: number;
  /** Piecewise-constant rate per equal segment, in rad/s. */
  readonly ratesRadPerSecond: readonly number[];
}

/** One violation as the source wrote it, with the span it names. */
export interface Violation {
  /** Stable violation code. */
  readonly code: string;
  /** Design pulse the violation names, if any. */
  readonly pulse: number | null;
  /** Native channel the violation names, if any. */
  readonly channel: string | null;
  /** The source's full text. */
  readonly text: string;
}

/** One native pulse on a physical channel. */
export interface NativePulse {
  /** ``d<q>`` drive or ``u<a>_<b>`` coupler channel. */
  readonly channel: string;
  /** Physical qubits after placement. */
  readonly physicalQubits: readonly number[];
  /** Start sample. */
  readonly t0: number;
  /** Native amplitudes, one per sample. */
  readonly samples: readonly number[];
}

/** Limits a design was checked against, in their own units. */
export interface PulseLimits {
  /** ``design`` or ``provider_metadata``. */
  readonly source: "design" | "provider_metadata";
  /** Native sample period in seconds. */
  readonly dtSeconds: number;
  /** Start and duration grid in samples. */
  readonly granularitySamples: number;
  /** Rabi rate in rad/s of native amplitude 1 on a drive. */
  readonly rabiRatePerAmplitude: number;
  /** Coupling rate in rad/s of native amplitude 1 on a coupler, if declared. */
  readonly couplerRatePerAmplitude: number | null;
  /** Largest native amplitude. */
  readonly maxAmplitude: number;
  /** Largest amplitude step between samples, if checked. */
  readonly maxSlew: number | null;
  /** Physical qubits. */
  readonly nQubits: number;
  /** Physical pairs a coupler may drive. */
  readonly couplingMap: readonly (readonly number[])[];
}

/** A dated capability snapshot the limits were copied from. */
export interface CapabilitySnapshot {
  /** Provider name. */
  readonly provider: string;
  /** Backend name. */
  readonly backendName: string;
  /** When the capabilities were observed. */
  readonly observedAt: string;
  /** When they stop being valid. */
  readonly expiresAt: string;
  /** Whether they were valid at the assessment time. */
  readonly validAtAssessment: boolean;
}

/** One logical-to-physical placement. */
export interface Placement {
  /** Logical qubit of the design. */
  readonly logical: number;
  /** Physical qubit it is placed on. */
  readonly physical: number;
}

/** One assessed design. */
export interface PulseCase {
  /** Stable identifier. */
  readonly caseId: string;
  /** What the case demonstrates. */
  readonly title: string;
  /** Logical-to-physical placement. */
  readonly placement: readonly Placement[];
  /** Limits checked against. */
  readonly limits: PulseLimits;
  /** Capability snapshot, when the limits came from one. */
  readonly snapshot: CapabilitySnapshot | null;
  /** Design pulses. */
  readonly design: readonly DesignPulse[];
  /** Whether no violation was found. */
  readonly feasible: boolean;
  /** Every violation. */
  readonly violations: readonly Violation[];
  /** Native pulses, only when feasible. */
  readonly native: readonly NativePulse[];
  /** The no-submit dossier, or ``null`` when refused. */
  readonly dossier: Readonly<Record<string, unknown>> | null;
  /** Why no dossier exists, or ``null``. */
  readonly dossierRefusal: string | null;
}

/** The whole committed instrument data. */
export interface PulseInstrument {
  /** Artefact identity. */
  readonly artifactId: string;
  /** What the data may be cited for. */
  readonly claimBoundary: string;
  /** The design boundary every assessment carries. */
  readonly designClaimBoundary: string;
  /** Fixed assessment time. */
  readonly assessedAt: string;
  /** The cases. */
  readonly cases: readonly PulseCase[];
}

function nullable<T>(value: unknown, read: (item: unknown) => T): T | null {
  return value === null ? null : read(value);
}

const UTC_SECOND = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;

/** Read an exact ``YYYY-MM-DDTHH:MM:SSZ`` instant as milliseconds, refusing anything else. */
function utcSecond(value: unknown, field: string): number {
  const instant = text(value, field);
  const milliseconds = Date.parse(instant);
  if (!UTC_SECOND.test(instant) || Number.isNaN(milliseconds))
    throw new Error(`${field} must be an exact UTC second, YYYY-MM-DDTHH:MM:SSZ`);
  return milliseconds;
}

function positive(value: unknown, field: string): number {
  return within(value, field, Number.MIN_VALUE, Number.MAX_VALUE);
}

/** Largest native amplitude the owner admits: the limit plus its rounding tolerance. */
const AMPLITUDE_TOLERANCE = 1e-12;

/** JSON text with every object's keys sorted, so equal values compare equal. */
function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "object" && item !== null && !Array.isArray(item)
      ? Object.fromEntries(
          Object.entries(item as Record<string, unknown>).sort(([left], [right]) =>
            left < right ? -1 : 1,
          ),
        )
      : item,
  );
}

function sameJson(left: unknown, right: unknown): boolean {
  return canonical(left) === canonical(right);
}

function limits(raw: unknown, field: string): PulseLimits {
  const row = record(raw, field);
  const source = row["source"];
  if (source !== "design" && source !== "provider_metadata")
    throw new Error(`${field}.source must be design or provider_metadata`);
  const nQubits = count(row["n_qubits"], `${field}.n_qubits`);
  if (nQubits < 1) throw new Error(`${field}.n_qubits must be at least 1`);
  const granularitySamples = count(row["granularity_samples"], `${field}.granularity`);
  if (granularitySamples < 1) throw new Error(`${field}.granularity must be at least 1`);
  return {
    source,
    dtSeconds: within(row["dt_seconds"], `${field}.dt_seconds`, Number.MIN_VALUE, 1),
    granularitySamples,
    rabiRatePerAmplitude: positive(row["rabi_rate_per_amplitude"], `${field}.rabi`),
    couplerRatePerAmplitude: nullable(row["coupler_rate_per_amplitude"], (item) =>
      positive(item, `${field}.coupler`),
    ),
    maxAmplitude: positive(row["max_amplitude"], `${field}.max_amplitude`),
    maxSlew: nullable(row["max_slew"], (item) => positive(item, `${field}.max_slew`)),
    nQubits,
    couplingMap: list(row["coupling_map"], `${field}.coupling_map`).map((raw, at) => {
      const pair = numbers(raw, `${field}.coupling_map[${at}]`);
      if (
        pair.length !== 2 ||
        pair[0] === pair[1] ||
        pair.some((qubit) => !Number.isInteger(qubit) || qubit < 0 || qubit >= nQubits)
      )
        throw new Error(`${field}.coupling_map[${at}] must be two distinct physical qubits`);
      return pair;
    }),
  };
}

/** A snapshot whose validity flag is the one its own dates give at the assessment time. */
function snapshot(raw: unknown, field: string, assessed: number): CapabilitySnapshot {
  const row = record(raw, field);
  const observed = utcSecond(row["observed_at"], `${field}.observed_at`);
  const expires = utcSecond(row["expires_at"], `${field}.expires_at`);
  if (observed >= expires) throw new Error(`${field} must be observed before it expires`);
  const validAtAssessment = flag(row["valid_at_assessment"], `${field}.valid_at_assessment`);
  if (validAtAssessment !== (observed <= assessed && assessed < expires))
    throw new Error(`${field}.valid_at_assessment disagrees with its dates`);
  return {
    provider: text(row["provider"], `${field}.provider`),
    backendName: text(row["backend_name"], `${field}.backend_name`),
    observedAt: text(row["observed_at"], `${field}.observed_at`),
    expiresAt: text(row["expires_at"], `${field}.expires_at`),
    validAtAssessment,
  };
}

/**
 * A dossier must be the owner's export of this case: its schema, its limits and
 * native pulses, and in its own fields no submission and no device authority.
 */
function dossier(
  raw: unknown,
  field: string,
  limits: PulseLimits,
  native: readonly NativePulse[],
): Readonly<Record<string, unknown>> {
  const row = record(raw, field);
  if (row["hardware_submission"] !== false || row["device_authority"] !== false)
    throw new Error(`${field} must not submit or claim device authority`);
  if (row["submissions"] !== 0) throw new Error(`${field} must record zero submissions`);
  if (row["schema"] !== "native_pulse_design_v1")
    throw new Error(`${field}.schema must be native_pulse_design_v1`);
  const expectedLimits = {
    coupler_rate_per_amplitude: limits.couplerRatePerAmplitude,
    coupling_map: limits.couplingMap,
    granularity: limits.granularitySamples,
    max_amplitude: limits.maxAmplitude,
    max_slew: limits.maxSlew,
    n_qubits: limits.nQubits,
    rabi_rate_per_amplitude: limits.rabiRatePerAmplitude,
  };
  if (
    row["dt"] !== limits.dtSeconds ||
    row["limits_source"] !== limits.source ||
    !sameJson(record(row["limits"], `${field}.limits`), expectedLimits)
  )
    throw new Error(`${field} must carry the case's own limits`);
  const pulses = native.map((pulse) => ({
    channel: pulse.channel,
    physical_qubits: pulse.physicalQubits,
    t0: pulse.t0,
    samples: pulse.samples,
  }));
  if (!sameJson(row["pulses"], pulses))
    throw new Error(`${field}.pulses must be the case's native pulses`);
  return row;
}

function pulseCase(raw: unknown, index: number, assessed: number): PulseCase {
  const field = `cases[${index}]`;
  const row = record(raw, field);
  if (row["device_authority"] !== false || row["hardware_submission"] !== false)
    throw new Error(`${field} must not submit or claim device authority`);
  const design = list(row["design"], `${field}.design`).map((item, at) => {
    const pulse = record(item, `${field}.design[${at}]`);
    if (pulse["index"] !== at) throw new Error(`${field}.design must be in index order`);
    return {
      index: at,
      qubits: numbers(pulse["qubits"], `${field}.design[${at}].qubits`),
      startSeconds: finite(pulse["start_seconds"], `${field}.design[${at}].start`),
      durationSeconds: finite(pulse["duration_seconds"], `${field}.design[${at}].duration`),
      ratesRadPerSecond: numbers(pulse["rates_rad_per_second"], `${field}.design[${at}].rates`),
    };
  });
  const violations = list(row["violations"], `${field}.violations`).map((item, at) => {
    const violation = record(item, `${field}.violations[${at}]`);
    const pulse = nullable(violation["pulse"], (value) =>
      count(value, `${field}.violations[${at}].pulse`),
    );
    if (pulse !== null && pulse >= design.length)
      throw new Error(`${field}.violations[${at}] names a pulse the design does not have`);
    return {
      code: text(violation["code"], `${field}.violations[${at}].code`),
      pulse,
      channel: nullable(violation["channel"], (value) =>
        text(value, `${field}.violations[${at}].channel`),
      ),
      text: text(violation["text"], `${field}.violations[${at}].text`),
    };
  });
  const feasible = flag(row["feasible"], `${field}.feasible`);
  if (feasible !== (violations.length === 0))
    throw new Error(`${field}.feasible must hold exactly when there is no violation`);
  const caseLimits = limits(row["limits"], `${field}.limits`);
  const native = list(row["native"], `${field}.native`).map((item, at) => {
    const pulse = record(item, `${field}.native[${at}]`);
    const samples = numbers(pulse["samples"], `${field}.native[${at}].samples`);
    if (samples.some((sample) => Math.abs(sample) > caseLimits.maxAmplitude + AMPLITUDE_TOLERANCE))
      throw new Error(`${field}.native[${at}] has a sample above the amplitude limit`);
    return {
      channel: text(pulse["channel"], `${field}.native[${at}].channel`),
      physicalQubits: numbers(pulse["physical_qubits"], `${field}.native[${at}].qubits`),
      t0: count(pulse["t0"], `${field}.native[${at}].t0`),
      samples,
    };
  });
  if (!feasible && native.length > 0)
    throw new Error(`${field} has native pulses although it is not feasible`);
  const snapshotView = nullable(row["snapshot"], (item) =>
    snapshot(item, `${field}.snapshot`, assessed),
  );
  const dossierRefusal = nullable(row["dossier_refusal"], (item) =>
    text(item, `${field}.dossier_refusal`),
  );
  const exported = row["dossier"] !== null;
  if (exported === (dossierRefusal !== null))
    throw new Error(`${field} must have either a dossier or the reason it has none`);
  if (exported && (!feasible || snapshotView?.validAtAssessment === false))
    throw new Error(`${field} exports a dossier for an infeasible design or an expired snapshot`);
  const dossierView = exported
    ? dossier(row["dossier"], `${field}.dossier`, caseLimits, native)
    : null;
  return {
    caseId: text(row["case_id"], `${field}.case_id`),
    title: text(row["title"], `${field}.title`),
    placement: list(row["placement"], `${field}.placement`).map((item, at) => {
      const pair = record(item, `${field}.placement[${at}]`);
      return {
        logical: count(pair["logical"], `${field}.placement[${at}].logical`),
        physical: count(pair["physical"], `${field}.placement[${at}].physical`),
      };
    }),
    limits: caseLimits,
    snapshot: snapshotView,
    design,
    feasible,
    violations,
    native,
    dossier: dossierView,
    dossierRefusal,
  };
}

/**
 * Parse the committed instrument data, refusing anything malformed.
 *
 * A case or dossier that submits or claims device authority, a feasibility
 * flag that disagrees with the violations, native pulses for an infeasible
 * design, or a dossier for an infeasible design or an expired snapshot
 * refuses the whole artefact.
 *
 * @param raw - Decoded JSON.
 * @returns The parsed instrument or the reason it was refused.
 */
export function parsePulseInstrument(raw: unknown): Loaded<PulseInstrument> {
  try {
    const payload = record(raw, "artefact");
    if (payload["schema"] !== PULSE_INSTRUMENT_SCHEMA)
      throw new Error(`schema must be ${PULSE_INSTRUMENT_SCHEMA}`);
    return {
      ok: true,
      value: {
        artifactId: text(payload["artifact_id"], "artifact_id"),
        claimBoundary: text(payload["claim_boundary"], "claim_boundary"),
        designClaimBoundary: text(payload["design_claim_boundary"], "design_claim_boundary"),
        assessedAt: text(payload["assessed_at"], "assessed_at"),
        cases: list(payload["cases"], "cases").map((item, index) =>
          pulseCase(item, index, utcSecond(payload["assessed_at"], "assessed_at")),
        ),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** The committed instrument data, parsed once. */
export const pulseInstrument = parsePulseInstrument(committedInstrumentJson);
