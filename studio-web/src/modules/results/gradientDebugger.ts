// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — gradient debugger model over source-owned plans and units

import committedUnitsJson from "../../../../data/studio/gradient_debugger_units_20261030.json";
import type { GradientPlanExplanationRowView, Loaded } from "../../shared/browser/data";

/** Schema of the committed gradient-debugger units. */
export const GRADIENT_UNITS_SCHEMA = "scpn_qc_studio_gradient_debugger_units_v1";

/** Frameworks the browser can run: only the shipped WASM replay of a committed unit. */
const BROWSER_FRAMEWORKS: ReadonlySet<string> = new Set(["native"]);

/** One named parameter of a committed unit. */
export interface UnitParameter {
  /** Parameter name in the program. */
  readonly name: string;
  /** Effect-IR value it binds. */
  readonly target: string;
  /** Committed input value. */
  readonly value: number;
}

/** One forward step of a unit's program, read from its effect-IR. */
export interface ProgramStep {
  /** Effect-IR value the step writes. */
  readonly target: string;
  /** `parameter` or the operation name, such as `cos`. */
  readonly operation: string;
  /** Values or literals the step reads. */
  readonly inputs: readonly string[];
}

/** Gradient the source states for one trainable mask. */
export interface MaskedGradient {
  /** Whether each parameter is trainable, in parameter order. */
  readonly trainable: readonly boolean[];
  /** Gradient with frozen parameters set to zero by the source rule. */
  readonly gradient: readonly number[];
}

/** One framework the owner reference was computed with. */
export interface ReferenceFramework {
  /** Backend name, such as `qiskit`. */
  readonly backend: string;
  /** Exact dependency versions of that run. */
  readonly versions: Readonly<Record<string, string>>;
  /** Fingerprint of the circuit the backend ran. */
  readonly circuitFingerprint: string;
  /** Backend evaluations one gradient took. */
  readonly evaluations: number;
}

/** The owner's exact-state reference for a unit. */
export interface UnitReference {
  /** Reference artefact identity. */
  readonly artifactId: string;
  /** Case within that artefact. */
  readonly caseId: string;
  /** Reference value. */
  readonly value: number;
  /** Reference gradient. */
  readonly gradient: readonly number[];
  /** Frameworks that reproduced it. */
  readonly frameworks: readonly ReferenceFramework[];
}

/** One committed program the debugger can replay. */
export interface GradientUnit {
  /** Stable unit identifier. */
  readonly unitId: string;
  /** Display title. */
  readonly title: string;
  /** Objective in words, such as `⟨Z⟩ = cos θ`. */
  readonly objective: string;
  /** Plan cell this unit realises, or null for a standalone program. */
  readonly cellId: string | null;
  /** Parameters in effect-IR order. */
  readonly parameters: readonly UnitParameter[];
  /** Forward steps read from the effect-IR. */
  readonly steps: readonly ProgramStep[];
  /** Exact WASM replay input, hex encoded. */
  readonly inputHex: string;
  /** SHA-256 the replay input must hash to before the kernel runs. */
  readonly inputSha256: string;
  /** Value and gradient from the differentiate action. */
  readonly expected: { readonly value: number; readonly gradient: readonly number[] };
  /** Central finite-difference cross-check from the same action. */
  readonly finiteDifference: {
    readonly gradient: readonly number[];
    readonly step: number;
    readonly maxAbsError: number;
    readonly tolerance: number;
  };
  /** Gradient under every trainable mask. */
  readonly masks: readonly MaskedGradient[];
  /** Owner reference, or null when none is linked. */
  readonly reference: UnitReference | null;
  /** Exact request a local runner accepts. */
  readonly localRunner: { readonly actionId: string; readonly params: unknown };
}

/** The committed units and the claim they carry. */
export interface GradientUnits {
  /** Artefact identity. */
  readonly artifactId: string;
  /** What the units may be cited for. */
  readonly claimBoundary: string;
  /** Units in artefact order. */
  readonly units: readonly GradientUnit[];
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${field} must be an object`);
  return value as Record<string, unknown>;
}

function text(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0)
    throw new Error(`${field} must be a non-empty string`);
  return value;
}

function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Error(`${field} must be a finite number`);
  return value;
}

function numbers(value: unknown, field: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${field} must be a list`);
  return value.map((item, index) => finite(item, `${field}[${index}]`));
}

function list(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${field} must be a list`);
  return value;
}

function steps(effectIr: unknown, field: string): ProgramStep[] {
  const ir = record(JSON.parse(text(effectIr, field)) as unknown, field);
  return list(ir["effects"], `${field}.effects`).map((raw, index) => {
    const effect = record(raw, `${field}.effects[${index}]`);
    return {
      target: text(effect["target"], `${field}.effects[${index}].target`),
      operation: text(effect["operation"], `${field}.effects[${index}].operation`),
      inputs: list(effect["inputs"], `${field}.effects[${index}].inputs`).map((input, at) =>
        text(input, `${field}.effects[${index}].inputs[${at}]`),
      ),
    };
  });
}

function reference(raw: unknown, field: string): UnitReference | null {
  if (raw === null) return null;
  const row = record(raw, field);
  return {
    artifactId: text(row["artifact_id"], `${field}.artifact_id`),
    caseId: text(row["case_id"], `${field}.case_id`),
    value: finite(row["value"], `${field}.value`),
    gradient: numbers(row["gradient"], `${field}.gradient`),
    frameworks: list(row["frameworks"], `${field}.frameworks`).map((item, index) => {
      const framework = record(item, `${field}.frameworks[${index}]`);
      const versions = record(framework["dependency_versions"], `${field}.versions`);
      return {
        backend: text(framework["backend"], `${field}.frameworks[${index}].backend`),
        versions: Object.fromEntries(
          Object.entries(versions).map(([name, version]) => [name, text(version, name)]),
        ),
        circuitFingerprint: text(framework["circuit_fingerprint"], `${field}.fingerprint`),
        evaluations: finite(framework["evaluations"], `${field}.evaluations`),
      };
    }),
  };
}

function unit(raw: unknown, index: number): GradientUnit {
  const field = `units[${index}]`;
  const row = record(raw, field);
  const parameters = list(row["parameters"], `${field}.parameters`).map((item, at) => {
    const parameter = record(item, `${field}.parameters[${at}]`);
    return {
      name: text(parameter["name"], `${field}.parameters[${at}].name`),
      target: text(parameter["target"], `${field}.parameters[${at}].target`),
      value: finite(parameter["value"], `${field}.parameters[${at}].value`),
    };
  });
  const expected = record(row["expected"], `${field}.expected`);
  const finiteDifference = record(row["finite_difference"], `${field}.finite_difference`);
  const runner = record(row["local_runner"], `${field}.local_runner`);
  const cellId = row["cell_id"];
  return {
    unitId: text(row["unit_id"], `${field}.unit_id`),
    title: text(row["title"], `${field}.title`),
    objective: text(row["objective"], `${field}.objective`),
    cellId: cellId === null ? null : text(cellId, `${field}.cell_id`),
    parameters,
    steps: steps(row["effect_ir"], `${field}.effect_ir`),
    inputHex: text(row["input_hex"], `${field}.input_hex`),
    inputSha256: text(row["input_sha256"], `${field}.input_sha256`),
    expected: {
      value: finite(expected["value"], `${field}.expected.value`),
      gradient: numbers(expected["gradient"], `${field}.expected.gradient`),
    },
    finiteDifference: {
      gradient: numbers(finiteDifference["gradient"], `${field}.finite_difference.gradient`),
      step: finite(finiteDifference["step"], `${field}.finite_difference.step`),
      maxAbsError: finite(finiteDifference["max_abs_error"], `${field}.max_abs_error`),
      tolerance: finite(finiteDifference["tolerance"], `${field}.tolerance`),
    },
    masks: list(row["masks"], `${field}.masks`).map((item, at) => {
      const mask = record(item, `${field}.masks[${at}]`);
      const trainable = list(mask["trainable"], `${field}.masks[${at}].trainable`).map((flag) => {
        if (typeof flag !== "boolean")
          throw new Error(`${field}.masks[${at}] flags must be true or false`);
        return flag;
      });
      return { trainable, gradient: numbers(mask["gradient"], `${field}.masks[${at}].gradient`) };
    }),
    reference: reference(row["reference"], `${field}.reference`),
    localRunner: {
      actionId: text(runner["action_id"], `${field}.local_runner.action_id`),
      params: runner["params"],
    },
  };
}

/**
 * Admit the committed gradient-debugger units, or say why they are refused.
 *
 * @param raw - Decoded artefact.
 * @returns The units, or the first reason they cannot be used.
 */
export function parseGradientUnits(raw: unknown): Loaded<GradientUnits> {
  try {
    const payload = record(raw, "artefact");
    if (payload["schema"] !== GRADIENT_UNITS_SCHEMA)
      throw new Error(`schema must be ${GRADIENT_UNITS_SCHEMA}`);
    return {
      ok: true,
      value: {
        artifactId: text(payload["artifact_id"], "artifact_id"),
        claimBoundary: text(payload["claim_boundary"], "claim_boundary"),
        units: list(payload["units"], "units").map(unit),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** What selecting a choice leads to; every route says why. */
export type DebuggerRoute =
  | {
      /** The shipped WASM kernel recomputes this committed unit. */
      readonly kind: "browser";
      /** The unit it replays. */
      readonly unit: GradientUnit;
    }
  | {
      /** The plan refuses this combination before anything runs. */
      readonly kind: "refused";
      /** Source-stated boundaries the plan fails closed at. */
      readonly reasons: readonly string[];
    }
  | {
      /** No runtime here can execute it; a local runner request is prepared instead. */
      readonly kind: "unavailable";
      /** Why the browser cannot run it. */
      readonly reason: string;
      /** The unit whose request is prepared, when one is committed. */
      readonly unit: GradientUnit | null;
    };

/** One selectable objective/parameter/transform/backend combination. */
export interface DebuggerChoice {
  /** Stable identifier: a plan cell or a standalone unit. */
  readonly id: string;
  /** Label for the selector. */
  readonly label: string;
  /** The plan row, when the choice is a declared plan cell. */
  readonly cell: GradientPlanExplanationRowView | null;
  /** Where selecting it leads. */
  readonly route: DebuggerRoute;
}

/**
 * Join declared plan cells with committed units into the debugger's choices.
 *
 * Only declared cells and committed units are selectable. A blocked cell is
 * refused with its source boundaries; a framework without a browser runtime,
 * or a supported cell with no committed unit, is unavailable here and gets a
 * local-runner request instead of an emulated result.
 *
 * @param cells - Declared plan cells.
 * @param units - Committed units.
 * @returns Choices in plan order, then standalone units.
 */
export function debuggerChoices(
  cells: readonly GradientPlanExplanationRowView[],
  units: readonly GradientUnit[],
): DebuggerChoice[] {
  // Gate, observable and state: cells sharing them differentiate the same objective.
  const objective = (cellId: string) => cellId.split("::").slice(0, 3).join("::");
  const planned = cells.map((cell): DebuggerChoice => {
    const unitForCell = units.find((candidate) => candidate.cellId === cell.cellId) ?? null;
    const sameObjective =
      unitForCell ??
      units.find(
        (candidate) =>
          candidate.cellId !== null && objective(candidate.cellId) === objective(cell.cellId),
      ) ??
      null;
    const label = `${cell.operation} · ${cell.transform} · ${cell.backend} · ${cell.framework}`;
    if (!cell.supported)
      return {
        id: cell.cellId,
        label,
        cell,
        route: { kind: "refused", reasons: cell.failClosedBoundaries },
      };
    if (!BROWSER_FRAMEWORKS.has(cell.framework))
      return {
        id: cell.cellId,
        label,
        cell,
        route: {
          kind: "unavailable",
          reason: `No ${cell.framework} runtime runs in this browser; it is not emulated.`,
          unit: sameObjective,
        },
      };
    if (unitForCell === null)
      return {
        id: cell.cellId,
        label,
        cell,
        route: {
          kind: "unavailable",
          reason: "No committed unit replays this cell in the browser.",
          unit: null,
        },
      };
    return { id: cell.cellId, label, cell, route: { kind: "browser", unit: unitForCell } };
  });
  const standalone = units
    .filter((candidate) => candidate.cellId === null)
    .map(
      (candidate): DebuggerChoice => ({
        id: candidate.unitId,
        label: `${candidate.title} · program unit`,
        cell: null,
        route: { kind: "browser", unit: candidate },
      }),
    );
  return [...planned, ...standalone];
}

/**
 * The gradient the source states for a trainable mask.
 *
 * @param unit - Committed unit.
 * @param trainable - Whether each parameter is trainable.
 * @returns The masked gradient, or null when the source states none for it.
 */
export function maskedGradient(
  unit: GradientUnit,
  trainable: readonly boolean[],
): readonly number[] | null {
  const match = unit.masks.find(
    (mask) =>
      mask.trainable.length === trainable.length &&
      mask.trainable.every((flag, index) => flag === trainable[index]),
  );
  return match?.gradient ?? null;
}

/**
 * Whether a browser recomputation agrees with the source within the frozen
 * float64 budget `|actual − reference| ≤ 1e-10 + 1e-8·|reference|`.
 *
 * `sin` and `cos` come from each platform's math library, so agreement is
 * judged within that budget rather than bit for bit.
 *
 * @param actual - Recomputed values.
 * @param reference - Source values.
 * @returns True when every value agrees.
 */
export function withinBudget(actual: readonly number[], reference: readonly number[]): boolean {
  return (
    actual.length === reference.length &&
    actual.every(
      (value, index) =>
        Math.abs(value - (reference[index] as number)) <=
        1e-10 + 1e-8 * Math.abs(reference[index] as number),
    )
  );
}

/**
 * The exact command and parameter file a local runner accepts for a unit.
 *
 * @param unit - Committed unit, or null for a cell without one.
 * @returns The shell command and the JSON parameters to save beside it.
 */
export function localRunnerRequest(unit: GradientUnit | null): {
  readonly command: string;
  readonly paramsJson: string | null;
} {
  if (unit === null)
    return {
      command:
        "scpn-studio-run differentiate --action-id <id> --params-file <program.json> --preview",
      paramsJson: null,
    };
  const file = `${unit.localRunner.actionId}.json`;
  return {
    command: `scpn-studio-run differentiate --action-id ${unit.localRunner.actionId} --params-file ${file}`,
    paramsJson: `${JSON.stringify(unit.localRunner.params, null, 2)}\n`,
  };
}

/** The committed gradient-debugger units, admitted or refused at load. */
export const gradientUnits: Loaded<GradientUnits> = parseGradientUnits(committedUnitsJson);
