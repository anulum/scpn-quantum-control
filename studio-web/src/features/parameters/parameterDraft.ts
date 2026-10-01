// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable linked parameter draft reducer

import { canonicalBytes, canonicalDigest, parseExperimentRevision, parseParameterSpec, validateParameterBinding } from "../../shared/contracts";
import type { ExperimentRevision, ParameterSpec, ParseResult } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";

/** Maximum total editable elements, checked before view/history allocation. */
export const parameterEditorElementLimit = 4096;
/** Maximum retained semantic changes; redo shares this bounded history. */
export const parameterEditorHistoryLimit = 64;

/** Original immutable revision and the source-owned specifications/units for its parameters. */
export interface ParameterDraftSource {
  /** Structural v1 input; workspace admission is required separately before persistence. */ readonly revision: ExperimentRevision;
  /** Original source parameter declarations; no view invents a domain. */ readonly specs: readonly ParameterSpec[];
  /** Explicit original units; mismatch refuses before editing. */ readonly units: Readonly<Record<string, string>>;
}

/** Original row-major typed wire value, with no scientific or unit transformation. */
export interface ParameterValue {
  /** Original supported workspace dtype. */ readonly dtype: string;
  /** Original dimensions, retained as exact integers. */ readonly shape: readonly bigint[];
  /** Binary64 hex or canonical signed/unsigned decimal elements. */ readonly values: readonly string[];
}

/** Semantic values and trainable subset; selection/edit policy are separate display state. */
export interface ParameterSnapshot {
  /** Source-bound immutable parameter values. */ readonly parameters: Readonly<Record<string, ParameterValue>>;
  /** Exact source units; user conversions must target these explicitly. */ readonly units: Readonly<Record<string, string>>;
  /** Per-element trainable subset of source-declared trainable parameters. */ readonly trainableMasks: Readonly<Record<string, readonly boolean[]>>;
}

/** One reducer owns shared selection, semantic values and bounded undo/redo. */
export interface ParameterDraft {
  /** Original immutable input, captured before the caller can mutate it. */ readonly source: ParameterDraftSource;
  /** The only current semantic snapshot used by every view and save. */ readonly snapshot: ParameterSnapshot;
  /** Shared selection of one original row-major element, or null for empty inputs. */ readonly selection: {
    /** Original parameter key. */ readonly key: string;
    /** Zero-based row-major element index. */ readonly index: number;
  } | null;
  /** Explicit edit-time policy; switching it never repairs existing values. */ readonly policy: "directed" | "symmetric";
  /** Earlier snapshots, in chronological order. */ readonly past: readonly ParameterSnapshot[];
  /** Undone snapshots, in restore order. */ readonly future: readonly ParameterSnapshot[];
  /** Authored refusal text; rejected edits retain semantic values and history. */ readonly refusal: string | null;
}

/** Public edit commands used by forms, matrix cells and sparse graph selections. */
export type ParameterAction =
  | {
    /** Select an element without changing its semantic value. */ readonly type: "select";
    /** Original parameter key. */ readonly key: string;
    /** Zero-based row-major element index. */ readonly index: number;
  }
  | {
    /** Choose an explicit edit policy without repairing prior coefficients. */ readonly type: "policy";
    /** Single directed coefficient or an explicit mirrored pair. */ readonly policy: "directed" | "symmetric";
  }
  | {
    /** Apply a validated form value. */ readonly type: "value";
    /** Original parameter key. */ readonly key: string;
    /** Zero-based row-major element index. */ readonly index: number;
    /** Decimal form text; integers remain canonical decimal strings. */ readonly text: string;
    /** Declared input unit; mismatches require an explicit supported conversion. */ readonly unit: string;
    /** Explicit user request for binary64 SI-prefix conversion before domain validation. */ readonly convert?: boolean;
  }
  | {
    /** Edit the per-element source-eligible trainable subset. */ readonly type: "mask";
    /** Original parameter key. */ readonly key: string;
    /** Zero-based row-major element index. */ readonly index: number;
    /** Requested eligibility within the source-declared trainable parameter. */ readonly enabled: boolean;
  }
  | {
    /** Replace values atomically after validating every original specification. */ readonly type: "replace";
    /** Complete source-bound typed payload index. */ readonly parameters: Readonly<Record<string, unknown>>;
    /** Complete exact original unit index. */ readonly units: Readonly<Record<string, string>>;
  }
  | {
    /** Restore the immediately prior semantic snapshot. */
    readonly type: "undo";
  }
  | {
    /** Reapply the immediately undone semantic snapshot. */
    readonly type: "redo";
  };

function take<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
  return result.value;
}

function freezeSnapshot(source: ParameterDraftSource, parameters: Readonly<Record<string, unknown>>, units: Readonly<Record<string, string>>, masks?: Readonly<Record<string, readonly boolean[]>>): ParameterSnapshot {
  canonicalBytes("studio_parameter_input.v1", { parameters, units, masks: masks ?? null });
  const entries = dataEntries(parameters);
  const keys = entries.map(([key]) => key);
  if (Object.keys(units).length !== keys.length || keys.some(key => !Object.hasOwn(units, key))) throw new Error("Parameter units must cover the values exactly");
  if (masks && (Object.keys(masks).length !== keys.length || keys.some(key => !Object.hasOwn(masks, key)))) throw new Error("Trainable masks must cover the values exactly");
  const specs = new Map(source.specs.map(spec => [spec.body["key"] as string, spec]));
  let total = 0;
  const values: [string, ParameterValue][] = [];
  const selectedMasks: [string, readonly boolean[]][] = [];
  for (const [key, payload] of entries) {
    const spec = specs.get(key);
    if (!spec) throw new Error(`Missing ParameterSpec for ${key}`);
    take(validateParameterBinding(spec, payload, units[key]!));
    const typed = payload as ParameterValue;
    const shape = spec.body["shape"] as readonly bigint[];
    if (shape.length > 2) throw new Error("Parameter editor supports scalar, vector and matrix shapes only");
    total += typed.values.length;
    if (total > parameterEditorElementLimit) throw new Error("Parameter editor exceeds the 4096-element bound");
    values.push([key, Object.freeze({ dtype: spec.body["dtype"] as string, shape, values: Object.freeze([...typed.values]) })]);
    const mask = masks?.[key] ?? typed.values.map(() => spec.body["trainable"] === true);
    if (mask.length !== typed.values.length || mask.some(value => typeof value !== "boolean" || (value && spec.body["trainable"] !== true))) throw new Error("Trainable mask must match shape and source eligibility");
    selectedMasks.push([key, Object.freeze([...mask])]);
  }
  return Object.freeze({ parameters: Object.freeze(Object.fromEntries(values)), units: Object.freeze({ ...units }), trainableMasks: Object.freeze(Object.fromEntries(selectedMasks)) });
}

/** Validate original parsers/bindings and capture an independent immutable draft. */
export function createParameterDraft(input: ParameterDraftSource): ParameterDraft {
  canonicalBytes("studio_parameter_source.v1", input);
  const revision = take(parseExperimentRevision(input.revision));
  const specs = Object.freeze(input.specs.map(spec => take(parseParameterSpec(spec))));
  if (new Set(specs.map(spec => spec.body["key"])).size !== specs.length) throw new Error("Duplicate ParameterSpec key");
  const source = Object.freeze({ revision, specs, units: Object.freeze({ ...input.units }) });
  const editor = revision.extensions["parameter_editor"];
  let masks: Readonly<Record<string, readonly boolean[]>> | undefined;
  if (editor !== undefined) {
    if (typeof editor !== "object" || editor === null || Array.isArray(editor)) throw new Error("Parameter editor metadata must be an object");
    const fields = Object.fromEntries(dataEntries(editor));
    if (Object.keys(fields).length !== 2 || fields["version"] !== 1n || typeof fields["trainable_masks"] !== "object" || fields["trainable_masks"] === null || Array.isArray(fields["trainable_masks"])) throw new Error("Unsupported parameter editor version or trainable masks");
    const maskFields = dataEntries(fields["trainable_masks"]);
    if (maskFields.some(([, mask]) => !Array.isArray(mask))) throw new Error("Trainable masks must be arrays");
    masks = Object.fromEntries(maskFields) as Readonly<Record<string, readonly boolean[]>>;
  }
  const snapshot = freezeSnapshot(source, revision.body["parameters"] as Readonly<Record<string, unknown>>, source.units, masks);
  const first = Object.entries(snapshot.parameters).find(([, value]) => value.values.length > 0);
  return Object.freeze({ source, snapshot, selection: first ? Object.freeze({ key: first[0], index: 0 }) : null, policy: "directed", past: Object.freeze([]), future: Object.freeze([]), refusal: null });
}

/** Display a validated typed element without losing negative zero or large integers. */
export function parameterElementText(value: ParameterValue, index: number): string {
  const element = value.values[index];
  if (element === undefined) throw new Error("Parameter element index out of range");
  if (value.dtype !== "float64") return element;
  const bytes = Uint8Array.from(element.match(/../g)!, pair => parseInt(pair, 16));
  const scalar = new DataView(bytes.buffer).getFloat64(0, false);
  return Object.is(scalar, -0) ? "-0" : String(scalar);
}

function encode(text: string, dtype: string): string {
  if (dtype !== "float64") {
    if (!/^(?:0|-?[1-9][0-9]{0,19})$/.test(text) || /[\r\n]$/.test(text)) throw new Error("Canonical int64/uint64 decimal required");
    return text;
  }
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text) || /[\r\n]$/.test(text) || !Number.isFinite(Number(text))) throw new Error("A finite float64 decimal value is required");
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, Number(text), false);
  return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

const inputUnits: Readonly<Record<string, Readonly<Record<string, number>>>> = Object.freeze({
  rad: Object.freeze({ rad: 1, mrad: 0.001, urad: 0.000001 }),
  s: Object.freeze({ s: 1, ms: 0.001, us: 0.000001 }),
  "rad/s": Object.freeze({ "rad/s": 1, "mrad/s": 0.001, "rad/ms": 1000 }),
});

/** List explicit SI-prefix input units supported for one original float64 unit. */
export function parameterInputUnits(unit: string): readonly string[] {
  return Object.freeze(Object.hasOwn(inputUnits, unit) ? Object.keys(inputUnits[unit]!) : [unit]);
}

function convertedText(text: string, from: string, to: string, dtype: string): string {
  encode(text, dtype);
  if (dtype !== "float64") throw new Error("Unit conversion requires float64; integer input retains its exact source unit");
  const units = Object.hasOwn(inputUnits, to) ? inputUnits[to]! : null;
  if (!units || !Object.hasOwn(units, from)) throw new Error("Unsupported unit conversion; source unit and values retained");
  const value = Number(text) * units[from]!;
  return Object.is(value, -0) ? "-0" : String(value);
}

function element(state: ParameterDraft, key: string, index: number): ParameterValue {
  const value = Object.hasOwn(state.snapshot.parameters, key) ? state.snapshot.parameters[key] : undefined;
  if (!value || !Number.isSafeInteger(index) || index < 0 || index >= value.values.length) throw new Error("Parameter selection out of range");
  return value;
}

/** Apply one validated edit atomically; invalid values never modify history or saved evidence. */
export function parameterDraftReducer(state: ParameterDraft, action: ParameterAction): ParameterDraft {
  try {
    if (action.type === "undo" || action.type === "redo") {
      const undo = action.type === "undo";
      const from = undo ? state.past : state.future;
      const target = from.at(-1);
      if (!target) return state;
      return Object.freeze({ ...state, snapshot: target, past: undo ? Object.freeze(state.past.slice(0, -1)) : Object.freeze([...state.past, state.snapshot]), future: undo ? Object.freeze([...state.future, state.snapshot]) : Object.freeze(state.future.slice(0, -1)), refusal: null });
    }
    if (action.type === "policy") return Object.freeze({ ...state, policy: action.policy, refusal: null });
    if (action.type === "select") {
      element(state, action.key, action.index);
      return Object.freeze({ ...state, selection: Object.freeze({ key: action.key, index: action.index }), refusal: null });
    }
    let snapshot: ParameterSnapshot;
    if (action.type === "replace") snapshot = freezeSnapshot(state.source, action.parameters, action.units, state.snapshot.trainableMasks);
    else {
      const value = element(state, action.key, action.index);
      const indices = [action.index];
      if (state.policy === "symmetric" && value.shape.length === 2) {
        if (value.shape[0] !== value.shape[1]) throw new Error("Symmetric edits require a square matrix");
        const n = Number(value.shape[0]);
        const reverse = (action.index % n) * n + Math.floor(action.index / n);
        if (reverse !== action.index) indices.push(reverse);
      }
      if (action.type === "value") {
        const targetUnit = state.snapshot.units[action.key]!;
        if (action.unit !== targetUnit && !action.convert) throw new Error("Input unit mismatch; choose an explicit supported conversion");
        const input = action.convert ? convertedText(action.text, action.unit, targetUnit, value.dtype) : action.text;
        const encoded = encode(input, value.dtype);
        const values = [...value.values];
        for (const index of indices) values[index] = encoded;
        snapshot = freezeSnapshot(state.source, { ...state.snapshot.parameters, [action.key]: { ...value, values } }, state.snapshot.units, state.snapshot.trainableMasks);
      } else {
        const mask = [...state.snapshot.trainableMasks[action.key]!];
        for (const index of indices) mask[index] = action.enabled;
        snapshot = freezeSnapshot(state.source, state.snapshot.parameters, state.snapshot.units, { ...state.snapshot.trainableMasks, [action.key]: mask });
      }
    }
    if (new TextDecoder().decode(canonicalBytes("studio_parameter_snapshot.v1", snapshot)) === new TextDecoder().decode(canonicalBytes("studio_parameter_snapshot.v1", state.snapshot))) return Object.freeze({ ...state, refusal: null });
    return Object.freeze({ ...state, snapshot, past: Object.freeze([...state.past, state.snapshot].slice(-parameterEditorHistoryLimit)), future: Object.freeze([]), refusal: null });
  } catch (cause: unknown) {
    return Object.freeze({ ...state, refusal: cause instanceof Error ? cause.message : "Parameter edit refused" });
  }
}

/** Address exact draft semantics with the original workspace codec, excluding view state. */
export function parameterDraftDigest(state: ParameterDraft): Promise<string> {
  return canonicalDigest("studio_parameter_draft.v1", { source: state.source, snapshot: state.snapshot });
}

/** Revalidate an external save candidate through original specifications before revision creation. */
export function validateParameterSnapshot(source: ParameterDraftSource, snapshot: ParameterSnapshot): ParameterSnapshot {
  const admitted = createParameterDraft(source);
  if (Object.keys(snapshot.parameters).length !== Object.keys(admitted.snapshot.parameters).length || Object.keys(admitted.snapshot.parameters).some(key => !Object.hasOwn(snapshot.parameters, key))) throw new Error("Revision parameter keys must remain unchanged");
  return freezeSnapshot(admitted.source, snapshot.parameters, snapshot.units, snapshot.trainableMasks);
}
