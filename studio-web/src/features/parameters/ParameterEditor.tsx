// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — linked parameter forms, matrices and sparse graph view

import { useEffect, useId, useMemo, useReducer, useRef, useState } from "react";
import { canonicalBytes } from "../../shared/contracts";
import { createParameterDraft, parameterDraftDigest, parameterDraftReducer, parameterElementText, parameterInputUnits } from "./parameterDraft";
import type { ParameterDraft, ParameterDraftSource, ParameterSnapshot } from "./parameterDraft";
import { CouplingTable } from "./CouplingTable";

/** Source declarations and a host-owned validated revision-save boundary. */
export interface ParameterEditorProps {
  /** Original ParameterSpecs, immutable revision and explicit units. Remount on source change. */ readonly source: ParameterDraftSource;
  /** Original archive/store owner returns the committed revision digest and observes disposal cancellation. */ readonly onSave?: (snapshot: ParameterSnapshot, signal: AbortSignal) => Promise<string>;
}

/** Linked views over one validated immutable draft; this component performs no numerical execution. */
export function ParameterEditor({ source, onSave }: ParameterEditorProps) {
  const prepared = useMemo(() => {
    try {
      const initial = createParameterDraft(source);
      return { ok: true as const, initial, key: new TextDecoder().decode(canonicalBytes("studio_parameter_source.v1", initial.source)) };
    } catch {
      return { ok: false as const };
    }
  }, [source]);
  if (!prepared.ok) return <section aria-label="Linked parameter editor"><h3>Parameter editor</h3><p role="alert">Parameter source refused; verify v1 specifications, dtype, shape, units, masks and editor bounds. Saved data remains unchanged.</p></section>;
  return <ParameterDraftEditor key={prepared.key} initial={prepared.initial} {...(onSave ? { onSave } : {})} />;
}

function ParameterDraftEditor({ initial, onSave }: { readonly initial: ParameterDraft; readonly onSave?: ParameterEditorProps["onSave"] }) {
  const [draft, dispatch] = useReducer(parameterDraftReducer, initial);
  const source = draft.source;
  const [text, setText] = useState("");
  const [unit, setUnit] = useState("");
  const [digest, setDigest] = useState("");
  const [identityRefused, setIdentityRefused] = useState(false);
  const [identityAttempt, retryIdentity] = useReducer((value: number) => value + 1, 0);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const generation = useRef(0);
  const mounted = useRef(true);
  const cancellation = useRef<AbortController | null>(null);
  const id = useId();
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; ++generation.current; cancellation.current?.abort(); };
  }, []);
  useEffect(() => {
    const epoch = ++generation.current;
    setDigest("");
    setIdentityRefused(false);
    void parameterDraftDigest(draft).then(value => { if (mounted.current && epoch === generation.current) setDigest(value); }).catch(() => {
      if (mounted.current && epoch === generation.current) setIdentityRefused(true);
    });
  }, [draft.source, draft.snapshot, identityAttempt]);
  useEffect(() => {
    const selection = draft.selection;
    setText(selection ? parameterElementText(draft.snapshot.parameters[selection.key]!, selection.index) : "");
    setUnit(selection ? draft.snapshot.units[selection.key]! : "");
  }, [draft.selection, draft.snapshot]);
  const selection = draft.selection;
  const selected = selection ? draft.snapshot.parameters[selection.key]! : null;
  const selectedSpec = source.specs.find(spec => spec.body["key"] === selection?.key);
  const unapplied = selection !== null && (text !== parameterElementText(selected!, selection.index) || unit !== draft.snapshot.units[selection.key]);
  const save = async (saveRevision: NonNullable<ParameterEditorProps["onSave"]>) => {
    setSaving(true);
    setMessage("");
    try {
      const controller = new AbortController();
      cancellation.current = controller;
      const hash = await saveRevision(draft.snapshot, controller.signal);
      if (!/^[0-9a-f]{64}$/.test(hash) || hash.length !== 64) throw new Error("Committed revision identity required");
      if (mounted.current) setMessage(`Parameter revision saved: ${hash}. Previous results retain their original revision.`);
    } catch {
      if (mounted.current) setMessage("Parameter revision save refused; previous saved workspace retained. Review the workspace refusal and retry.");
    } finally { if (mounted.current) setSaving(false); }
  };
  return <section aria-label="Linked parameter editor" id={id}>
    <h3>Parameter editor</h3>
    <p>Forms, matrices and directed sparse edges share one draft. Values retain their source units, dtype and shape. Editing does not run a solver or submit a job.</p>
    <label>Matrix edit policy <select value={draft.policy} disabled={saving} onChange={event => dispatch({ type: "policy", policy: event.target.value as "directed" | "symmetric" })}>
      <option value="directed">Directed: edit the selected coefficient only</option>
      <option value="symmetric">Symmetric: explicitly edit both coefficients</option>
    </select></label>
    <button type="button" disabled={saving || draft.past.length === 0} onClick={() => dispatch({ type: "undo" })}>Undo parameter edit</button>
    <button type="button" disabled={saving || draft.future.length === 0} onClick={() => dispatch({ type: "redo" })}>Redo parameter edit</button>
    <output aria-label="Draft semantic digest" role="note">{digest}</output>
    {identityRefused && <div role="alert"><p>Draft identity unavailable; previous saved workspace retained.</p>
      <button type="button" disabled={saving} onClick={() => retryIdentity()}>Retry draft identity</button></div>}
    {draft.refusal && <p role="alert">{draft.refusal}</p>}
    <fieldset disabled={saving}>
      <legend>Source parameter forms and matrices</legend>
      {Object.entries(draft.snapshot.parameters).map(([key, value], parameterIndex) => <section key={key} aria-label={`${key} parameter`}>
        <h4>{key} · {value.dtype} · [{value.shape.map(String).join(", ")}] · {draft.snapshot.units[key]}</h4>
        <p>Source: {String(source.specs.find(spec => spec.body["key"] === key)!.body["default_source"])}</p>
        {value.values.length === 0 && <p>Empty parameter; no elements to edit.</p>}
        <div role="group" aria-label={`${key} values`}>
          {value.values.map((_, index) => {
            const address = value.shape.length === 2 ? `${Math.floor(index / Number(value.shape[1]))},${index % Number(value.shape[1])}` : value.shape.length === 0 ? "scalar" : String(index);
            const display = parameterElementText(value, index);
            return <button type="button" key={index} aria-pressed={selection?.key === key && selection.index === index} onClick={() => dispatch({ type: "select", key, index })}>{key}[{address}] = {display}</button>;
          })}
        </div>
        {value.shape.length === 2 && value.shape[0] === value.shape[1] && <div role="group" aria-label={`${key} sparse coupling graph`}>
          <p>Coefficient [i,j] is shown as j → i. Zero coefficients have no sparse edge; use the matrix to add one.</p>
          <svg role="img" aria-label={`${key} coupling graph diagram`} viewBox="0 0 240 240" width="240" height="240">
            <title>{key}: directed signed coefficients</title>
            <defs><marker id={`${id}-${parameterIndex}-arrow`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" /></marker></defs>
            {value.values.map((_, index) => {
              const display = parameterElementText(value, index);
              if (display === "0" || display === "-0") return null;
              const n = Number(value.shape[0]);
              const from = index % n;
              const to = Math.floor(index / n);
              const fromAngle = 2 * Math.PI * from / n - Math.PI / 2;
              const toAngle = 2 * Math.PI * to / n - Math.PI / 2;
              const x1 = 120 + 85 * Math.cos(fromAngle), y1 = 120 + 85 * Math.sin(fromAngle);
              const x2 = 120 + 85 * Math.cos(toAngle), y2 = 120 + 85 * Math.sin(toAngle);
              const path = from === to ? `M ${x1 - 8} ${y1} C ${x1 - 40} ${y1 - 35} ${x1 + 40} ${y1 - 35} ${x1 + 8} ${y1}` : `M ${x1} ${y1} Q ${(x1 + x2) / 2 + (y2 - y1) / 8} ${(y1 + y2) / 2 + (x1 - x2) / 8} ${x2} ${y2}`;
              return <path key={index} d={path} fill="none" stroke="currentColor" strokeWidth={selection?.key === key && selection.index === index ? 3 : 1}
                strokeDasharray={display.startsWith("-") ? "4 3" : undefined} markerEnd={`url(#${id}-${parameterIndex}-arrow)`}><title>{from} → {to}: {display} {draft.snapshot.units[key]}</title></path>;
            })}
            {Array.from({ length: Number(value.shape[0]) }, (_, node) => {
              const angle = 2 * Math.PI * node / Number(value.shape[0]) - Math.PI / 2;
              const x = 120 + 85 * Math.cos(angle), y = 120 + 85 * Math.sin(angle);
              return <g key={node}><circle cx={x} cy={y} r="10" fill="Canvas" stroke="currentColor" /><text x={x} y={y} textAnchor="middle" dominantBaseline="central" fill="currentColor" fontSize="10">{node}</text></g>;
            })}
          </svg>
          <CouplingTable parameterKey={key} value={value} unit={draft.snapshot.units[key]!}
            selectedIndex={selection?.key === key ? selection.index : null}
            onSelect={index => dispatch({ type: "select", key, index })} />
        </div>}
      </section>)}
      {selection && selected && <div role="group" aria-label="Selected parameter form">
        <p>Selected {selection.key}, row-major element {selection.index}; domain {String(selectedSpec!.body["domain"] && (selectedSpec!.body["domain"] as Readonly<Record<string, unknown>>)["kind"])}</p>
        <label>Selected value <input value={text} onChange={event => setText(event.target.value)} /></label>
        <label>Input unit <input value={unit} onChange={event => setUnit(event.target.value)} /></label>
        <button type="button" onClick={() => dispatch({ type: "value", key: selection.key, index: selection.index, text, unit })}>Apply value</button>
        {unit !== draft.snapshot.units[selection.key] && <button type="button" disabled={selected.dtype !== "float64" || !parameterInputUnits(draft.snapshot.units[selection.key]!).includes(unit)}
          onClick={() => dispatch({ type: "value", key: selection.key, index: selection.index, text, unit, convert: true })}>Convert {unit} → {draft.snapshot.units[selection.key]} and apply value</button>}
        <p>Supported explicit input units: {parameterInputUnits(draft.snapshot.units[selection.key]!).join(", ")}. Conversion uses binary64 SI-prefix multiplication; stored units remain unchanged. Other dimensions and frequency conventions are refused.</p>
        <label>Selected element trainable <input type="checkbox" checked={draft.snapshot.trainableMasks[selection.key]![selection.index]} disabled={selectedSpec!.body["trainable"] !== true}
          onChange={event => dispatch({ type: "mask", key: selection.key, index: selection.index, enabled: event.target.checked })} /></label>
      </div>}
    </fieldset>
    {unapplied && <p role="note">Form changes are unapplied; apply a valid value before saving.</p>}
    {onSave && <button type="button" disabled={saving || draft.refusal !== null || !digest || unapplied} onClick={() => { void save(onSave); }}>Save parameter revision</button>}
    {message && <p role="note" aria-live="polite">{message}</p>}
  </section>;
}
