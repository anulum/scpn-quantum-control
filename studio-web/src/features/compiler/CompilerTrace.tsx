// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound compiler pass inspector

import { useEffect, useRef, useState } from "react";
import nativeExample from "../../../../data/studio/compiler_trace_demo.json?raw";
import { writeJson } from "../../shared/contracts/jsonTransport";
import { sourceSelection } from "../programs/programSource";
import { parseCompilerTrace } from "./compilerTrace";
import type { CompilerTraceSnapshot, QualifiedTracePass } from "./compilerTrace";

/** Display the declared native input/output mappings and metadata deltas. */
function PassDetails({ pass }: { /** Selected original native pass declaration. */ pass: QualifiedTracePass }) {
  const gates = [...new Set([...Object.keys(pass.before.gates), ...Object.keys(pass.after.gates)])].sort();
  return <section aria-label="Selected compiler pass">
    <h4>{pass.name}</h4>
    <p>Native record digest: <code>{pass.nativeDigest}</code></p>
    <p>Operator error {String(pass.operatorError)} / tolerance {String(pass.tolerance)}; global phase delta {String(pass.phaseDelta)} radians; global phase {pass.allowGlobalPhase ? "allowed" : "refused"}.</p>
    <div className="qsp-table-scroll"><table aria-label="Qubit mapping"><thead><tr><th>Logical</th><th>Input physical</th><th>Output physical</th></tr></thead>
      <tbody>{pass.inputLayout.map((physical, logical) => <tr key={logical}><td>q[{logical}]</td><td>q[{physical}]</td><td>q[{pass.outputLayout[logical]}]</td></tr>)}</tbody>
    </table></div>
    <p aria-label="Classical mapping">Classical output: {pass.classicalLayout.length === 0 ? "none" : pass.classicalLayout.map((physical, logical) => `c[${logical}] → c[${physical}]`).join(", ")}</p>
    <p aria-label="Mapped readout">Mapped readout: {pass.observableMap.length === 0 ? "none" : pass.observableMap.map(([q, c, mappedQ, mappedC]) => `q[${q}] / c[${c}] → q[${mappedQ}] / c[${mappedC}]`).join(", ")}</p>
    <p aria-label="Readout effects">Ordered input readout: {writeJson(pass.inputMeasurements)}; ordered output readout: {writeJson(pass.outputMeasurements)}.</p>
    <div className="qsp-table-scroll"><table aria-label="Gate changes"><thead><tr><th>Gate</th><th>Input</th><th>Output</th><th>Delta</th></tr></thead>
      <tbody>{gates.map(name => <tr key={name}><td>{name}</td><td>{pass.before.gates[name] ?? 0}</td><td>{pass.after.gates[name] ?? 0}</td><td>{(pass.after.gates[name] ?? 0) - (pass.before.gates[name] ?? 0)}</td></tr>)}</tbody>
    </table></div>
    <p>Operation depth: {pass.before.depth} → {pass.after.depth} (delta {pass.after.depth - pass.before.depth}); operations: {pass.before.operationCount} → {pass.after.operationCount}.</p>
    <p>Declared dense payload: statevector {pass.before.statevectorBytes} → {pass.after.statevectorBytes} bytes; reference operator {pass.before.operatorBytes} → {pass.after.operatorBytes} bytes. Allocator, compiler scratch and runtime overhead are outside these declarations.</p>
    <p>Input IR: <code>{pass.input.sha256}</code>; output IR: <code>{pass.output.sha256}</code>.</p>
    <details><summary>Pass parameters</summary><pre aria-label="Pass parameters">{writeJson(pass.parameters)}</pre></details>
    <details><summary>Pass input representation</summary><pre>{pass.input.source}</pre><p>Global phase: {String(pass.input.globalPhase)} radians.</p></details>
    <details><summary>Pass output representation</summary><pre>{pass.output.source}</pre><p>Global phase: {String(pass.output.globalPhase)} radians.</p></details>
    <p>Each representation retains its own source spans. Cross-pass operation correspondence is unavailable; the original source selection stays pinned.</p>
  </section>;
}

/** Inspect immutable native pass records through the original Build view.
 * @returns Read-only importer, original source selection and exact trace export.
 */
export function CompilerTrace() {
  const [draft, setDraft] = useState("");
  const [snapshot, setSnapshot] = useState<CompilerTraceSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [index, setIndex] = useState(0), [selected, setSelected] = useState<number | null>(null);
  const field = useRef<HTMLTextAreaElement>(null), ticket = useRef(0), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; ticket.current++; }; }, []);
  function change(value: string) { ticket.current++; setDraft(value); setError(null); setPending(false); }
  async function inspect(value: string) {
    const current = ++ticket.current;
    setPending(true); setError(null);
    const result = await parseCompilerTrace(value);
    if (!mounted.current || current !== ticket.current) return;
    setPending(false);
    if (!result.ok) { setError(`${result.message} (${result.code}, ${result.path})`); return; }
    setSnapshot(result.value); setIndex(0); setSelected(null);
  }
  function download(admitted: CompilerTraceSnapshot) {
    const url = URL.createObjectURL(new Blob([admitted.text], { type: "application/json;charset=utf-8" }));
    try { const anchor = document.createElement("a"); anchor.href = url; anchor.download = "compiler-trace.json"; anchor.click(); }
    finally { URL.revokeObjectURL(url); }
  }
  const pass = snapshot === null ? null : snapshot.passes[index]!;
  const operations = snapshot === null ? [] : snapshot.passes[0].input.operations;
  const selectedOperation = selected === null ? undefined : operations[selected];
  const selection = snapshot && selectedOperation ? sourceSelection(snapshot.source, selectedOperation.source_span) : null;
  return <section className="qsp-workspace" aria-label="Compiler trace inspector">
    <h3>Compiler trace inspector</h3>
    <p>Import a native compiler trace to inspect pass inputs, outputs, mapping and effects. Import binds metadata; it does not rerun operator qualification, execute emitted IR or submit a job.</p>
    <label>Compiler trace JSON<textarea aria-label="Compiler trace JSON" rows={6} value={draft} spellCheck={false} onChange={event => change(event.target.value)} /></label>
    <button type="button" disabled={pending} onClick={() => { void inspect(draft); }}>{pending ? "Inspecting trace…" : "Inspect trace"}</button>
    <button type="button" onClick={() => { change(nativeExample); void inspect(nativeExample); }}>Open native example</button>
    <button type="button" disabled={snapshot === null} onClick={snapshot === null ? undefined : () => download(snapshot)}>Export admitted trace</button>
    {error !== null && <p role="alert">{error} The prior admitted trace and saved workspace remain unchanged.</p>}
    {snapshot === null && !pending && <p role="status">Missing compiler pass artifact — import a native trace or open the explicit example.</p>}
    {snapshot !== null && <section aria-label="Admitted compiler trace">
      <h4>Emitted — not executed</h4>
      <p>{snapshot.complete ? "Complete trace" : "Incomplete trace"}</p>
      <p>Envelope SHA-256: <code>{snapshot.sha256}</code></p>
      <p>Original source SHA-256: <code>{snapshot.sourceSha256}</code></p>
      <p>Compiler version: <span>{snapshot.compilerVersion}</span>; target: no physical device.</p>
      <details><summary>Exact backend snapshot</summary><pre aria-label="Backend snapshot">{writeJson(snapshot.backend)}</pre></details>
      <label>Compiler pass<select aria-label="Compiler pass" value={index} onChange={event => setIndex(Number(event.target.value))}>
        {snapshot.passes.map((row, position) => <option key={position} value={position}>{position + 1}: {row.state === "qualified" ? row.name : "Missing pass artifact"}</option>)}
      </select></label>
      <label>Original compiler source<textarea aria-label="Original compiler source" ref={field} value={snapshot.source} readOnly rows={8} spellCheck={false} /></label>
      <div>{operations.map((operation, position) => <button type="button" key={position} aria-label={`Select source operation ${position + 1}`} onClick={() => {
        const [start, end] = sourceSelection(snapshot.source, operation.source_span);
        // Textarea values normalise CRLF/CR to LF; retained source/export stay exact.
        const cursor = (offset: number) => snapshot.source.slice(0, offset).replace(/\r\n?/g, "\n").length;
        setSelected(position); field.current!.focus(); field.current!.setSelectionRange(cursor(start), cursor(end));
      }}>{position + 1}. {operation.name} ({operation.source_span.line}:{operation.source_span.column})</button>)}</div>
      {selection !== null && <pre aria-label="Selected original source">{snapshot.source.slice(selection[0], selection[1])}</pre>}
      {pass !== null && pass.state === "qualified" && <PassDetails pass={pass} />}
      {pass !== null && pass.state === "missing" && <p role="status">{pass.reason} Input/output mappings are unavailable for this slot.</p>}
      {snapshot.emittedText === null ? <p>Textual MLIR artifact missing</p> : <details><summary>Textual MLIR — not executed</summary><pre>{snapshot.emittedText}</pre></details>}
    </section>}
  </section>;
}
