// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — supported source authoring and immutable emission trace

import { useEffect, useRef, useState } from "react";
import { compileProgramSource } from "./programCompiler";
import type { ProgramCompiler } from "./programCompiler";
import { PROGRAM_GATE_SHAPES, parameterValue, sourceSelection } from "./programSource";
import type { CompiledProgram, ProgramCompileResult, ProgramDiagnostic } from "./programSource";

/** Reproducible original source with an ordered two-bit readout. */
export const INITIAL_PROGRAM_SOURCE = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];\n';

/** One retained compilation attempt tied to its original editor revision. */
interface TraceEntry {
  /** Original draft revision, independent of subsequent edits. */
  readonly revision: number;
  /** Immutable actual compiler emission or refusal. */
  readonly result: ProgramCompileResult;
}

/** Structured form construction; admission remains with the native compiler. */
function statement(name: string, parameters: string, qubits: string, destination: string, condition: string): string | null {
  const shape = PROGRAM_GATE_SHAPES[name]!;
  const params = parameters.trim() === "" ? [] : parameters.split(",").map(part => part.trim());
  const operands = qubits.split(",").map(part => part.trim());
  if (params.length !== shape[0] || !params.every(part => /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(part))) return null;
  if (operands.length === 0 || (shape[1] !== 0 && operands.length !== shape[1]) || !operands.every(part => /^(0|[1-9][0-9]*)$/.test(part))) return null;
  if (name === "measure" && !/^(0|[1-9][0-9]*)$/.test(destination)) return null;
  if (condition !== "" && (["measure", "reset", "barrier"].includes(name) || !/^(0|[1-9][0-9]{0,19})$/.test(condition))) return null;
  const prefix = condition === "" ? "" : `if(c==${condition}) `;
  const args = params.length === 0 ? "" : `(${params.join(",")})`;
  const output = name === "measure" ? ` -> c[${destination}]` : "";
  return `${prefix}${name}${args} ${operands.map(part => `q[${part}]`).join(",")}${output};`;
}

/** Display exact native records without executing emitted source. */
function ProgramRecord({ plan }: { /** Current source-bound immutable emission. */ plan: CompiledProgram }) {
  return <section aria-label="Compiled program">
    <h4>Emitted — not executed</h4>
    <p>{plan.num_qubits} qubits · {plan.num_clbits} classical bits · {plan.operations.length} operations</p>
    <p>Source SHA-256: <code>{plan.source_sha256}</code></p>
    <p aria-label="Measurement map">Readout: {plan.measurements.length === 0 ? "none" : plan.measurements.map(([q, c]) => `q[${q}] → c[${c}]`).join(", ")}</p>
    <div className="qsp-table-scroll"><table aria-label="Program IR"><thead><tr><th>Operation</th><th>Parameters (radians)</th><th>Qubits</th><th>Classical effect</th><th>Source</th></tr></thead>
      <tbody>{plan.operations.map((op, index) => <tr key={index}>
        <td>{op.name}</td><td>{op.parameters.map(hex => `${String(parameterValue(hex))} [${hex}]`).join(", ")}</td>
        <td>{op.qubits.join(", ")}</td><td>{op.condition === null ? op.clbits.map(c => `c[${c}]`).join(", ") : `if(c==${op.condition.value})`}</td>
        <td>{op.source_span.line}:{op.source_span.column} [{op.source_span.start}, {op.source_span.end})</td>
      </tr>)}</tbody>
    </table></div>
    <details><summary>Exact emitted record</summary><div className="qsp-table-scroll"><pre>{JSON.stringify(plan, null, 2)}</pre></div></details>
  </section>;
}

/** Author bounded OpenQASM, compile through real WASM and export unchanged source.
 * @param props Optional actual compiler binding for embedding and boundary tests.
 * @returns Editor with source-bound emission, located refusal and retained trace.
 */
export function ProgramEditor({ compiler = compileProgramSource }: {
  /** Native source compiler; production uses the shipped Rust/WASM binding. */
  compiler?: ProgramCompiler;
}) {
  const [source, setSource] = useState(INITIAL_PROGRAM_SOURCE);
  const [plan, setPlan] = useState<CompiledProgram | null>(null);
  const [diagnostic, setDiagnostic] = useState<ProgramDiagnostic | null>(null);
  const [trace, setTrace] = useState<readonly TraceEntry[]>([]);
  const [pending, setPending] = useState(false);
  const [name, setName] = useState("h"), [parameters, setParameters] = useState("");
  const [qubits, setQubits] = useState("0"), [destination, setDestination] = useState("0");
  const [condition, setCondition] = useState(""), [formError, setFormError] = useState<string | null>(null);
  const revision = useRef(0), request = useRef(0), mounted = useRef(true);
  const textarea = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; request.current += 1; }; }, []);

  function changeSource(next: string) {
    revision.current += 1; request.current += 1;
    setSource(next); setPlan(null); setDiagnostic(null); setPending(false); setFormError(null);
  }

  function appendOperation() {
    const next = statement(name, parameters, qubits, destination, condition);
    if (next === null) { setFormError("Use the operation's exact operand and parameter counts, decimal radians and unsigned register indices."); return; }
    changeSource(source + (source.endsWith("\n") ? "" : "\n") + next + "\n");
  }

  async function compile() {
    const originalRevision = revision.current, ticket = ++request.current;
    setPending(true); setPlan(null); setDiagnostic(null);
    let result: ProgramCompileResult;
    try { result = await compiler(source); }
    catch { result = { ok: false, diagnostic: { code: "compiler_failed", message: "Source compiler could not complete this request.", source_span: { start: 0, end: 0, line: 1, column: 1 } } }; }
    if (!mounted.current || ticket !== request.current) return;
    setPending(false);
    setTrace(previous => Object.freeze([...previous.slice(-9), Object.freeze({ revision: originalRevision, result })]));
    if (result.ok) setPlan(result.value);
    else setDiagnostic(result.diagnostic);
  }

  function selectDiagnostic(location: ProgramDiagnostic) {
    const [start, end] = sourceSelection(source, location.source_span);
    textarea.current!.focus(); textarea.current!.setSelectionRange(start, end);
  }

  function exportSource(original: string) {
    const url = URL.createObjectURL(new Blob([original], { type: "text/plain;charset=utf-8" }));
    try {
      const anchor = document.createElement("a"); anchor.href = url; anchor.download = "program.qasm"; anchor.click();
    } finally { URL.revokeObjectURL(url); }
  }

  const selected = diagnostic === null ? null : sourceSelection(source, diagnostic.source_span);
  return <section className="qsp-workspace" aria-label="Program authoring">
    <h3>Program editor</h3>
    <p>Import supported OpenQASM 2.0 or append an operation. Compilation emits a source record; it does not run gates or submit a job.</p>
    <details><summary>Supported source subset</summary><p>Use qreg q[1..8], optional creg c[1..64], qelib1.inc, indexed operands and finite decimal radians. Listed gates, measurement, reset, barrier and whole-register gate conditions are supported. Expressions, custom gates, arbitrary includes and Python are refused.</p></details>
    <label>Program source<textarea aria-label="Program source" ref={textarea} rows={12} spellCheck={false} value={source} onChange={event => changeSource(event.target.value)} /></label>
    <fieldset><legend>Append structured operation</legend>
      <label>Operation<select aria-label="Operation" value={name} onChange={event => { const gate = event.target.value; setName(gate); setParameters(""); setQubits(PROGRAM_GATE_SHAPES[gate]![1] === 2 ? "0,1" : "0"); setCondition(""); setFormError(null); }}>
        {Object.keys(PROGRAM_GATE_SHAPES).map(gate => <option key={gate}>{gate}</option>)}
      </select></label>
      <label>Parameters (decimal radians)<input value={parameters} onChange={event => setParameters(event.target.value)} /></label>
      <label>Qubit indices<input value={qubits} onChange={event => setQubits(event.target.value)} /></label>
      {name === "measure" && <label>Classical destination<input value={destination} onChange={event => setDestination(event.target.value)} /></label>}
      <label>Classical condition c equals (optional)<input value={condition} onChange={event => setCondition(event.target.value)} /></label>
      <button type="button" onClick={appendOperation}>Append operation</button>
      {formError !== null && <p role="alert">{formError}</p>}
    </fieldset>
    <button type="button" disabled={pending} onClick={() => { void compile(); }}>{pending ? "Compiling source…" : "Compile source"}</button>
    <button type="button" disabled={plan === null} onClick={plan === null ? undefined : () => exportSource(plan.source)}>Export exact source</button>
    {plan === null && !pending && diagnostic === null && <p role="status">Draft — no current compiled plan</p>}
    {diagnostic !== null && <div role="alert"><p>{diagnostic.message} ({diagnostic.code}) at {diagnostic.source_span.line}:{diagnostic.source_span.column}</p>
      <button type="button" onClick={() => selectDiagnostic(diagnostic)}>Select offending source</button>
      <div className="qsp-table-scroll"><pre aria-label="Located source diagnostic">{source.slice(0, selected![0])}<mark>{source.slice(selected![0], selected![1])}</mark>{source.slice(selected![1])}</pre></div>
    </div>}
    {plan !== null && <ProgramRecord plan={plan} />}
    <section aria-label="Compilation trace"><h4>Compilation trace</h4>
      <ol>{trace.map((entry, index) => <li key={index}>Revision {entry.revision}: {entry.result.ok ? `emitted — not executed (${entry.result.value.source_sha256})` : `refused (${entry.result.diagnostic.code})`}</li>)}</ol>
    </section>
  </section>;
}
