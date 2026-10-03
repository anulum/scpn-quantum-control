// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public authoring interactions through actual WASM

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, expect, it, vi } from "vitest";
import { INITIAL_PROGRAM_SOURCE, ProgramEditor } from "./ProgramEditor";
import { instantiateProgramCompiler } from "./programCompiler";
import type { ProgramCompiler } from "./programCompiler";
import type { ProgramCompileResult } from "./programSource";

let compiler: ProgramCompiler;
beforeAll(async () => {
  const buffer = readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));
  compiler = await instantiateProgramCompiler(new Uint8Array(buffer).buffer);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function edit(source: string) { fireEvent.change(screen.getByLabelText("Program source"), { target: { value: source } }); }
async function compile() { fireEvent.click(screen.getByRole("button", { name: "Compile source" })); await waitFor(() => expect(screen.getByRole("button", { name: "Compile source" })).toHaveProperty("disabled", false)); }

it("test_program_authoring_01", async () => {
  render(<ProgramEditor compiler={compiler} />);
  await compile();
  expect(screen.getByLabelText("Measurement map").textContent).toBe("Readout: q[1] → c[0], q[0] → c[1]");
  expect(screen.getByRole("table", { name: "Program IR" }).querySelectorAll("tbody tr")).toHaveLength(4);
  for (const name of ["Program IR table scrolling", "Exact emitted record scrolling"]) {
    const scrolling = screen.getByRole("region", { name, hidden: true });
    expect(scrolling.tabIndex).toBe(0);
  }
  const scrolling = screen.getByRole("region", { name: "Program IR table scrolling" });
  scrolling.focus();
  expect(document.activeElement).toBe(scrolling);
  expect(screen.getByLabelText("Program source")).toHaveProperty("value", INITIAL_PROGRAM_SOURCE);
});

it("test_program_authoring_02", async () => {
  render(<ProgramEditor compiler={compiler} />);
  const source = INITIAL_PROGRAM_SOURCE + "mystery q[0];\n";
  edit(source); await compile();
  expect(screen.getByRole("alert").textContent).toContain("unsupported_operation");
  expect(screen.getByLabelText("Located source diagnostic").querySelector("mark")?.textContent).toBe("mystery");
  const scrolling = screen.getByRole("region", { name: "Located source diagnostic scrolling" });
  expect(scrolling.tabIndex).toBe(0);
  scrolling.focus();
  expect(document.activeElement).toBe(scrolling);
  fireEvent.click(screen.getByRole("button", { name: "Select offending source" }));
  const textarea = screen.getByLabelText<HTMLTextAreaElement>("Program source");
  expect(textarea.selectionStart).toBe(source.indexOf("mystery"));
  expect(textarea.selectionEnd).toBe(source.indexOf("mystery") + 7);
  expect(screen.queryByLabelText("Compiled program")).toBeNull();
  expect(screen.getByRole("button", { name: "Export exact source" })).toHaveProperty("disabled", true);
});

it("test_program_authoring_03", async () => {
  render(<ProgramEditor compiler={compiler} />);
  const source = 'import os\nos.system("touch /tmp/authoring-must-not-evaluate")';
  edit(source); await compile();
  expect(screen.getByRole("alert").textContent).toContain("invalid_source");
  expect(screen.getByLabelText("Program source")).toHaveProperty("value", source);
  expect(screen.queryByLabelText("Compiled program")).toBeNull();
});

it("test_program_authoring_04", async () => {
  render(<ProgramEditor compiler={compiler} />); await compile();
  const original = screen.getByLabelText("Compilation trace").textContent;
  edit(INITIAL_PROGRAM_SOURCE + "x q[0];\n");
  expect(screen.queryByLabelText("Compiled program")).toBeNull();
  expect(screen.getByRole("status").textContent).toContain("Draft");
  expect(screen.getByLabelText("Compilation trace").textContent).toBe(original);
  await compile();
  expect(screen.getByRole("table", { name: "Program IR" }).querySelectorAll("tbody tr")).toHaveLength(5);
  expect(screen.getByLabelText("Compilation trace").querySelectorAll("li")).toHaveLength(2);
});

it("test_program_authoring_05", async () => {
  render(<ProgramEditor compiler={compiler} />); await compile();
  expect(screen.getByRole("heading", { name: "Emitted — not executed" })).toBeTruthy();
  expect(screen.getByLabelText("Compiled program").textContent).toContain('"execution_status": "emitted_not_executed"');
  expect(screen.queryByText(/^executed$/i)).toBeNull();
});

it("appends exact phase and classical control through the structured form", async () => {
  render(<ProgramEditor compiler={compiler} />);
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: "rz" } });
  fireEvent.change(screen.getByLabelText("Parameters (decimal radians)"), { target: { value: "-0.7853981633974492" } });
  fireEvent.change(screen.getByLabelText("Classical condition c equals (optional)"), { target: { value: "2" } });
  fireEvent.click(screen.getByRole("button", { name: "Append operation" })); await compile();
  const row = screen.getByRole("table", { name: "Program IR" }).querySelector("tbody tr:last-child");
  expect(row?.textContent).toContain("bfe921fb54442d20");
  expect(row?.textContent).toContain("if(c==2)");
});

it("refuses form syntax injection without changing the original source", () => {
  render(<ProgramEditor compiler={compiler} />);
  fireEvent.change(screen.getByLabelText("Qubit indices"), { target: { value: "0]; x q[1" } });
  fireEvent.click(screen.getByRole("button", { name: "Append operation" }));
  expect(screen.getByRole("alert").textContent).toContain("operand and parameter counts");
  expect(screen.getByLabelText("Program source")).toHaveProperty("value", INITIAL_PROGRAM_SOURCE);
});

it("appends measurement destinations and preserves their original order", async () => {
  render(<ProgramEditor compiler={compiler} />);
  edit(INITIAL_PROGRAM_SOURCE.trimEnd());
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: "measure" } });
  fireEvent.change(screen.getByLabelText("Classical destination"), { target: { value: "1" } });
  fireEvent.click(screen.getByRole("button", { name: "Append operation" })); await compile();
  expect(screen.getByLabelText("Measurement map").textContent).toBe("Readout: q[1] → c[0], q[0] → c[1], q[0] → c[1]");
});

it("constructs a two-qubit operation with the native operand order", async () => {
  render(<ProgramEditor compiler={compiler} />);
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: "cx" } });
  expect(screen.getByLabelText("Qubit indices")).toHaveProperty("value", "0,1");
  fireEvent.click(screen.getByRole("button", { name: "Append operation" })); await compile();
  expect(screen.getByRole("table", { name: "Program IR" }).querySelector("tbody tr:last-child")?.textContent).toContain("cx");
});

it("shows explicit absence of readout for a supported gate-only source", async () => {
  render(<ProgramEditor compiler={compiler} />);
  edit('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'); await compile();
  expect(screen.getByLabelText("Measurement map").textContent).toBe("Readout: none");
});

it.each([
  { gate: "rz", params: "pi/4", bits: "0", destination: "0", condition: "" },
  { gate: "rz", params: "", bits: "0", destination: "0", condition: "" },
  { gate: "cx", params: "", bits: "0", destination: "0", condition: "" },
  { gate: "measure", params: "", bits: "0", destination: "-1", condition: "" },
  { gate: "measure", params: "", bits: "0", destination: "0", condition: "1" },
  { gate: "h", params: "", bits: "0", destination: "0", condition: "x" },
])("refuses malformed structured $gate operands or parameters without source mutation", ({ gate, params, bits, destination, condition }) => {
  render(<ProgramEditor compiler={compiler} />);
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: gate } });
  fireEvent.change(screen.getByLabelText("Parameters (decimal radians)"), { target: { value: params } });
  fireEvent.change(screen.getByLabelText("Qubit indices"), { target: { value: bits } });
  if (gate === "measure") fireEvent.change(screen.getByLabelText("Classical destination"), { target: { value: destination } });
  fireEvent.change(screen.getByLabelText("Classical condition c equals (optional)"), { target: { value: condition } });
  fireEvent.click(screen.getByRole("button", { name: "Append operation" }));
  expect(screen.getByRole("alert").textContent).toContain("operand and parameter counts");
  expect(screen.getByLabelText("Program source")).toHaveProperty("value", INITIAL_PROGRAM_SOURCE);
});

it("discards a real result that resolves after the source changed", async () => {
  let finish: ((result: ProgramCompileResult) => void) | undefined;
  const actual = await compiler(INITIAL_PROGRAM_SOURCE);
  const delayed: ProgramCompiler = () => new Promise(resolve => { finish = resolve; });
  render(<ProgramEditor compiler={delayed} />);
  fireEvent.click(screen.getByRole("button", { name: "Compile source" }));
  edit(INITIAL_PROGRAM_SOURCE + "x q[1];\n");
  finish!(actual);
  await waitFor(() => expect(screen.queryByLabelText("Compiled program")).toBeNull());
  expect(screen.getByLabelText("Compilation trace").querySelectorAll("li")).toHaveLength(0);
});

it("releases a completed request after the editor unmounts", async () => {
  let finish: ((result: ProgramCompileResult) => void) | undefined;
  const actual = await compiler(INITIAL_PROGRAM_SOURCE);
  const mounted = render(<ProgramEditor compiler={() => new Promise(resolve => { finish = resolve; })} />);
  fireEvent.click(screen.getByRole("button", { name: "Compile source" }));
  mounted.unmount(); finish!(actual);
  await Promise.resolve();
  expect(screen.queryByLabelText("Program authoring")).toBeNull();
});

it("contains an embedding compiler exception without exposing its message", async () => {
  render(<ProgramEditor compiler={async () => { throw new Error("private native detail"); }} />); await compile();
  expect(screen.getByRole("alert").textContent).toContain("compiler_failed");
  expect(screen.getByRole("alert").textContent).not.toContain("private native detail");
});

it("downloads the original source bytes and releases the object URL", async () => {
  const create = vi.fn(() => "blob:authoring-source"), revoke = vi.fn();
  vi.stubGlobal("URL", class extends URL { static override createObjectURL = create; static override revokeObjectURL = revoke; });
  const clicked = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  render(<ProgramEditor compiler={compiler} />); await compile();
  fireEvent.click(screen.getByRole("button", { name: "Export exact source" }));
  expect(create).toHaveBeenCalledOnce(); expect(create.mock.calls[0]).toHaveLength(1);
  expect(clicked).toHaveBeenCalledOnce(); expect(revoke).toHaveBeenCalledWith("blob:authoring-source");
  expect(screen.getByLabelText("Program source")).toHaveProperty("value", INITIAL_PROGRAM_SOURCE);
  vi.unstubAllGlobals();
});
