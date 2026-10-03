// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — compiler inspector real component contracts

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import nativeTrace from "../../../../data/studio/compiler_trace_demo.json?raw";
import nativeCases from "../../../../data/studio/compiler_trace_cases.json?raw";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson, writeJson } from "../../shared/contracts/jsonTransport";
import { CompilerTrace } from "./CompilerTrace";

/** Alter an unsigned imported metadata declaration and bind its resulting envelope. */
async function changedTrace(change: (body: Record<string, unknown>) => void): Promise<string> {
  const wire = readJson(nativeTrace) as Record<string, unknown>;
  change(wire["body"] as Record<string, unknown>);
  wire["sha256"] = await canonicalDigest("studio.compiler-trace.v1", {
    schema: wire["schema"], body: wire["body"], extensions: wire["extensions"],
  });
  return writeJson(wire);
}

/** Drive the actual public JSON import controls. */
function importText(text: string): void {
  fireEvent.change(screen.getByLabelText("Compiler trace JSON"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect trace" }));
}

/** Load the actual native producer example through the production importer. */
async function example(): Promise<void> {
  fireEvent.click(screen.getByRole("button", { name: "Open native example" }));
  await screen.findByRole("table", { name: "Qubit mapping" });
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("test_compiler_trace_inspector_01: displays the actual qualified physical swap and unchanged observable map", async () => {
  render(<CompilerTrace />);
  await example();
  expect(screen.getByRole("table", { name: "Qubit mapping" }).textContent).toContain("q[0]q[0]q[1]");
  expect(screen.getByLabelText("Mapped readout").textContent).toContain("q[0] / c[1] → q[1] / c[1]");
  expect(screen.getByText(/Import binds metadata/)).toBeTruthy();
  expect(screen.getByText("2.5.1")).toBeTruthy();
  expect(screen.getByRole("table", { name: "Gate changes" }).textContent).toContain("ry");
  expect(screen.getByLabelText("Pass parameters").textContent).toContain("output_layout");
});

it("test_compiler_trace_inspector_02: missing artifacts stay explicit and malformed imports preserve the admitted source", async () => {
  render(<CompilerTrace />);
  expect(screen.getByRole("status").textContent).toContain("Missing compiler pass artifact");
  await example();
  const original = (screen.getByLabelText("Original compiler source") as HTMLTextAreaElement).value;
  const missing = await changedTrace(body => {
    const passes = body["passes"] as unknown[];
    passes[1] = { state: "missing", reason: "Native pass artifact was not supplied." };
    body["complete"] = false;
  });
  importText(missing);
  await screen.findByText("Incomplete trace");
  fireEvent.change(screen.getByLabelText("Compiler pass"), { target: { value: "1" } });
  expect(screen.getByRole("status").textContent).toContain("Native pass artifact was not supplied");
  importText('{"schema":"studio.compiler-trace.v2"}');
  await screen.findByRole("alert");
  expect((screen.getByLabelText("Original compiler source") as HTMLTextAreaElement).value).toBe(original);
  expect(screen.getByRole("button", { name: "Export admitted trace" }).hasAttribute("disabled")).toBe(false);
  await example();
  expect(screen.queryByRole("alert")).toBeNull();
});

it("test_compiler_trace_inspector_03: pinned Unicode source selection survives every pass navigation", async () => {
  render(<CompilerTrace />);
  await example();
  fireEvent.click(screen.getByRole("button", { name: "Select source operation 1" }));
  const field = screen.getByLabelText("Original compiler source") as HTMLTextAreaElement;
  const start = field.selectionStart, end = field.selectionEnd;
  const originalSelection = field.value.slice(start, end);
  expect(originalSelection).toMatch(/^ry\(/);
  expect(field.value).toContain("α🧪");
  fireEvent.change(screen.getByLabelText("Compiler pass"), { target: { value: "1" } });
  expect(field.selectionStart).toBe(start);
  expect(field.selectionEnd).toBe(end);
  expect(field.value.slice(start, end)).toBe(originalSelection);
  fireEvent.change(screen.getByLabelText("Compiler pass"), { target: { value: "0" } });
  expect(screen.getByLabelText("Selected original source").textContent).toBe(originalSelection);
});

it("test_compiler_trace_inspector_04: exports the exact admitted backend snapshot and keeps execution unavailable", async () => {
  const create = vi.fn<(blob: Blob) => string>(() => "blob:compiler-trace");
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: create });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  render(<CompilerTrace />);
  await example();
  expect(screen.getByText("Emitted — not executed")).toBeTruthy();
  expect(screen.getByText("Textual MLIR artifact missing")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Export admitted trace" }));
  const blob = create.mock.calls[0]![0];
  const text = await new Promise<string>(resolve => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.readAsText(blob);
  });
  expect(text).toBe(nativeTrace);
  expect((readJson(text) as Record<string, unknown>)["sha256"]).toBe("20aa768ffd8c536e3e5e6c05b222dee3d0ecc91b4e65fbdc0cd3b2e9beaebaa6");
  await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:compiler-trace"));
});

it("renders actual native lowering deltas and explicit empty/readout-only effects", async () => {
  const cases = readJson(nativeCases) as Record<string, unknown>;
  render(<CompilerTrace />);
  importText(writeJson(cases["lowering"]));
  await screen.findByText("Textual MLIR — not executed");
  expect(screen.getByRole("table", { name: "Gate changes" }).textContent).toContain("h10-1");
  expect(screen.getByLabelText("Classical mapping").textContent).toContain("none");
  expect(screen.getByLabelText("Mapped readout").textContent).toContain("none");
  importText(writeJson(cases["empty"]));
  await screen.findByRole("heading", { name: "empty" });
  expect(screen.getByRole("table", { name: "Gate changes" }).querySelectorAll("tbody tr")).toHaveLength(0);
  expect(screen.queryByRole("button", { name: "Select source operation 1" })).toBeNull();
  expect(screen.getByText(/global phase refused/)).toBeTruthy();
  importText(writeJson(cases["readout"]));
  await screen.findByRole("heading", { name: "readout" });
  expect(screen.getByLabelText("Readout effects").textContent).toContain("[[0.0,0.0]]");
});

/** Hold one actual digest result while every numeric binding uses real WebCrypto. */
function holdFirstDigest(): { release: () => void; completed: () => number } {
  const original = crypto.subtle.digest.bind(crypto.subtle);
  let release!: () => void, completed = 0, first = true;
  const gate = new Promise<void>(resolve => { release = resolve; });
  vi.spyOn(crypto.subtle, "digest").mockImplementation(async (algorithm, bytes) => {
    const result = await original(algorithm, bytes);
    if (first) { first = false; await gate; }
    completed++;
    return result;
  });
  return { release, completed: () => completed };
}

it("drops a fully hashed stale response after the draft changes and recovers", async () => {
  const digest = holdFirstDigest();
  render(<CompilerTrace />);
  fireEvent.click(screen.getByRole("button", { name: "Open native example" }));
  expect(screen.getByRole("button", { name: "Inspecting trace…" }).hasAttribute("disabled")).toBe(true);
  expect(screen.queryByRole("status")).toBeNull();
  fireEvent.change(screen.getByLabelText("Compiler trace JSON"), { target: { value: "new draft" } });
  await act(async () => { digest.release(); });
  await waitFor(() => expect(digest.completed()).toBe(9));
  expect((screen.getByLabelText("Compiler trace JSON") as HTMLTextAreaElement).value).toBe("new draft");
  expect(screen.getByRole("button", { name: "Export admitted trace" }).hasAttribute("disabled")).toBe(true);
  expect(screen.queryByRole("region", { name: "Admitted compiler trace" })).toBeNull();
  await example();
  expect(screen.getByRole("region", { name: "Admitted compiler trace" })).toBeTruthy();
});

it("finishes real hashing after unmount without restoring an abandoned inspector", async () => {
  const digest = holdFirstDigest();
  const view = render(<CompilerTrace />);
  fireEvent.click(screen.getByRole("button", { name: "Open native example" }));
  view.unmount();
  await act(async () => { digest.release(); });
  await waitFor(() => expect(digest.completed()).toBe(9));
  expect(screen.queryByRole("region", { name: "Compiler trace inspector" })).toBeNull();
});

it("selects native Unicode scalar spans in a CRLF source through the actual textarea", async () => {
  const cases = readJson(nativeCases) as Record<string, unknown>;
  render(<CompilerTrace />);
  importText(writeJson(cases["crlf"]));
  await screen.findByRole("heading", { name: "crlf" });
  fireEvent.click(screen.getByRole("button", { name: "Select source operation 1" }));
  const field = screen.getByLabelText("Original compiler source") as HTMLTextAreaElement;
  expect(field.value.slice(field.selectionStart, field.selectionEnd)).toBe("h q[0];");
  expect(screen.getByLabelText("Selected original source").textContent).toBe("h q[0];");
});
