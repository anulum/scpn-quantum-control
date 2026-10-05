// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — linked parameter editor public interaction tests

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { documentDigest, parseExperimentRevision, parseParameterSpec, writeJson } from "../../shared/contracts";
import type { ExperimentRevision, ParameterSpec, ParseResult } from "../../shared/contracts";
import { ParameterEditor } from "./ParameterEditor";
import { createParameterRevision } from "./parameterRevision";
import type { ParameterDraftSource, ParameterSnapshot } from "./parameterDraft";

afterEach(cleanup);

function take<T>(parsed: ParseResult<T>): T {
  if (!parsed.ok) throw new Error(parsed.message);
  return parsed.value;
}

function binary64(value: number): string {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

function source(key = "K_nm"): ParameterDraftSource {
  const spec: ParameterSpec = take(parseParameterSpec({ schema: "parameter_spec.v1", body: {
    key, dtype: "float64", shape: [3n, 3n], unit: "rad/s", domain: { kind: "finite" },
    default_source: "Independent signed directed matrix oracle", trainable: true, dependency_keys: [],
  }, extensions: {} }));
  const reference = { schema: "declared_test_input.v1", sha256: "a".repeat(64), media_type: "application/json" };
  const revision: ExperimentRevision = take(parseExperimentRevision({ schema: "experiment_revision.v1", body: {
    project_id: "00000000-0000-4000-8000-000000000001", parent_revision_hashes: [],
    problem_ref: reference, program_ref: reference, semantic_settings_ref: { ...reference, schema: "resolved_settings.v1" },
    parameters: { [key]: { dtype: "float64", shape: [3n, 3n], values: [0, -2, 0, 5, 0, 3, -4, 0, 0].map(binary64) } }, input_refs: [],
  }, extensions: { retained_note: "structural UI oracle; no numerical execution claim" } }));
  return { revision, specs: [spec], units: { [key]: "rad/s" } };
}

it("retains Unicode parameter names while assigning resolvable diagram marker identities", async () => {
  const key = "Coupling (A → B)";
  render(<ParameterEditor source={source(key)} />);
  await waitFor(() => expect(screen.getByLabelText("Draft semantic digest").textContent).toMatch(/^[0-9a-f]{64}$/));
  const diagram = screen.getByRole("img", { name: `${key} coupling graph diagram` });
  const marker = diagram.querySelector("marker")!;
  expect(marker.id).not.toMatch(/[\s()]/);
  for (const path of diagram.querySelectorAll(":scope > path")) expect(path.getAttribute("marker-end")).toBe(`url(#${marker.id})`);
  fireEvent.click(screen.getByRole("button", { name: "Edge 1 → 0: -2 rad/s" }));
  expect(screen.getByRole("button", { name: `${key}[0,1] = -2` }).getAttribute("aria-pressed")).toBe("true");
});

function edit(value: string): void {
  fireEvent.change(screen.getByLabelText("Selected value"), { target: { value } });
  fireEvent.click(screen.getByRole("button", { name: "Apply value" }));
}

it("test_parameter_graph_editor_01", async () => {
  render(<ParameterEditor source={source()} />);
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("-7");
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -7" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "K_nm[1,0] = 5" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Edge 1 → 0: -7 rad/s" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Edge 0 → 1: 5 rad/s" })).toBeTruthy();
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-7");
});

it("test_parameter_graph_editor_02", () => {
  render(<ParameterEditor source={source()} />);
  fireEvent.change(screen.getByLabelText("Matrix edit policy"), { target: { value: "symmetric" } });
  expect(screen.getByRole("button", { name: "K_nm[1,0] = 5" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("-7");
  expect(screen.getByRole("button", { name: "K_nm[1,0] = -7" })).toBeTruthy();
});

it("test_parameter_graph_editor_03", () => {
  render(<ParameterEditor source={source()} />);
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("NaN");
  expect(screen.getByRole("alert").textContent).toContain("finite float64");
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -2" })).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Input unit"), { target: { value: "seconds" } });
  edit("4");
  expect(screen.getByRole("alert").textContent).toContain("unit mismatch");
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -2" })).toBeTruthy();
});

it("test_parameter_graph_editor_04", async () => {
  render(<ParameterEditor source={source()} />);
  const digest = screen.getByLabelText("Draft semantic digest");
  await waitFor(() => expect(digest.textContent).toMatch(/^[0-9a-f]{64}$/));
  const before = digest.textContent;
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("-7");
  await waitFor(() => expect(digest.textContent).not.toBe(before));
  fireEvent.click(screen.getByRole("button", { name: "Undo parameter edit" }));
  await waitFor(() => expect(digest.textContent).toBe(before));
  fireEvent.click(screen.getByRole("button", { name: "Redo parameter edit" }));
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -7" })).toBeTruthy();
});

it("test_parameter_graph_editor_05", async () => {
  const original = source();
  const oldBytes = writeJson(original.revision);
  const oldHash = await documentDigest(original.revision);
  const saved: ExperimentRevision[] = [];
  render(<ParameterEditor source={original} onSave={async snapshot => {
    const revision = await createParameterRevision(original, snapshot);
    saved.push(revision);
    return documentDigest(revision);
  }} />);
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("-7");
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(screen.getByText(/Parameter revision saved/)).toBeTruthy());
  expect(writeJson(original.revision)).toBe(oldBytes);
  const revision = saved[0];
  if (!revision) throw new Error("The actual child revision was not created");
  expect(revision.body["parent_revision_hashes"]).toEqual([oldHash]);
  expect(await documentDigest(revision)).not.toBe(oldHash);
});

it("converts an explicitly selected SI prefix only after a separate user action", async () => {
  render(<ParameterEditor source={source()} />);
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  fireEvent.change(screen.getByLabelText("Input unit"), { target: { value: "mrad/s" } });
  fireEvent.change(screen.getByLabelText("Selected value"), { target: { value: "-7000" } });
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -2" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Convert mrad/s → rad/s and apply value" }));
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -7" })).toBeTruthy();
  expect(screen.getByLabelText("Input unit")).toHaveProperty("value", "rad/s");
  expect(screen.getByRole("button", { name: "K_nm[1,0] = 5" })).toBeTruthy();
});

it("shares graph selection with the form and restores mask edits with undo", () => {
  render(<ParameterEditor source={source()} />);
  expect(screen.getByRole("img", { name: "K_nm coupling graph diagram" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Edge 0 → 1: 5 rad/s" }));
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "5");
  const mask = screen.getByLabelText("Selected element trainable");
  expect(mask).toHaveProperty("checked", true);
  fireEvent.click(mask);
  expect(mask).toHaveProperty("checked", false);
  fireEvent.click(screen.getByRole("button", { name: "Undo parameter edit" }));
  expect(mask).toHaveProperty("checked", true);
});

it("renders a signed self-coupling and restores its absent edge with undo", () => {
  render(<ParameterEditor source={source()} />);
  expect(screen.queryByRole("button", { name: "Edge 0 → 0: -1 rad/s" })).toBeNull();
  edit("-1");
  fireEvent.click(screen.getByRole("button", { name: "Edge 0 → 0: -1 rad/s" }));
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-1");
  expect(screen.getByText("0 → 0: -1 rad/s", { selector: "title" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Undo parameter edit" }));
  expect(screen.queryByRole("button", { name: "Edge 0 → 0: -1 rad/s" })).toBeNull();
  expect(screen.getByRole("button", { name: "K_nm[0,0] = 0" })).toBeTruthy();
});

it("keeps unapplied or refused form edits outside the host save boundary", async () => {
  let saves = 0;
  render(<ParameterEditor source={source()} onSave={async snapshot => {
    saves++;
    return documentDigest(await createParameterRevision(source(), snapshot));
  }} />);
  const save = screen.getByRole("button", { name: "Save parameter revision" });
  await waitFor(() => expect(save).toHaveProperty("disabled", false));
  fireEvent.change(screen.getByLabelText("Selected value"), { target: { value: "NaN" } });
  expect(save).toHaveProperty("disabled", true);
  fireEvent.click(save);
  fireEvent.click(screen.getByRole("button", { name: "Apply value" }));
  expect(save).toHaveProperty("disabled", true);
  fireEvent.click(save);
  expect(saves).toBe(0);
  expect(screen.getByRole("button", { name: "K_nm[0,0] = 0" })).toBeTruthy();
});

it("disposes a rejected identity calculation without rendering a stale refusal", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  if (!descriptor) throw new Error("Native WebCrypto descriptor required");
  try {
    if (!Reflect.deleteProperty(globalThis, "crypto")) throw new Error("Owned test platform cannot remove WebCrypto");
    const view = render(<ParameterEditor source={source()} />);
    view.unmount();
    await Promise.resolve();
    await Promise.resolve();
    expect(screen.queryByText(/Draft identity unavailable/)).toBeNull();
  } finally { Object.defineProperty(globalThis, "crypto", descriptor); }
});

it("retains a new source when a disposed host save subsequently completes", async () => {
  const completion: { resolve: ((hash: string) => void) | null } = { resolve: null };
  const original = source();
  const view = render(<ParameterEditor source={original} onSave={() => new Promise<string>(resolve => { completion.resolve = resolve; })} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(completion.resolve).not.toBeNull());
  const revision = { ...original.revision, extensions: { new_source: "Original source replaced during host save" } };
  view.rerender(<ParameterEditor source={{ ...original, revision }} />);
  completion.resolve?.(await documentDigest(original.revision));
  await waitFor(() => expect(screen.getByLabelText("Draft semantic digest").textContent).toMatch(/^[0-9a-f]{64}$/));
  expect(screen.queryByText(/Parameter revision saved/)).toBeNull();
});

it("shows malformed source refusal and recovers with valid source without attempting save", async () => {
  let saves = 0;
  const original = source();
  const component = render(<ParameterEditor source={{ ...original, units: { K_nm: "s" } }} onSave={async () => { saves++; return "0".repeat(64); }} />);
  expect(screen.getByRole("alert").textContent).toContain("source refused");
  expect(screen.queryByRole("button", { name: "Save parameter revision" })).toBeNull();
  component.rerender(<ParameterEditor source={original} />);
  await waitFor(() => expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "0"));
  expect(saves).toBe(0);
});

it("refuses a failed save and keeps the edited matrix for explicit retry", async () => {
  render(<ParameterEditor source={source()} onSave={async () => { throw new Error("test transport/store interruption"); }} />);
  fireEvent.click(screen.getByRole("button", { name: "K_nm[0,1] = -2" }));
  edit("-7");
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(screen.getByText(/revision save refused/)).toBeTruthy());
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -7" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false);
});

it("retains the draft when WebCrypto is unavailable and retries the actual restored provider", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  if (!descriptor) throw new Error("Native WebCrypto descriptor required");
  let saves = 0;
  try {
    if (!Reflect.deleteProperty(globalThis, "crypto")) throw new Error("Owned test platform cannot remove WebCrypto");
    render(<ParameterEditor source={source()} onSave={async snapshot => {
      saves++;
      return documentDigest(await createParameterRevision(source(), snapshot));
    }} />);
    await waitFor(() => expect(screen.getByText(/Draft identity unavailable/)).toBeTruthy());
    expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", true);
    expect(screen.getByRole("button", { name: "K_nm[0,1] = -2" })).toBeTruthy();
    expect(saves).toBe(0);
    Object.defineProperty(globalThis, "crypto", descriptor);
    fireEvent.click(screen.getByRole("button", { name: "Retry draft identity" }));
    await waitFor(() => expect(screen.getByLabelText("Draft semantic digest").textContent).toMatch(/^[0-9a-f]{64}$/));
    expect(screen.queryByText(/Draft identity unavailable/)).toBeNull();
    expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false);
    expect(saves).toBe(0);
  } finally {
    cleanup();
    Object.defineProperty(globalThis, "crypto", descriptor);
  }
});

it("refuses an invalid returned revision identity without exposing callback error text", async () => {
  render(<ParameterEditor source={source()} onSave={async () => "unsupported revision identity"} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(screen.getByText(/revision save refused/)).toBeTruthy());
  expect(screen.queryByText(/unsupported revision identity/)).toBeNull();
  expect(screen.getByRole("button", { name: "K_nm[0,1] = -2" })).toBeTruthy();
});

it("cancels an in-flight host save when the original source changes", async () => {
  const host: { signal: AbortSignal | null; resolve: (() => void) | null } = { signal: null, resolve: null };
  const original = source();
  const callback = async (_snapshot: ParameterSnapshot, signal: AbortSignal): Promise<string> => {
    host.signal = signal;
    await new Promise<void>(resolve => { host.resolve = resolve; });
    if (signal.aborted) throw new Error("disposed source save refused");
    return "a".repeat(64);
  };
  const view = render(<ParameterEditor source={original} onSave={callback} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(host.signal).not.toBeNull());
  const revision = { ...original.revision, extensions: { other_source: "new original identity" } };
  view.rerender(<ParameterEditor source={{ ...original, revision }} onSave={callback} />);
  expect(host.signal?.aborted).toBe(true);
  host.resolve?.();
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  expect(screen.queryByText(/Parameter revision saved/)).toBeNull();
});

it("keeps an empty source vector visible and retains an integer scalar above 2^53", async () => {
  const original = source();
  const emptySpec = take(parseParameterSpec({ ...original.specs[0]!, body: { ...original.specs[0]!.body, shape: [0n] } }));
  const emptyRevision = take(parseExperimentRevision({ ...original.revision, body: { ...original.revision.body,
    parameters: { K_nm: { dtype: "float64", shape: [0n], values: [] } },
  } }));
  const view = render(<ParameterEditor source={{ ...original, revision: emptyRevision, specs: [emptySpec] }} />);
  expect(screen.getByText("Empty parameter; no elements to edit.")).toBeTruthy();
  expect(screen.queryByLabelText("Selected value")).toBeNull();
  const scalarSpec = take(parseParameterSpec({ ...original.specs[0]!, body: { ...original.specs[0]!.body, dtype: "uint64", shape: [], trainable: false } }));
  const scalarRevision = take(parseExperimentRevision({ ...original.revision, body: { ...original.revision.body,
    parameters: { K_nm: { dtype: "uint64", shape: [], values: ["9007199254740993"] } },
  } }));
  view.rerender(<ParameterEditor source={{ ...original, revision: scalarRevision, specs: [scalarSpec] }} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "K_nm[scalar] = 9007199254740993" })).toBeTruthy());
  expect(screen.getByLabelText("Selected element trainable")).toHaveProperty("disabled", true);
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "9007199254740993");
});

it("leaves every coupling edge unselected while another parameter owns the selection", () => {
  const original = source();
  const [couplingSpec] = original.specs;
  const frequencySpec = take(parseParameterSpec({ ...couplingSpec, body: { ...couplingSpec?.body, key: "omega", shape: [2n] } }));
  const revision = take(parseExperimentRevision({ ...original.revision, body: { ...original.revision.body, parameters: {
    K_nm: { dtype: "float64", shape: [3n, 3n], values: [0, -2, 0, 5, 0, 3, -4, 0, 0].map(binary64) },
    omega: { dtype: "float64", shape: [2n], values: [1, 2].map(binary64) },
  } } }));
  render(<ParameterEditor source={{ revision, specs: [...original.specs, frequencySpec], units: { ...original.units, omega: "rad/s" } }} />);
  const edge = screen.getByRole("button", { name: "Edge 0 → 1: 5 rad/s" });
  fireEvent.click(edge);
  expect(edge.getAttribute("aria-pressed")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "omega[1] = 2" }));
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "2");
  expect(edge.getAttribute("aria-pressed")).toBe("false");
  expect(screen.getAllByRole("button", { pressed: true }).map(button => button.textContent)).toEqual(["omega[1] = 2"]);
});
