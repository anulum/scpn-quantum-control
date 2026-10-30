// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Build live bench: tabs, linked selection, estimate, run and session runs

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  cleanup,
  configure,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { controlsToRequest } from "../../shared/browser/KuramotoPlayPanel";
import type { KuramotoKernel, KuramotoScenario } from "../../shared/browser/kuramoto";
import { committedScenario, instantiateKuramoto } from "../../shared/browser/kuramoto";
import { BuildWorkbench, kibibytes } from "./BuildWorkbench";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
);

let kernel: KuramotoKernel;

beforeAll(async () => {
  configure({ asyncUtilTimeout: 12000 });
  const buffer = readFileSync(WASM_PATH);
  kernel = await instantiateKuramoto(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer,
  );
});

beforeEach(() => {
  vi.stubGlobal("Worker", BuiltKernelWorker);
  vi.stubGlobal("crypto", webcrypto);
});

afterEach(async () => {
  cleanup();
  await waitFor(() => expect(BuiltKernelWorker.activeCount).toBe(0));
  vi.unstubAllGlobals();
});

/** The committed scenario the Build stage starts from. */
function scenario(): KuramotoScenario {
  if (!committedScenario.ok) throw new Error(committedScenario.reason);
  return committedScenario.value;
}

/**
 * Render Build with a supplied kernel loader.
 *
 * @param loadKernel - Kernel the instrument receives.
 * @param committed - Scenario whose ground truth the run is checked against.
 */
function renderBuild(
  loadKernel: () => Promise<KuramotoKernel>,
  committed: KuramotoScenario = scenario(),
): void {
  render(
    <BuildWorkbench
      scenario={committed}
      editors={<section aria-label="Program authoring">program editor</section>}
      evidence={<section aria-label="Compile recomputation evidence">evidence</section>}
      compareHref="#/results?project=p"
      loadKernel={loadKernel}
    />,
  );
}

const pending = () => new Promise<KuramotoKernel>(() => undefined);

/** Direction of each node's phase glyph on the coupling graph, in node order. */
function glyphAngles(graph: HTMLElement): number[] {
  return [...graph.querySelectorAll(".qsp-graph-phase")].map((glyph) => {
    const step = /l(-?[\d.]+) (-?[\d.]+)$/.exec(glyph.getAttribute("d") ?? "");
    if (step === null) throw new Error("phase glyph without a direction");
    return Math.atan2(-Number(step[2]), Number(step[1]));
  });
}

/** Phase of each dot on a phase circle centred at (90, 90), in oscillator order. */
function circleAngles(circle: HTMLElement): number[] {
  return [...circle.querySelectorAll(".qsp-phase-dot")].map((dot) =>
    Math.atan2(90 - Number(dot.getAttribute("cy")), Number(dot.getAttribute("cx")) - 90),
  );
}

/** Smallest angle between two phases. */
function phaseGap(a: number, b: number): number {
  return Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
}

/** Largest gap between two equally long phase lists. */
function largestGap(drawn: readonly number[], expected: ArrayLike<number>): number {
  expect(drawn).toHaveLength(expected.length);
  return Math.max(...drawn.map((angle, index) => phaseGap(angle, expected[index] as number)));
}

it("opens on the program, moves between tabs with the keyboard and shows the exact request", () => {
  renderBuild(pending);
  const tabs = within(screen.getByRole("tablist", { name: "Build views" })).getAllByRole("tab");
  expect(tabs.map((tab) => [tab.textContent, tab.getAttribute("aria-selected")])).toEqual([
    ["Program", "true"],
    ["Coupling", "false"],
    ["Script", "false"],
  ]);
  expect(screen.getByRole("region", { name: "Program authoring" })).toBeTruthy();
  fireEvent.keyDown(tabs[0] as HTMLElement, { key: "ArrowRight" });
  expect(tabs[1]?.getAttribute("aria-selected")).toBe("true");
  expect(document.activeElement).toBe(tabs[1]);
  fireEvent.keyDown(tabs[1] as HTMLElement, { key: "End" });
  expect(tabs[2]?.getAttribute("aria-selected")).toBe("true");
  const request = controlsToRequest({ ...scenario(), spread: 1 });
  expect(screen.getByText(/"mode": "mean-field"/).textContent).toBe(
    JSON.stringify(request, null, 2),
  );
  fireEvent.keyDown(tabs[2] as HTMLElement, { key: "ArrowRight" });
  expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.keyDown(tabs[0] as HTMLElement, { key: "ArrowLeft" });
  expect(tabs[2]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.keyDown(tabs[2] as HTMLElement, { key: "Home" });
  expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.keyDown(tabs[0] as HTMLElement, { key: "a" });
  expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
  expect(screen.getByRole("link", { name: "Compare runs" }).getAttribute("href")).toBe(
    "#/results?project=p",
  );
  expect(screen.getAllByText("loading the WASM simulator kernel…").length).toBeGreaterThan(0);
  expect(screen.getByText("Loading the WASM simulator kernel.")).toBeTruthy();
});

it("shows initial phases on the graph and circle, and links matrix and graph selections", () => {
  renderBuild(pending);
  fireEvent.click(screen.getByRole("tab", { name: "Coupling" }));
  const graph = screen.getByRole("img", {
    name: /Coupling graph of 12 oscillators, mean-field, each node showing its initial phase/,
  });
  expect(graph.querySelectorAll(".qsp-phase-dot")).toHaveLength(12);
  const initial = controlsToRequest({ ...scenario(), spread: 1 }).theta0;
  expect(largestGap(glyphAngles(graph), initial)).toBeLessThan(2e-3);
  expect(screen.getByRole("img", { name: /Initial phases of 12 oscillators/ })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Matrix K" }));
  expect(screen.getByRole("button", { name: "Matrix K" }).getAttribute("aria-pressed")).toBe(
    "true",
  );
  const cell = screen.getByRole("button", { name: "K[0,3] = 0.208" });
  fireEvent.click(cell);
  expect(cell.getAttribute("aria-pressed")).toBe("true");
  expect(screen.getByRole("button", { name: "K[3,0] = 0.208" }).dataset["linked"]).toBe("true");
  expect(screen.getByRole("button", { name: "K[1,1] = 0.000" })).toHaveProperty("disabled", true);
  expect(
    screen.getAllByRole("status").some((node) => node.textContent?.includes("[0,3] [3,0]")),
  ).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Graph" }));
  const selected = screen.getByRole("img", { name: /Coupling graph/ });
  expect(selected.querySelectorAll(".qsp-edge-selected")).toHaveLength(1);
  expect(selected.querySelectorAll(".qsp-node-selected")).toHaveLength(2);
  fireEvent.change(screen.getByLabelText("j"), { target: { value: "0" } });
  expect(selected.querySelectorAll(".qsp-edge-selected")).toHaveLength(1);
  fireEvent.change(screen.getByLabelText("j"), { target: { value: "5" } });
  fireEvent.change(screen.getByLabelText("Coupling i"), { target: { value: "2" } });
  fireEvent.click(screen.getByRole("button", { name: "Matrix K" }));
  expect(screen.getByRole("button", { name: "K[2,5] = 0.208" }).getAttribute("aria-pressed")).toBe(
    "true",
  );
  expect(screen.getByRole("button", { name: "K[5,2] = 0.208" }).dataset["linked"]).toBe("true");
});

it("summarises above the drawn limit and drops a selection the controls removed", async () => {
  renderBuild(async () => kernel);
  await screen.findByRole("button", { name: "Run simulation" });
  fireEvent.click(screen.getByRole("tab", { name: "Coupling" }));
  fireEvent.click(screen.getByRole("button", { name: "Matrix K" }));
  fireEvent.click(screen.getByRole("button", { name: "K[0,11] = 0.208" }));
  fireEvent.change(screen.getByLabelText(/Oscillators N/), { target: { value: "30" } });
  expect(screen.getByText(/30 × 30 matrix: K\/N = 0.0833/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Graph" }));
  expect(screen.getByText(/30 oscillators, every distinct pair coupled with K\/N/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText(/Oscillators N/), { target: { value: "6" } });
  expect(screen.getByText("Select a coupling here or in Matrix K.")).toBeTruthy();
}, 60000);

it("records a finished run, colours the result by final phases and overlays an earlier run", async () => {
  renderBuild(async () => kernel);
  const result = screen.getByRole("region", { name: "Live result" });
  await within(result).findByRole("img", { name: /Order parameter R over time for current/ });
  const panel = screen.getByRole("region", { name: "Bench panel" });
  await within(panel).findByRole("tab", { name: "Runs (1)" });
  expect(within(result).getByText("R initial").nextElementSibling?.textContent).toMatch(
    /^0\.\d{3}$/,
  );
  expect(
    within(result).getByText("Time to 90 % of final R").nextElementSibling?.textContent,
  ).toMatch(/^\d+\.\d{2}$/);
  fireEvent.click(screen.getByRole("tab", { name: "Coupling" }));
  const final = circleAngles(screen.getByRole("img", { name: /Final phases of 12 oscillators/ }));
  const graph = screen.getByRole("img", { name: /each node showing its final phase/ });
  const glyphs = glyphAngles(graph);
  expect(largestGap(glyphs, final)).toBeLessThan(2e-3);
  expect(
    largestGap(glyphs, controlsToRequest({ ...scenario(), spread: 1 }).theta0),
  ).toBeGreaterThan(0.05);
  fireEvent.click(screen.getByRole("button", { name: "Run simulation" }));
  await within(panel).findByRole("tab", { name: "Runs (2)" });
  fireEvent.click(within(panel).getByLabelText("Overlay run 1"));
  expect(
    within(result).getByRole("img", { name: "Order parameter R over time for current, run 1" }),
  ).toBeTruthy();
  fireEvent.click(within(panel).getByRole("tab", { name: "Log" }));
  expect(within(panel).getAllByText(/^run \d · mean-field · N=12/)).toHaveLength(2);
  fireEvent.click(within(panel).getByRole("tab", { name: /Messages/ }));
  expect(within(panel).getByText("Verified against the committed ground truth")).toBeTruthy();
  fireEvent.click(within(panel).getByRole("tab", { name: "Progress" }));
  expect(within(panel).getByRole("status").textContent).toMatch(/^(Idle|Running)/);
  fireEvent.click(within(panel).getByRole("tab", { name: "Runs (2)" }));
  fireEvent.click(within(panel).getByRole("button", { name: "Clear session runs" }));
  expect(
    within(panel).getAllByText(/No runs yet in this session/, { ignore: "[hidden] *" }),
  ).toHaveLength(1);
  expect(within(panel).getByRole("tab", { name: "Runs" })).toBeTruthy();
}, 60000);

it("shows the admitted estimate, a refusal in the same card, a message and the gated run", async () => {
  renderBuild(async () => kernel);
  const estimate = await screen.findByRole("region", { name: "Resource estimate" });
  const run = await within(estimate).findByRole("button", { name: "Run simulation" });
  await waitFor(() => expect(run).toHaveProperty("disabled", false));
  expect(within(estimate).getByText("admitted")).toBeTruthy();
  expect(within(estimate).getByText("Local limit").nextElementSibling?.textContent).toBe("within");
  fireEvent.click(screen.getByRole("button", { name: "Estimate" }));
  expect(document.activeElement).toBe(estimate);
  // Closed disclosures hold no controls the keyboard cannot reach.
  expect(within(estimate).queryByLabelText(/Memory ceiling/)).toBeNull();
  fireEvent.click(within(estimate).getByText("Declared plan and ceilings"));
  const ceiling = await within(estimate).findByLabelText(/Memory ceiling/);
  fireEvent.change(ceiling, { target: { value: "1" } });
  await waitFor(() => expect(within(estimate).getByText("refused")).toBeTruthy());
  expect(within(estimate).getByText(/^Refused: /)).toBeTruthy();
  expect(within(estimate).getByRole("button", { name: "Run simulation" })).toHaveProperty(
    "disabled",
    true,
  );
  const panel = screen.getByRole("region", { name: "Bench panel" });
  fireEvent.click(within(panel).getByRole("tab", { name: "Messages (1)" }));
  expect(within(panel).getByText(/^Resource plan refused: /)).toBeTruthy();
  fireEvent.click(within(panel).getByRole("tab", { name: "Progress" }));
  expect(within(panel).getByRole("status").textContent).toBe(
    "Waiting for an admitted resource plan; nothing runs.",
  );
  fireEvent.change(within(estimate).getByLabelText(/Memory ceiling/), { target: { value: "-1" } });
  await waitFor(() =>
    expect(within(estimate).getByRole("alert").textContent).toContain("Resource plan refused"),
  );
}, 60000);

it("reports a kernel that fails to load in every place it would act", async () => {
  renderBuild(async () => {
    throw new Error("kernel unavailable for the test");
  });
  await waitFor(() =>
    expect(screen.getAllByText("unverifiable — kernel unavailable for the test")).toHaveLength(3),
  );
  const panel = screen.getByRole("region", { name: "Bench panel" });
  fireEvent.click(within(panel).getByRole("tab", { name: "Messages (1)" }));
  expect(
    within(panel).getByText("Kernel unavailable: kernel unavailable for the test"),
  ).toBeTruthy();
  fireEvent.click(within(panel).getByRole("tab", { name: "Progress" }));
  expect(within(panel).getByRole("status").textContent).toBe("Kernel unavailable; nothing runs.");
});

it("refuses kernel limits it cannot read instead of drawing controls past them", async () => {
  renderBuild(async () => ({ ...kernel, bounds: { maxOscillators: 0, maxSteps: 0 } }));
  await waitFor(() =>
    expect(screen.getAllByText("unverifiable — original kernel limits unavailable")).toHaveLength(
      3,
    ),
  );
  expect(screen.queryByRole("button", { name: "Run simulation" })).toBeNull();
});

it("writes declared sizes in kibibytes and names an undeclared size", () => {
  expect(kibibytes(357488n)).toBe("349.1 KiB");
  expect(kibibytes(null)).toBe("unknown");
});

it("reports a ground truth the run does not reproduce", async () => {
  const committed = scenario();
  renderBuild(async () => kernel, {
    ...committed,
    expectedOrderParameter: committed.expectedOrderParameter.map(() => 2),
  });
  const panel = screen.getByRole("region", { name: "Bench panel" });
  fireEvent.click(within(panel).getByRole("tab", { name: /Messages/ }));
  await within(panel).findByText("Committed ground truth not reproduced");
}, 60000);
