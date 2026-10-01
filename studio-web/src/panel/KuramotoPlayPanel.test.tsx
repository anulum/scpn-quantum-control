// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Kuramoto Play panel component tests

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { KuramotoPlayPanel, controlsToRequest, sparklinePoints } from "./KuramotoPlayPanel";
import {
  type KernelSimulate,
  type KuramotoBounds,
  type KuramotoScenario,
  committedScenario,
  instantiateKuramoto,
} from "./kuramoto";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
);

const BOUNDS: KuramotoBounds = { maxOscillators: 128, maxSteps: 4096 };

let realLoaded: { simulate: KernelSimulate; bounds: KuramotoBounds };

function scenario(): KuramotoScenario {
  if (!committedScenario.ok) throw new Error(committedScenario.reason);
  return committedScenario.value;
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  const bytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  realLoaded = await instantiateKuramoto(bytes);
});

afterEach(cleanup);

describe("KuramotoPlayPanel with the real kernel", () => {
  it("integrates R(t) and verifies the committed ground truth", async () => {
    render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByText(/R initial/)).toBeTruthy());
    expect(screen.getByText(/verified against the committed ground truth/)).toBeTruthy();
    expect(screen.getByLabelText(/order parameter over time/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Edit source parameters in Workspace" }).getAttribute("href")).toBe("#/workspace");
  });

  it("re-integrates when a control changes", async () => {
    render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByText(/Oscillators N/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Oscillators N/), { target: { value: "24" } });
    await waitFor(() => expect(screen.getByText(/Oscillators N: 24/)).toBeTruthy());
    // move every remaining control so its handler runs
    fireEvent.change(screen.getByLabelText(/Coupling K/), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText(/Frequency spread/), { target: { value: "2.5" } });
    fireEvent.change(screen.getByLabelText(/Steps/), { target: { value: "120" } });
    await waitFor(() => expect(screen.getByText(/Coupling K: 5.00/)).toBeTruthy());
    // switch topology as well
    fireEvent.change(screen.getByLabelText(/Topology/), { target: { value: "networked" } });
    await waitFor(() => expect(screen.getByText(/R final/)).toBeTruthy());
  });
});

describe("KuramotoPlayPanel degraded paths", () => {
  it("shows a loading state before the kernel resolves", () => {
    render(
      <KuramotoPlayPanel scenario={scenario()} loadKernel={() => new Promise(() => undefined)} />,
    );
    expect(screen.getByText(/loading the WASM simulator kernel/)).toBeTruthy();
  });

  it("surfaces a kernel load failure", async () => {
    render(
      <KuramotoPlayPanel
        scenario={scenario()}
        loadKernel={async () => {
          throw new Error("boom");
        }}
      />,
    );
    await waitFor(() => expect(screen.getByText(/unverifiable — boom/)).toBeTruthy());
  });

  it("renders a loud boundary when the kernel rejects the request", async () => {
    const simulate: KernelSimulate = () => ({ ok: false, reason: "n out of range" });
    render(
      <KuramotoPlayPanel
        scenario={scenario()}
        loadKernel={async () => ({ simulate, bounds: BOUNDS })}
      />,
    );
    await waitFor(() => expect(screen.getByText(/unverifiable — n out of range/)).toBeTruthy());
    expect(screen.getByText(/committed ground truth not reproduced/)).toBeTruthy();
  });
});

describe("pure helpers", () => {
  it("builds mean-field and networked requests", () => {
    const mean = controlsToRequest({ mode: "mean-field", n: 3, coupling: 1, spread: 1, steps: 5 });
    expect(mean.omega).toHaveLength(3);
    expect("kNm" in mean).toBe(false);
    const net = controlsToRequest({ mode: "networked", n: 3, coupling: 1.5, spread: 1, steps: 5 });
    expect(net.kNm).toHaveLength(9);
    expect(net.kNm![0]).toBe(0); // zero diagonal
    // a single oscillator collapses the spread cleanly
    const solo = controlsToRequest({ mode: "mean-field", n: 1, coupling: 1, spread: 2, steps: 5 });
    expect(solo.omega).toEqual([0]);
    expect(solo.theta0).toEqual([0]);
  });

  it("maps a series to a polyline and clamps out-of-range values", () => {
    expect(sparklinePoints(new Float64Array([0.5]), 300, 80)).toBe("");
    const points = sparklinePoints(new Float64Array([0, 1.5, -0.2]), 300, 80);
    expect(points.split(" ")).toHaveLength(3);
    // 1.5 clamps to the top (y=0), -0.2 clamps to the bottom (y=height)
    expect(points).toContain("150.00,0.00");
    expect(points).toContain("300.00,80.00");
  });
});


describe("Kuramoto Play resource admission with the real kernel", () => {
  it("shows the resource projection and clears a trajectory when an explicit policy refuses", async () => {
    let runs = 0;
    const observed = { ...realLoaded, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
    const base = { source: "component declared test policy", addressableBytes: 0xffff_ffffn, memoryBytes: 4n * 1024n * 1024n, overheadBytes: 0n, workUnits: 1000000000n };
    const component = render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => observed} resourcePolicy={base} />);
    await waitFor(() => expect(screen.getByLabelText("Resource plan")).toBeTruthy());
    expect(screen.getByLabelText("order parameter over time")).toBeTruthy();
    const previousRuns = runs;
    component.rerender(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => observed} resourcePolicy={{ ...base, memoryBytes: 0n }} />);
    await waitFor(() => expect(screen.queryByLabelText("order parameter over time")).toBeNull());
    expect(runs).toBe(previousRuns);
    expect(screen.getByText(/Resource plan refused: declared_storage_exceeds_budget/)).toBeTruthy();
    expect(screen.getByText(/allocator\/object overhead excluded|component declared test policy/)).toBeTruthy();
  });

  it("recalculates the declared memory when topology changes without reducing float64 precision", async () => {
    render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByLabelText("Resource plan")).toBeTruthy());
    const previous = screen.getByLabelText("Resource plan").textContent;
    fireEvent.change(screen.getByLabelText(/Topology/), { target: { value: "networked" } });
    expect(screen.getByLabelText("Resource plan").textContent).not.toBe(previous);
    expect(screen.getByLabelText("Resource plan").textContent).toContain("float64");
    expect(screen.getByLabelText("order parameter over time")).toBeTruthy();
  });
});


it("allows an explicit smaller configuration after user resource refusal", async () => {
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
  await waitFor(() => expect(screen.getByLabelText("Resource plan")).toBeTruthy());
  fireEvent.change(screen.getByLabelText(/Oscillators N/), { target: { value: "16" } });
  fireEvent.change(screen.getByLabelText(/Steps:/), { target: { value: "100" } });
  fireEvent.change(screen.getByLabelText(/Topology/), { target: { value: "networked" } });
  fireEvent.change(screen.getByLabelText("Memory ceiling (KiB)"), { target: { value: "0" } });
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(screen.getByText(/committed ground truth not evaluated/)).toBeTruthy();
  expect(screen.queryByText("committed ground truth not reproduced")).toBeNull();
  fireEvent.change(screen.getByLabelText("Memory ceiling (KiB)"), { target: { value: "1" } });
  const smaller = screen.getByRole("button", { name: /Apply smaller supported configuration/ });
  fireEvent.click(smaller);
  expect(screen.getByLabelText("order parameter over time")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Memory ceiling (KiB)"), { target: { value: "5000" } });
  expect(screen.getAllByText(/memory ceiling must be an integer/).length).toBeGreaterThan(0);
});


it("clears the actual trajectory on unsupported or invalid wall-clock admission", async () => {
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
  await waitFor(() => expect(screen.getByLabelText("order parameter over time")).toBeTruthy());
  fireEvent.change(screen.getByLabelText("Wall-clock ceiling (ms; optional)"), { target: { value: "1" } });
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(screen.getByLabelText("Resource plan").textContent).toContain("wall_clock_admission_unavailable");
  expect(screen.queryByRole("button", { name: /Apply smaller supported configuration/ })).toBeNull();
  fireEvent.change(screen.getByLabelText("Wall-clock ceiling (ms; optional)"), { target: { value: "-1" } });
  expect(screen.getAllByText(/wall-clock ceiling must be a positive integer/).length).toBeGreaterThan(0);
  fireEvent.change(screen.getByLabelText("Wall-clock ceiling (ms; optional)"), { target: { value: "" } });
  expect(screen.getByLabelText("order parameter over time")).toBeTruthy();
});


it("refuses malformed policy data without reading an accessor or running the native kernel", async () => {
  let reads = 0;
  let runs = 0;
  const policy = { source: "source policy", addressableBytes: 0xffff_ffffn, memoryBytes: 4096n, workUnits: 1000n, overheadBytes: 0n };
  Object.defineProperty(policy, "memoryBytes", { enumerable: true, get() { reads++; return 4096n; } });
  const loaded = { ...realLoaded, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => loaded} resourcePolicy={policy} />);
  await waitFor(() => expect(screen.getAllByText(/source policy must be available/).length).toBeGreaterThan(0));
  expect(reads).toBe(0);
  expect(runs).toBe(0);
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
});


it("refuses out-of-bounds initial controls and leaves committed ground truth unevaluated", async () => {
  let runs = 0;
  const loaded = { ...realLoaded, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
  render(<KuramotoPlayPanel scenario={{ ...scenario(), n: realLoaded.bounds.maxOscillators + 1 }} loadKernel={async () => loaded} />);
  await waitFor(() => expect(screen.getAllByText(/request exceeds declared kernel bounds/).length).toBeGreaterThan(0));
  expect(screen.getByText(/committed ground truth not evaluated/)).toBeTruthy();
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(runs).toBe(0);
});

it("refuses caller kernel-bound traps during admission without entering native execution", async () => {
  let reads = 0;
  let runs = 0;
  const bounds = new Proxy(realLoaded.bounds, {
    get(target, key, receiver) {
      if (key === "maxOscillators" && ++reads <= 2) throw "caller bounds unavailable";
      return Reflect.get(target, key, receiver);
    },
  });
  const policy = { source: "caller policy", addressableBytes: 0xffff_ffffn, memoryBytes: 4096n, workUnits: 1000000n, overheadBytes: 0n };
  const loaded = { ...realLoaded, bounds, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => loaded} resourcePolicy={policy} />);
  await waitFor(() => expect(screen.getAllByText(/resource metadata refused/).length).toBeGreaterThan(0));
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(runs).toBe(0);
});


it.each([-1n, 1 as unknown as bigint])("refuses invalid source memory ceilings %s before native execution", async memoryBytes => {
  const policy = { source: "caller policy", addressableBytes: 0xffff_ffffn, memoryBytes, workUnits: 1000000n, overheadBytes: 0n };
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => realLoaded} resourcePolicy={policy} />);
  await waitFor(() => expect(screen.getAllByText(/source policy must be available/).length).toBeGreaterThan(0));
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
});

it.each(["resolve", "reject-error", "reject-string"])("keeps kernel completion after unmount inert: %s", async terminal => {
  let settle!: () => void;
  const pending = new Promise<typeof realLoaded>((resolve, reject) => {
    settle = () => terminal === "resolve" ? resolve(realLoaded) : reject(terminal === "reject-error" ? new Error("load interrupted") : "load interrupted");
  });
  const component = render(<KuramotoPlayPanel scenario={scenario()} loadKernel={() => pending} />);
  expect(screen.getByText(/loading the WASM simulator kernel/)).toBeTruthy();
  component.unmount();
  await act(async () => { settle(); await pending.catch(() => undefined); });
  expect(component.container.textContent).toBe("");
});

it("reports a non-Error loader refusal while mounted", async () => {
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => { throw "unavailable"; }} />);
  await waitFor(() => expect(screen.getByText(/kernel load failed/)).toBeTruthy());
});


it("retains explicit refusal when the caller policy cannot enumerate its fields", async () => {
  let runs = 0;
  const observed = { ...realLoaded, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
  const policy = new Proxy({ source: "caller policy", addressableBytes: 0xffff_ffffn, memoryBytes: 4096n, workUnits: 1000000n, overheadBytes: 0n }, {
    ownKeys() { throw new Error("caller policy inaccessible"); },
  });
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => observed} resourcePolicy={policy} />);
  await waitFor(() => expect(screen.getAllByText(/source policy must be available/).length).toBeGreaterThan(0));
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(runs).toBe(0);
});


it("preserves an unknown source memory ceiling as refusal without executing a trajectory", async () => {
  let runs = 0;
  const observed = { ...realLoaded, simulate: (request: Parameters<KernelSimulate>[0]) => { runs++; return realLoaded.simulate(request); } };
  render(<KuramotoPlayPanel scenario={scenario()} loadKernel={async () => observed} resourcePolicy={{ source: "unknown source memory", addressableBytes: 0xffff_ffffn, memoryBytes: null, workUnits: 1000000n, overheadBytes: 0n }} />);
  await waitFor(() => expect(screen.getAllByText(/memory_limit_unknown/).length).toBeGreaterThan(0));
  expect(screen.queryByLabelText("order parameter over time")).toBeNull();
  expect(screen.getByText(/committed ground truth not evaluated/)).toBeTruthy();
  expect(runs).toBe(0);
});
