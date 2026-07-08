// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Kuramoto Play panel component tests (ST-11)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
