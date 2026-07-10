// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — 3D Lab panel component tests (ST-21)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { Lab3DPanel } from "./Lab3DPanel";
import {
  type KernelSimulate,
  type KuramotoBounds,
  type KuramotoScenario,
  committedScenario,
  instantiateKuramoto,
} from "./kuramoto";
import { LAB_MAX_OSCILLATORS, LAB_MAX_STEPS } from "./phaseTrajectory";

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

describe("Lab3DPanel with the real kernel", () => {
  it("captures a verified trajectory and renders both scenes", async () => {
    render(<Lab3DPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByText(/capture verified/)).toBeTruthy());
    expect(
      screen.getByLabelText(/phase-space cylinder: oscillator phases over time/),
    ).toBeTruthy();
    expect(
      screen.getByLabelText(/Bloch sphere equator: final phases as spin-coherent points/),
    ).toBeTruthy();
    expect(
      screen.getByText(new RegExp(`N ≤ ${LAB_MAX_OSCILLATORS}, steps ≤ ${LAB_MAX_STEPS}`)),
    ).toBeTruthy();
    // the Bloch caption states the classical-limit claim boundary
    expect(screen.getByText(/No z-axis dynamics, entanglement, or hardware/)).toBeTruthy();
  });

  it("recaptures when the simulation controls move", async () => {
    render(<Lab3DPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByText(/capture verified/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Oscillators N/), { target: { value: "8" } });
    await waitFor(() => expect(screen.getByText(/Oscillators N: 8/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Coupling K/), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText(/Frequency spread/), { target: { value: "2.5" } });
    fireEvent.change(screen.getByLabelText(/Steps/), { target: { value: "60" } });
    await waitFor(() => expect(screen.getByText(/Steps: 60/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Topology/), { target: { value: "networked" } });
    await waitFor(() => expect(screen.getByText(/capture verified/)).toBeTruthy());
  });

  it("re-projects when the orbit controls move", async () => {
    render(<Lab3DPanel scenario={scenario()} loadKernel={async () => realLoaded} />);
    await waitFor(() => expect(screen.getByText(/capture verified/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Azimuth/), { target: { value: "90" } });
    fireEvent.change(screen.getByLabelText(/Elevation/), { target: { value: "-45" } });
    await waitFor(() => expect(screen.getByText(/Azimuth: 90°/)).toBeTruthy());
    expect(screen.getByText(/Elevation: -45°/)).toBeTruthy();
    expect(screen.getByText(/capture verified/)).toBeTruthy();
  });
});

describe("Lab3DPanel scenario clamping", () => {
  it("clamps an oversized committed scenario to the Lab bounds", async () => {
    const oversized: KuramotoScenario = {
      ...scenario(),
      n: LAB_MAX_OSCILLATORS + 8,
      steps: LAB_MAX_STEPS + 40,
    };
    render(<Lab3DPanel scenario={oversized} loadKernel={async () => realLoaded} />);
    await waitFor(() =>
      expect(screen.getByText(new RegExp(`Oscillators N: ${LAB_MAX_OSCILLATORS}`))).toBeTruthy(),
    );
    expect(screen.getByText(new RegExp(`Steps: ${LAB_MAX_STEPS}`))).toBeTruthy();
    await waitFor(() => expect(screen.getByText(/capture verified/)).toBeTruthy());
  });
});

describe("Lab3DPanel degraded paths", () => {
  it("shows a loading state before the kernel resolves", () => {
    render(
      <Lab3DPanel scenario={scenario()} loadKernel={() => new Promise(() => undefined)} />,
    );
    expect(screen.getByText(/loading the WASM simulator kernel/)).toBeTruthy();
  });

  it("surfaces a kernel load failure", async () => {
    render(
      <Lab3DPanel
        scenario={scenario()}
        loadKernel={async () => {
          throw new Error("boom");
        }}
      />,
    );
    await waitFor(() => expect(screen.getByText(/unverifiable — boom/)).toBeTruthy());
  });

  it("surfaces a non-Error kernel load failure with the generic reason", async () => {
    render(
      <Lab3DPanel
        scenario={scenario()}
        loadKernel={async () => {
          throw "not an Error";
        }}
      />,
    );
    await waitFor(() => expect(screen.getByText(/unverifiable — kernel load failed/)).toBeTruthy());
  });

  it("ignores a kernel that settles only after unmount", async () => {
    let resolveKernel!: (value: typeof realLoaded) => void;
    const pending = render(
      <Lab3DPanel
        scenario={scenario()}
        loadKernel={() =>
          new Promise<typeof realLoaded>((resolve) => {
            resolveKernel = resolve;
          })
        }
      />,
    );
    pending.unmount();
    resolveKernel(realLoaded);

    let rejectKernel!: (reason: Error) => void;
    const failing = render(
      <Lab3DPanel
        scenario={scenario()}
        loadKernel={() =>
          new Promise<typeof realLoaded>((_resolve, reject) => {
            rejectKernel = reject;
          })
        }
      />,
    );
    failing.unmount();
    rejectKernel(new Error("late failure"));
    // let both settled promises run their (now inert) continuations
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByText(/late failure/)).toBeNull();
  });

  it("renders a loud boundary when the kernel rejects the request", async () => {
    const simulate: KernelSimulate = () => ({ ok: false, reason: "n out of range" });
    render(
      <Lab3DPanel scenario={scenario()} loadKernel={async () => ({ simulate, bounds: BOUNDS })} />,
    );
    await waitFor(() => expect(screen.getByText(/unverifiable — n out of range/)).toBeTruthy());
  });

  it("renders a loud parity alert when R(t) disagrees with the phases", async () => {
    // A consistent chain and one-shot (so the capture verifies bit-exactly)
    // whose claimed R(t) does NOT match the returned phases: the TypeScript
    // centroid recomputation must catch the contradiction.
    const simulate: KernelSimulate = (request) => ({
      ok: true,
      run: {
        orderParameter: new Float64Array(request.steps + 1).fill(0.9),
        thetaFinal: new Float64Array(request.omega.length),
      },
    });
    render(
      <Lab3DPanel scenario={scenario()} loadKernel={async () => ({ simulate, bounds: BOUNDS })} />,
    );
    await waitFor(() =>
      expect(screen.getByText(/centroid radius diverged from the kernel/)).toBeTruthy(),
    );
  });
});
