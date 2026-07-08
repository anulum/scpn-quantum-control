// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — program-AD replay card component tests (ST-12)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { ProgramADReplayCard } from "./ProgramADReplayCard";
import {
  type KernelReplay,
  type ProgramAdUnit,
  instantiateProgramAd,
  parseProgramAdUnit,
  programAdUnit,
} from "./programAd";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm",
);

let replay: KernelReplay;

function unit(): ProgramAdUnit {
  if (!programAdUnit.ok) throw new Error(programAdUnit.reason);
  return programAdUnit.value;
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  const bytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  replay = await instantiateProgramAd(bytes);
});

afterEach(cleanup);

describe("ProgramADReplayCard", () => {
  it("shows the claimed gradient and boundary before running", () => {
    render(<ProgramADReplayCard unit={unit()} loadKernel={async () => replay} />);
    expect(screen.getByText(/\[6, 2\]/)).toBeTruthy();
    expect(screen.getByText(/not a claim about transcendental/)).toBeTruthy();
  });

  it("recomputes the gradient and reports a match through the real kernel", async () => {
    render(<ProgramADReplayCard unit={unit()} loadKernel={async () => replay} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() =>
      expect(screen.getByText(/recomputed value \+ gradient match/)).toBeTruthy(),
    );
    expect(screen.getByText(/gradient \[6, 2\]/)).toBeTruthy();
  });

  it("reports a mismatch for a forged claim", async () => {
    const forged: ProgramAdUnit = { ...unit(), expectedGradient: [6, 99] };
    render(<ProgramADReplayCard unit={forged} loadKernel={async () => replay} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByText(/claim forged/)).toBeTruthy());
  });

  it("surfaces a kernel load failure as a loud boundary", async () => {
    render(
      <ProgramADReplayCard
        unit={unit()}
        loadKernel={async () => {
          throw new Error("boom");
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByText(/unverifiable — boom/)).toBeTruthy());
  });
});
