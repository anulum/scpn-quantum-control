// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — program-AD replay card component tests

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { ProgramADReplayCard } from "./ProgramADReplayCard";
import {
  type KernelReplay,
  type ProgramAdUnit,
  instantiateProgramAd,
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

/** A second unit, differing in both identity fields. */
function otherUnit(): ProgramAdUnit {
  const base = unit();
  return {
    ...base,
    artifactId: `${base.artifactId}-second`,
    inputSha256: `sha256:${"d".repeat(64)}`,
  };
}

/** A promise whose resolution the test controls. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

afterEach(cleanup);

describe("ProgramADReplayCard", () => {
  it.each([
    { expectedValue: 99 },
    { expectedGradient: [6, 99] },
    { schema: "unsupported" },
    { claimBoundary: "a different claim boundary" },
    { parameterTargets: ["different", "targets"] },
    { inputHex: "00" },
  ])("clears a verdict when the same artifact receives changed content %j", async (change) => {
    const { rerender } = render(
      <ProgramADReplayCard unit={unit()} loadKernel={async () => replay} />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(/match/));
    rerender(<ProgramADReplayCard unit={{ ...unit(), ...change }} loadKernel={async () => replay} />);
    expect(screen.queryByRole("status")).toBeNull();
  });

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

  it("reports a mismatch when the claimed gradient disagrees", async () => {
    // The badge used to read "claim forged". A numeric
    // disagreement is a disagreement; it does not by itself establish forgery.
    const disagreeing: ProgramAdUnit = { ...unit(), expectedGradient: [6, 99] };
    render(<ProgramADReplayCard unit={disagreeing} loadKernel={async () => replay} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() =>
      expect(
        screen.getByText(/recomputed value or gradient does NOT match the committed claim/),
      ).toBeTruthy(),
    );
    expect(screen.queryByText(/forged/)).toBeNull();
  });

  it("refuses a forged replay input even when its expected claim changes too", async () => {
    const base = unit();
    const last = base.inputHex.endsWith("00") ? "01" : "00";
    const forged: ProgramAdUnit = {
      ...base,
      inputHex: `${base.inputHex.slice(0, -2)}${last}`,
      expectedValue: 0,
      expectedGradient: [0, 0],
    };
    render(<ProgramADReplayCard unit={forged} loadKernel={async () => replay} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByText(/SHA-256 binding/)).toBeTruthy());
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

  it("uses a stable boundary when a loader throws a non-Error value", async () => {
    render(
      <ProgramADReplayCard
        unit={unit()}
        loadKernel={async () => {
          throw "not-an-error";
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByText(/unverifiable — kernel load failed/)).toBeTruthy());
  });

  it("drops a completed verdict when the unit prop is replaced", async () => {
    const { rerender } = render(
      <ProgramADReplayCard unit={unit()} loadKernel={async () => replay} />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() =>
      expect(
        screen.getByText(/recomputed value \+ gradient match the committed claim/),
      ).toBeTruthy(),
    );

    rerender(<ProgramADReplayCard unit={otherUnit()} loadKernel={async () => replay} />);

    expect(
      screen.queryByText(/recomputed value \+ gradient match the committed claim/),
    ).toBeNull();
  });

  it("never shows the previous unit's verdict when a load resolves after the swap", async () => {
    // Acceptance: rerender from A to B before the deferred
    // kernel load resolves. A's verdict must not appear beside B's claim.
    const gate = deferred<KernelReplay>();
    const { rerender } = render(
      <ProgramADReplayCard unit={unit()} loadKernel={async () => gate.promise} />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByRole("button").textContent).toMatch(/Recomputing/));

    rerender(<ProgramADReplayCard unit={otherUnit()} loadKernel={async () => gate.promise} />);
    gate.resolve(replay);
    await waitFor(() => expect(screen.getByRole("button").textContent).toMatch(/Recompute/));

    expect(
      screen.queryByText(/recomputed value \+ gradient match the committed claim/),
    ).toBeNull();
    expect(screen.queryByText(/does NOT match/)).toBeNull();
  });
});
