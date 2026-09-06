// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — RecomputeCard render tests

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { RecomputeCard } from "./RecomputeCard";
import type { KernelRecompute, RecomputeUnit } from "./recompute";

const UNIT: RecomputeUnit = {
  schema: "studio.xy-compile-recompute.v1",
  verifiabilityMode: "recompute",
  exactnessClass: "bit-exact",
  claimedDigest: `sha256:${"a".repeat(64)}`,
  inputHex: "01020304",
};

const matchingKernel: KernelRecompute = () => ({ ok: true, digest: UNIT.claimedDigest });
const forgingKernel: KernelRecompute = () => ({
  ok: true,
  digest: `sha256:${"b".repeat(64)}`,
});

/** A second unit, differing in both identity fields. */
const OTHER_UNIT: RecomputeUnit = {
  ...UNIT,
  claimedDigest: `sha256:${"c".repeat(64)}`,
  inputHex: "0a0b0c0d",
};

/** A promise whose resolution the test controls. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

afterEach(cleanup);

describe("RecomputeCard", () => {
  it("shows the signed claim before any recompute", () => {
    render(<RecomputeCard unit={UNIT} loadKernel={async () => matchingKernel} />);
    expect(screen.getByText(UNIT.claimedDigest)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Recompute in browser/ })).toBeTruthy();
  });

  it("renders a match verdict when the recomputed digest agrees", async () => {
    render(<RecomputeCard unit={UNIT} loadKernel={async () => matchingKernel} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByText(/recomputed digest matches the signed claim/)).toBeTruthy();
    });
  });

  it("renders a loud mismatch when the digest disagrees", async () => {
    // The badge used to read "claim forged". A digest
    // disagreement is a disagreement; it does not by itself establish who
    // produced the claim or why it differs.
    render(<RecomputeCard unit={UNIT} loadKernel={async () => forgingKernel} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(
        screen.getByText(/recomputed digest does NOT match the signed claim/),
      ).toBeTruthy();
    });
    expect(screen.queryByText(/forged/)).toBeNull();
  });

  it("renders unverifiable when the grade is stripped", async () => {
    const stripped: RecomputeUnit = { ...UNIT, exactnessClass: "tolerance" };
    render(<RecomputeCard unit={stripped} loadKernel={async () => matchingKernel} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByText(/unverifiable/)).toBeTruthy();
    });
  });

  it("renders a loud error when the kernel fails to load", async () => {
    render(
      <RecomputeCard
        unit={UNIT}
        loadKernel={async () => {
          throw new Error("kernel fetch failed: 404");
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain("kernel fetch failed: 404");
    });
  });

  it("falls back to a generic reason when a non-Error is thrown", async () => {
    render(
      <RecomputeCard
        unit={UNIT}
        loadKernel={async () => {
          throw "opaque failure";
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain("kernel load failed");
    });
  });

  it("drops a completed verdict when the unit prop is replaced", async () => {
    const { rerender } = render(
      <RecomputeCard unit={UNIT} loadKernel={async () => matchingKernel} />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByText(/recomputed digest matches the signed claim/)).toBeTruthy();
    });

    rerender(<RecomputeCard unit={OTHER_UNIT} loadKernel={async () => matchingKernel} />);

    expect(screen.queryByText(/recomputed digest matches the signed claim/)).toBeNull();
    expect(screen.getByText(OTHER_UNIT.claimedDigest)).toBeTruthy();
  });

  it("never shows the previous unit's verdict when a load resolves after the swap", async () => {
    // Acceptance: rerender from A to B before the deferred
    // kernel load resolves. A's verdict must not appear beside B's claim.
    const gate = deferred<KernelRecompute>();
    const { rerender } = render(
      <RecomputeCard unit={UNIT} loadKernel={async () => gate.promise} />,
    );
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByRole("button").textContent).toMatch(/Recomputing/));

    rerender(<RecomputeCard unit={OTHER_UNIT} loadKernel={async () => gate.promise} />);
    gate.resolve(matchingKernel);
    await waitFor(() => expect(screen.getByText(OTHER_UNIT.claimedDigest)).toBeTruthy());

    expect(screen.queryByText(/recomputed digest matches the signed claim/)).toBeNull();
    expect(screen.queryByText(/does NOT match/)).toBeNull();
  });
});
