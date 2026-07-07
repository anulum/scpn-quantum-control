// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — RecomputeCard render tests

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

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

  it("renders a loud mismatch when the digest is forged", async () => {
    render(<RecomputeCard unit={UNIT} loadKernel={async () => forgingKernel} />);
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() => {
      expect(screen.getByText(/claim forged/)).toBeTruthy();
    });
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
});
