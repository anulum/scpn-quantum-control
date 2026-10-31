// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — sensing provenance, non-certification and safe exports

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import committed from "../../../../../data/studio/sensing_instrument_20261031.json";
import { SensingInspector } from "./SensingInspector";
import { parseSensingInstrument, sensingInstrument } from "./sensingInstrument";

afterEach(cleanup);

const copy = () => structuredClone(committed);
const firstEntropy = (data: ReturnType<typeof copy>) =>
  data.entropy[0] as (typeof committed.entropy)[number];
const firstCandidate = (data: ReturnType<typeof copy>) =>
  data.candidates[0] as (typeof committed.candidates)[number];

function exportText(): string {
  const link = screen.getByRole("link", { name: "Download public entropy metadata" });
  return decodeURIComponent((link.getAttribute("href") ?? "").split(",").slice(1).join(","));
}

describe("sensing and entropy provenance", () => {
  it("test_sensing_entropy_instrument_01: keeps synthetic data and absent device observations distinct", () => {
    render(<SensingInspector instrument={sensingInstrument} />);
    expect(within(screen.getByLabelText("Entropy provenance")).getByText("synthetic")).toBeTruthy();
    expect(
      screen.getByRole("table", { name: "Measurement-design candidates" }).textContent,
    ).toContain("sync_order");
    fireEvent.change(screen.getByLabelText("Entropy record"), { target: { value: "1" } });
    expect(within(screen.getByLabelText("Entropy provenance")).getByText("device")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("device evidence unavailable");
  });
  it("test_sensing_entropy_instrument_02: passing tests and a signature never certify source entropy", () => {
    const data = copy();
    firstEntropy(data).diagnostics = {
      frequency_passed: true,
      runs_passed: true,
      signature_valid: true,
    };
    render(<SensingInspector instrument={parseSensingInstrument(data)} />);
    expect(screen.getByRole("note").textContent).toContain("Source entropy: unqualified");
    expect(exportText()).toContain('"source_entropy_qualified": false');
    expect(screen.getByLabelText("Recorded diagnostic outcomes").textContent).toContain(
      "not independently verified",
    );
    expect(screen.getByRole("region", { name: "Evidence inspector" })).toBeTruthy();
  });
  it("test_sensing_entropy_instrument_03: keeps missing provenance visible and recovers after refusal", () => {
    const { rerender } = render(
      <SensingInspector instrument={{ ok: false, reason: "unsupported source schema" }} />,
    );
    expect(screen.getByRole("alert").textContent).toContain("unsupported source schema");
    rerender(<SensingInspector instrument={sensingInstrument} />);
    fireEvent.change(screen.getByLabelText("Entropy record"), { target: { value: "1" } });
    expect(screen.getByRole("list", { name: "Evidence limitations" }).textContent).toContain(
      "Missing source provenance",
    );
    expect(exportText()).toContain('"source": null');
  });
  it("test_sensing_entropy_instrument_04: exports an allowlist excluding secret keys, samples and seeds", () => {
    const data = copy();
    Object.assign(firstEntropy(data), {
      seed: "PRIVATE_SEED",
      samples: "PRIVATE_SAMPLES",
      private_key: "PRIVATE_KEY",
    });
    render(<SensingInspector instrument={parseSensingInstrument(data)} />);
    expect(exportText()).not.toContain("PRIVATE_");
    expect(document.body.textContent).not.toContain("PRIVATE_");
    expect(JSON.parse(exportText()).payload_sha256).toBe(firstEntropy(committed).payload_sha256);
  });
  it.each(["source", "extractor", "assumptions"] as const)(
    "refuses whitespace-only %s provenance before rendering or export",
    (field) => {
      const data = copy();
      firstEntropy(data)[field] = " \t\n ";
      render(<SensingInspector instrument={parseSensingInstrument(data)} />);
      expect(screen.getByRole("alert").textContent).toContain(`${field} must be nonblank`);
      expect(screen.queryByRole("link", { name: "Download public entropy metadata" })).toBeNull();
    },
  );
  it("selects the remaining source when refreshed metadata removes a selected record", () => {
    const { rerender } = render(<SensingInspector instrument={sensingInstrument} />);
    fireEvent.change(screen.getByLabelText("Entropy record"), { target: { value: "1" } });
    const data = copy();
    data.entropy = [firstEntropy(data)];
    rerender(<SensingInspector instrument={parseSensingInstrument(data)} />);
    expect(within(screen.getByLabelText("Entropy provenance")).getByText("synthetic")).toBeTruthy();
    expect(exportText()).toContain('"source_kind": "synthetic"');
  });
  it("renders failed diagnostics and unknown provenance without upgrading them", () => {
    const data = copy();
    const raw = firstEntropy(data);
    raw.source_kind = "unknown";
    raw.diagnostics = { frequency_passed: false, runs_passed: false, signature_valid: false };
    render(<SensingInspector instrument={parseSensingInstrument(data)} />);
    expect(screen.getByLabelText("Recorded diagnostic outcomes").textContent).toContain("failed");
    expect(screen.getByRole("note").textContent).toContain("unqualified");
  });
  it.each([
    (data: ReturnType<typeof copy>) => {
      data.schema = "future";
    },
    (data: ReturnType<typeof copy>) => {
      firstEntropy(data).source_kind = "certified";
    },
    (data: ReturnType<typeof copy>) => {
      firstEntropy(data).source_entropy_qualified = true;
    },
    (data: ReturnType<typeof copy>) => {
      firstEntropy(data).claim_boundary = "certified";
    },
    (data: ReturnType<typeof copy>) => {
      firstEntropy(data).payload_sha256 = "bad";
    },
    (data: ReturnType<typeof copy>) => {
      firstEntropy(data).payload_sha256 = null;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).shots = 0;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).prior_variance = 0;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).noise_variance = 0;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).expected_information_gain_nats = -1;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).posterior_variance = 0;
    },
    (data: ReturnType<typeof copy>) => {
      firstCandidate(data).posterior_variance = 10;
    },
    (data: ReturnType<typeof copy>) => {
      data.entropy.push(firstEntropy(data));
    },
    (data: ReturnType<typeof copy>) => {
      data.entropy = [];
    },
    (data: ReturnType<typeof copy>) => {
      data.candidates = [];
    },
  ])("refuses malformed or contradictory source data", (mutate) => {
    const data = copy();
    mutate(data);
    expect(parseSensingInstrument(data).ok).toBe(false);
  });
});
