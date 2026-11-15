// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — attestation inspector, agreement display and local file check

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { AttestationInspector } from "./AttestationInspector";
import { attestationInstrument } from "./attestationInstrument";
import type { AttestationInstrument } from "./attestationInstrument";
import { maxEnvelopeBytes } from "./attestationVerifier";
import EvidenceView from "./EvidenceView";
import { pythonCanonicalText } from "./pythonCanonical";

afterEach(cleanup);

const committed = (): AttestationInstrument => {
  if (!attestationInstrument.ok) throw new Error(attestationInstrument.reason);
  return attestationInstrument.value;
};
const sealedText = () => {
  const envelope = committed().cases[0]?.envelope;
  if (envelope === undefined) throw new Error("no sealed case");
  return pythonCanonicalText({
    claim_id: envelope.claimId,
    claim_axes: envelope.claimAxes,
    content: envelope.content,
    content_digest: envelope.contentDigest,
  });
};
const choose = (text: string, size?: number) => {
  const file = new File([text], "envelope.json", { type: "application/json" });
  if (size !== undefined) Object.defineProperty(file, "size", { value: size });
  fireEvent.change(screen.getByLabelText(/Envelope file/u), { target: { files: [file] } });
};

it("test_attestation_browser_03: shows each browser verdict beside the Python verdict", async () => {
  render(<AttestationInspector instrument={attestationInstrument} />);
  const table = screen.getByRole("table", { name: "Attestation cases" });
  await waitFor(() => expect(within(table).queryByText("checking…")).toBeNull());
  const rows = within(table).getAllByRole("row").slice(1);
  expect(rows.map((row) => row.querySelector("[data-agrees]")?.textContent)).toEqual(
    Array(7).fill("Agrees"),
  );
  expect(rows[0]?.querySelectorAll('[data-status="VERIFIED"]')).toHaveLength(2);
  expect(rows[1]?.textContent).toContain("content.A_even.mean 0.9185 -> 0.9285");
  expect(rows[4]?.textContent).toContain("none");
  await waitFor(() =>
    expect(screen.getByLabelText("Canonical form").textContent).toContain("for all 6 samples"),
  );
  expect(screen.getByLabelText("Sealed source").textContent).toContain(committed().source.sha256);
});

it("says so when a recorded verdict or oracle text disagrees with the browser", async () => {
  const data = committed();
  const [first, ...rest] = data.cases;
  const [sample, ...others] = data.oracle;
  if (first === undefined || sample === undefined) throw new Error("empty instrument");
  const changed: AttestationInstrument = {
    ...data,
    cases: [{ ...first, python: { ...first.python, status: "FORGED" } }, ...rest],
    oracle: [{ ...sample, pythonText: `${sample.pythonText} ` }, ...others],
  };
  render(<AttestationInspector instrument={{ ok: true, value: changed }} />);
  await waitFor(() => expect(screen.getAllByText("Disagrees")).toHaveLength(1));
  await waitFor(() =>
    expect(screen.getByLabelText("Canonical form").textContent).toContain(
      "differs from the Python owner for: edge floats",
    ),
  );
});

it("renders an unverifiable block for data that failed its guard", () => {
  render(<AttestationInspector instrument={{ ok: false, reason: "schema" }} />);
  expect(screen.getByRole("alert").textContent).toContain("attestation_instrument_20261115.json");
});

it("test_attestation_browser_04: checks a local envelope file without uploading it", async () => {
  render(<AttestationInspector instrument={attestationInstrument} />);
  choose(sealedText());
  await waitFor(() => expect(screen.getByText(/unsigned envelope intact/u)).toBeTruthy());
  fireEvent.click(screen.getByLabelText("Require a signature"));
  choose(sealedText());
  await waitFor(() => expect(screen.getByText(/signature required but missing/u)).toBeTruthy());
  choose("{");
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain("Refused before a verdict"),
  );
  choose(sealedText(), maxEnvelopeBytes + 1);
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain(`exceeds ${maxEnvelopeBytes} bytes`),
  );
  fireEvent.change(screen.getByLabelText(/Envelope file/u), { target: { files: [] } });
  expect(screen.getByRole("alert")).toBeTruthy();
});

it("focuses the attestation instrument on its deep link only", async () => {
  const { unmount } = render(<EvidenceView focusAttestation={true} />);
  await waitFor(() => expect(document.activeElement?.id).toBe("/evidence/attestation"));
  unmount();
  render(<EvidenceView focusAttestation={false} />);
  expect(document.activeElement?.id).not.toBe("/evidence/attestation");
  expect(screen.getByRole("heading", { name: "Evidence catalogue" })).toBeTruthy();
});

it("drops verdicts that arrive after the inspector has closed", async () => {
  const { unmount } = render(<AttestationInspector instrument={attestationInstrument} />);
  unmount();
  await new Promise((resolve) => setTimeout(resolve, 100));
  expect(screen.queryByRole("table", { name: "Attestation cases" })).toBeNull();
});
