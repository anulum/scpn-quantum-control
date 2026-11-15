// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — attestation case table, canonical-form oracle and file check

import { useEffect, useId, useState } from "react";
import type { AttestationCase, OracleRow } from "./attestationInstrument";
import { checkEnvelopeText, maxEnvelopeBytes, verifyEnvelope } from "./attestationVerifier";
import type { AttestationVerdict, EnvelopeCheck } from "./attestationVerifier";
import { pythonCanonicalText, sha256Hex } from "./pythonCanonical";

/** Status text with the shared chip; the word alone identifies the outcome. */
export function StatusChip({ verdict }: { readonly verdict: AttestationVerdict }) {
  const tone = verdict.status === "VERIFIED" ? "qsp-status-ok" : "qsp-status-refused";
  return (
    <span className={`qsp-status ${tone}`} data-status={verdict.status}>
      {verdict.status}
    </span>
  );
}

function short(digest: string): string {
  return digest === "" ? "none" : `${digest.slice(0, 12)}…`;
}

function agreement(browser: AttestationVerdict, python: AttestationVerdict): boolean {
  return (
    browser.status === python.status &&
    browser.reason === python.reason &&
    browser.expectedDigest === python.expectedDigest &&
    browser.observedDigest === python.observedDigest
  );
}

function CaseRow({
  item,
  browser,
}: {
  readonly item: AttestationCase;
  readonly browser: AttestationVerdict | undefined;
}) {
  return (
    <tr data-case={item.id}>
      <th scope="row">{item.title}</th>
      <td>{item.change}</td>
      <td>{browser === undefined ? "checking…" : <StatusChip verdict={browser} />}</td>
      <td>
        <StatusChip verdict={item.python} />
      </td>
      <td>{browser === undefined ? "…" : short(browser.expectedDigest)}</td>
      <td data-agrees={browser === undefined ? "pending" : String(agreement(browser, item.python))}>
        {browser === undefined ? "…" : agreement(browser, item.python) ? "Agrees" : "Disagrees"}
      </td>
    </tr>
  );
}

/** Every committed case, decided again here beside the recorded Python verdict. */
export function CaseTable({
  cases,
  whitespace,
}: {
  readonly cases: readonly AttestationCase[];
  readonly whitespace: ReadonlySet<number>;
}) {
  const [verdicts, setVerdicts] = useState<readonly AttestationVerdict[] | null>(null);
  useEffect(() => {
    let live = true;
    void Promise.all(
      cases.map((item) => verifyEnvelope(item.envelope, item.requireSignature, whitespace)),
    ).then((decided) => {
      if (live) setVerdicts(decided);
    });
    return () => {
      live = false;
    };
  }, [cases, whitespace]);
  return (
    <table aria-label="Attestation cases">
      <thead>
        <tr>
          <th scope="col">Case</th>
          <th scope="col">Change after sealing</th>
          <th scope="col">Browser verdict</th>
          <th scope="col">Python verdict</th>
          <th scope="col">Recomputed digest</th>
          <th scope="col">Agreement</th>
        </tr>
      </thead>
      <tbody>
        {cases.map((item, at) => (
          <CaseRow key={item.id} item={item} browser={verdicts?.[at]} />
        ))}
      </tbody>
    </table>
  );
}

async function oracleAgreement(rows: readonly OracleRow[]): Promise<readonly string[]> {
  const disagreeing: string[] = [];
  for (const row of rows) {
    const textHere = pythonCanonicalText({ claim_axes: {}, content: row.content });
    if (textHere !== row.pythonText || (await sha256Hex(textHere)) !== row.digest)
      disagreeing.push(row.label);
  }
  return disagreeing;
}

/** The canonical text written here, compared with the owner's text and digest. */
export function OraclePanel({ rows }: { readonly rows: readonly OracleRow[] }) {
  const [disagreeing, setDisagreeing] = useState<readonly string[] | null>(null);
  useEffect(() => {
    let live = true;
    void oracleAgreement(rows).then((labels) => {
      if (live) setDisagreeing(labels);
    });
    return () => {
      live = false;
    };
  }, [rows]);
  const status =
    disagreeing === null
      ? "Checking the canonical form…"
      : disagreeing.length === 0
        ? `The canonical text written here equals the Python owner's text and digest for all ${rows.length} samples.`
        : `The canonical text differs from the Python owner for: ${disagreeing.join(", ")}.`;
  return (
    <section aria-label="Canonical form">
      <h5>Canonical form</h5>
      <p
        role="status"
        data-agrees={disagreeing === null ? "pending" : String(disagreeing.length === 0)}
      >
        {status}
      </p>
      <p>Samples: {rows.map((row) => row.label).join(", ")}.</p>
    </section>
  );
}

function CheckOutcome({ outcome }: { readonly outcome: EnvelopeCheck }) {
  if (!outcome.ok) return <p role="alert">Refused before a verdict: {outcome.reason}</p>;
  return (
    <p role="status">
      <StatusChip verdict={outcome.verdict} /> {outcome.verdict.reason}. Recomputed digest{" "}
      {short(outcome.verdict.expectedDigest)}, carried {short(outcome.verdict.observedDigest)}.
    </p>
  );
}

/** Verify a local envelope file in this browser; nothing is uploaded. */
export function EnvelopeFileCheck({ whitespace }: { readonly whitespace: ReadonlySet<number> }) {
  const input = useId();
  const [requireSignature, setRequireSignature] = useState(false);
  const [outcome, setOutcome] = useState<EnvelopeCheck | null>(null);
  const check = async (file: File | undefined) => {
    if (file === undefined) return;
    if (file.size > maxEnvelopeBytes)
      return setOutcome({ ok: false, reason: `the file exceeds ${maxEnvelopeBytes} bytes` });
    setOutcome(await checkEnvelopeText(await file.text(), requireSignature, whitespace));
  };
  return (
    <section aria-label="Check an envelope file">
      <h5>Check an envelope file</h5>
      <label htmlFor={input}>Envelope file (attested_result_pack.v1 JSON, read only here)</label>
      <input
        id={input}
        type="file"
        accept="application/json,.json"
        onChange={(event) => void check(event.currentTarget.files?.[0])}
      />
      <label>
        <input
          type="checkbox"
          checked={requireSignature}
          onChange={(event) => setRequireSignature(event.currentTarget.checked)}
        />
        Require a signature
      </label>
      {outcome === null ? null : <CheckOutcome outcome={outcome} />}
    </section>
  );
}
