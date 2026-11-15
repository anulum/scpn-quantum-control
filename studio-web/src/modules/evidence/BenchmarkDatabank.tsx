// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — canonical tier benchmarks of the CI and local environments

import { useEffect, useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import {
  benchmarkEnvironments,
  provenanceFields,
  verifyCommittedDatabank,
} from "./benchmarkDatabank";
import type {
  BackendTiming,
  BenchmarkDatabank,
  EnvironmentRecord,
  OperationResult,
} from "./benchmarkDatabank";

const environmentLabel = { ci: "CI", local: "Local" } as const;

function ProvenanceTable({ databank }: { readonly databank: BenchmarkDatabank }) {
  return (
    <table aria-label="Benchmark provenance">
      <thead>
        <tr>
          <th scope="col">Field</th>
          {benchmarkEnvironments.map((name) => (
            <th scope="col" key={name}>
              {environmentLabel[name]}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {["generated", ...provenanceFields].map((field) => (
          <tr key={field}>
            <th scope="row">{field}</th>
            {benchmarkEnvironments.map((name) => {
              const record = databank.environments[name];
              return (
                <td key={name}>
                  {field === "generated" ? record.generatedUtc : record.provenance[field]}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function cell(row: BackendTiming | undefined): string {
  if (row === undefined) return "not run";
  return row.p50 === null ? "unavailable" : row.p50.toPrecision(3);
}

function sizes(record: EnvironmentRecord, operation: string): readonly OperationResult[] {
  return record.results
    .filter((item) => item.operation === operation)
    .sort((a, b) => a.size - b.size);
}

function ResultHeader({ backends }: { readonly backends: readonly string[] }) {
  return (
    <thead>
      <tr>
        <th scope="col">Size</th>
        {benchmarkEnvironments.flatMap((name) => [
          ...backends.map((backend) => (
            <th scope="col" key={`${name}:${backend}`}>
              {`${environmentLabel[name]} ${backend}`}
            </th>
          )),
          <th scope="col" key={`${name}:fastest`}>{`${environmentLabel[name]} fastest`}</th>,
        ])}
        <th scope="col">CI parity</th>
      </tr>
    </thead>
  );
}

function ResultRow({
  backends,
  measured,
}: {
  readonly backends: readonly string[];
  readonly measured: { readonly ci: OperationResult; readonly local: OperationResult | undefined };
}) {
  return (
    <tr data-size={measured.ci.size}>
      <th scope="row">{measured.ci.size}</th>
      {benchmarkEnvironments.flatMap((name) => [
        ...backends.map((backend) => (
          <td key={`${name}:${backend}`}>
            {cell(measured[name]?.rows.find((row) => row.backend === backend))}
          </td>
        )),
        <td key={`${name}:fastest`} data-fastest={measured[name]?.fastest ?? "none"}>
          {measured[name]?.fastest ?? "none"}
        </td>,
      ])}
      <td>{measured.ci.parity === null ? "not compared" : measured.ci.parity.toExponential(2)}</td>
    </tr>
  );
}

function ResultTable({
  databank,
  operation,
}: {
  readonly databank: BenchmarkDatabank;
  readonly operation: string;
}) {
  const { ci, local } = databank.environments;
  const backends = [...new Set(ci.results.flatMap((item) => item.rows.map((row) => row.backend)))];
  const localBySize = new Map(sizes(local, operation).map((item) => [item.size, item]));
  return (
    <table aria-label="Median latency by size">
      <caption>{`${operation}: median latency per call in microseconds (P50 of ${ci.repeats} repeats)`}</caption>
      <ResultHeader backends={backends} />
      <tbody>
        {sizes(ci, operation).map((item) => (
          <ResultRow
            key={item.size}
            backends={backends}
            measured={{ ci: item, local: localBySize.get(item.size) }}
          />
        ))}
      </tbody>
    </table>
  );
}

function UnavailableNote({ databank }: { readonly databank: BenchmarkDatabank }) {
  const reasons = benchmarkEnvironments.flatMap((name) => {
    const rows = databank.environments[name].results.flatMap((item) => item.rows);
    const found = new Set(
      rows.flatMap((row) => (row.reason === null ? [] : [`${row.backend}: ${row.reason}`])),
    );
    return [...found].map((reason) => `${environmentLabel[name]} ${reason}`);
  });
  if (reasons.length === 0) return null;
  return (
    <ul aria-label="Unavailable measurements">
      {reasons.map((reason) => (
        <li key={reason}>{reason}</li>
      ))}
    </ul>
  );
}

function DatabankView({ databank }: { readonly databank: BenchmarkDatabank }) {
  const select = useId();
  const initial = databank.operations.includes("order_parameter")
    ? "order_parameter"
    : databank.operations[0];
  const [operation, setOperation] = useState(initial as string);
  return (
    <article className="qsp-panel" aria-label="Benchmark databank">
      <h4>Benchmark databank</h4>
      <p className="qsp-banner">{databank.boundary}.</p>
      <p role="status">
        The file and payload digests of both artefacts were verified in this browser against the
        committed pin.
      </p>
      <ProvenanceTable databank={databank} />
      <label htmlFor={select}>Operation</label>
      <select
        id={select}
        value={operation}
        onChange={(event) => setOperation(event.currentTarget.value)}
      >
        {databank.operations.map((name) => (
          <option key={name}>{name}</option>
        ))}
      </select>
      <ResultTable databank={databank} operation={operation} />
      <UnavailableNote databank={databank} />
    </article>
  );
}

/**
 * The canonical tier benchmarks, verified here before they are shown.
 *
 * @param props.load - Verifier of the committed pin and artefacts; tests pass their own.
 */
export function BenchmarkDatabankInspector({
  load = verifyCommittedDatabank,
}: {
  readonly load?: () => Promise<Loaded<BenchmarkDatabank>>;
}) {
  const [loaded, setLoaded] = useState<Loaded<BenchmarkDatabank> | null>(null);
  useEffect(() => {
    let live = true;
    void load().then((value) => {
      if (live) setLoaded(value);
    });
    return () => {
      live = false;
    };
  }, [load]);
  if (loaded === null)
    return (
      <article className="qsp-panel" aria-label="Benchmark databank">
        <h4>Benchmark databank</h4>
        <p role="status">Verifying the benchmark artefacts…</p>
      </article>
    );
  if (!loaded.ok)
    return (
      <Unverifiable
        surface="docs/benchmarks/tiers/kuramoto_tiers.{ci,local}.json"
        reason={loaded.reason}
      />
    );
  return <DatabankView databank={loaded.value} />;
}
