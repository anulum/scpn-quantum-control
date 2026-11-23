// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — dated calibration snapshots, drift, policies and plan verdicts

import { useId, useState } from "react";
import recordText from "../../../../../data/studio/calibration_inspection_20261123.json?raw";
import {
  type CalibrationInspection,
  type CalibrationInspectionAdmission,
  type InspectedSnapshot,
  admitCalibrationInspection,
  referenceStatus,
} from "./calibrationInspection";

function value(reading: number | null): string {
  return reading === null ? "unknown" : String(reading);
}

function Policies({ record }: { readonly record: CalibrationInspection }) {
  return (
    <table aria-label="Freshness policies">
      <thead>
        <tr>
          <th scope="col">Experiment type</th>
          <th scope="col">Declared in</th>
          <th scope="col">Rules</th>
        </tr>
      </thead>
      <tbody>
        {record.policies.map((policy) => (
          <tr key={policy.experimentType}>
            <th scope="row">{policy.experimentType}</th>
            <td>{policy.citation}</td>
            <td>{policy.rules.join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Readings({ snapshot }: { readonly snapshot: InspectedSnapshot }) {
  const rows = [
    ...snapshot.qubits.map((item) => ({ ...item, quantity: "readout error" })),
    ...snapshot.couplers.map((item) => ({ ...item, quantity: "CZ fidelity" })),
  ];
  return (
    <div className="qsp-data-scroll">
      <table aria-label="Element readings">
        <thead>
          <tr>
            <th scope="col">Element</th>
            <th scope="col">Quantity</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.element}>
              <th scope="row">{row.element}</th>
              <td>{row.quantity}</td>
              <td>{value(row.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Drift({
  record,
  snapshot,
}: {
  readonly record: CalibrationInspection;
  readonly snapshot: InspectedSnapshot;
}) {
  const drift = record.drifts.find((item) => item.after === snapshot.calibrationSetId);
  if (drift === undefined) return <p>No earlier snapshot of this target is recorded.</p>;
  return (
    <div className="qsp-data-scroll">
      <p>{`Changes from ${drift.before}: ${drift.changes.length} element(s) differ, ${drift.unchanged} read the same${drift.calibrationSetChanged ? "; the calibration set changed" : ""}.`}</p>
      <table aria-label="Changes from the previous snapshot">
        <thead>
          <tr>
            <th scope="col">Element</th>
            <th scope="col">Change</th>
            <th scope="col">Before</th>
            <th scope="col">After</th>
          </tr>
        </thead>
        <tbody>
          {drift.changes.map((change) => (
            <tr key={`${change.element} ${change.quantity}`}>
              <th scope="row">{change.element}</th>
              <td>{change.change}</td>
              <td>{value(change.before)}</td>
              <td>{value(change.after)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Plans({ record }: { readonly record: CalibrationInspection }) {
  return (
    <div className="qsp-data-scroll">
      <table aria-label="Plans and their calibration">
        <thead>
          <tr>
            <th scope="col">Plan</th>
            <th scope="col">Block</th>
            <th scope="col">Calibration set</th>
            <th scope="col">Verdict</th>
            <th scope="col">Why</th>
          </tr>
        </thead>
        <tbody>
          {record.plans.map((plan) => (
            <tr key={plan.planId} data-status={plan.status}>
              <th scope="row">{plan.planId}</th>
              <td>{plan.blockOn}</td>
              <td>{plan.calibrationSetId ?? "none named"}</td>
              <td>{plan.status}</td>
              <td>{plan.reasons.join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReferenceCheck({ record }: { readonly record: CalibrationInspection }) {
  const [digest, setDigest] = useState("");
  const inputId = useId();
  const status = digest === "" ? null : referenceStatus(record, digest.trim());
  const verdict =
    status === null
      ? "Enter the calibration file sha256 a dossier or profile binds."
      : status.kind === "unknown"
        ? "Not an admitted snapshot: dossiers bound to it are blocked."
        : status.kind === "superseded"
          ? `Superseded by ${status.snapshot.supersededBy}: dossiers bound to ${status.snapshot.calibrationSetId} are blocked, and no other snapshot is substituted.`
          : `Current snapshot of ${status.snapshot.targetId}, ${status.snapshot.observedOn}.`;
  return (
    <section aria-label="Calibration reference check">
      <label htmlFor={inputId}>Calibration file sha256</label>
      <input id={inputId} value={digest} onChange={(event) => setDigest(event.target.value)} />
      <p role="status" aria-label="Calibration reference verdict">
        {verdict}
      </p>
    </section>
  );
}

function Inspection({ record }: { readonly record: CalibrationInspection }) {
  const [index, setIndex] = useState(record.snapshots.length - 1);
  const selectId = useId();
  const snapshot = record.snapshots[index];
  return (
    <>
      <p>{`Record ${record.recordDigest.slice(0, 12)} verified: ${record.snapshots.length} snapshots, ${record.plans.length} plans.`}</p>
      <Policies record={record} />
      <label htmlFor={selectId}>Calibration snapshot</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {record.snapshots.map((item, at) => (
          <option key={item.calibrationSetId} value={at}>
            {`${item.observedOn} · ${item.calibrationSetId}`}
          </option>
        ))}
      </select>
      {snapshot === undefined ? (
        <p>No snapshot is recorded.</p>
      ) : (
        <section aria-label="Snapshot provenance">
          <p>{`${snapshot.source} (${snapshot.targetId}), ${snapshot.observedOn}, ${snapshot.path} sha256 ${snapshot.sha256}. ${snapshot.fidelitySemantics}.`}</p>
          <p role="note" aria-label="Snapshot standing">
            {snapshot.supersededBy === null
              ? "Current calibration of this target."
              : `Superseded by ${snapshot.supersededBy}.`}
          </p>
          <p>{`Referenced by: ${snapshot.referencedBy.length === 0 ? "no plan" : snapshot.referencedBy.join(", ")}.`}</p>
          <Readings snapshot={snapshot} />
          <Drift record={record} snapshot={snapshot} />
        </section>
      )}
      <Plans record={record} />
      <ReferenceCheck record={record} />
    </>
  );
}

/**
 * Show dated calibration snapshots, their exact drift, the declared freshness policies and the plan verdicts.
 *
 * Every verdict comes from the sealed source record. A missing reading is shown as unknown, a blocked
 * plan keeps the set it named, and nothing selects another snapshot or device.
 *
 * @param props - How to admit the record; by default the record committed with this build.
 * @returns The panel.
 */
export function CalibrationInspector({
  admit = () => admitCalibrationInspection(recordText),
}: {
  readonly admit?: () => Promise<CalibrationInspectionAdmission>;
}) {
  const [admission, setAdmission] = useState<CalibrationInspectionAdmission | null>(null);
  const [pending, setPending] = useState(false);
  return (
    <section className="qsp-workspace" aria-label="Calibration inspection">
      <h4>Calibration snapshots</h4>
      <button
        type="button"
        disabled={pending}
        onClick={() => {
          setPending(true);
          void admit().then((result) => {
            setAdmission(result);
            setPending(false);
          });
        }}
      >
        Open calibration inspection
      </button>
      {admission === null ? null : admission.ok ? (
        <Inspection record={admission.record} />
      ) : (
        <p role="alert">{admission.reason}</p>
      )}
    </section>
  );
}
