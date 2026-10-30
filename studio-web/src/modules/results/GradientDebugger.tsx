// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — gradient debugger over declared plans and committed units

import { useId, useMemo, useState } from "react";
import type { GradientPlanExplanationView, Loaded } from "../../shared/browser/data";
import { digestProgramAdInput, fetchProgramAd, hexToBytes } from "../../shared/browser/programAd";
import type { KernelReplay } from "../../shared/browser/programAd";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import {
  debuggerChoices,
  localRunnerRequest,
  maskedGradient,
  withinBudget,
} from "./gradientDebugger";
import type { DebuggerChoice, GradientUnit, GradientUnits } from "./gradientDebugger";

/** State of one browser recomputation. */
type Recompute =
  | { readonly phase: "idle" }
  | { readonly phase: "running" }
  | {
      readonly phase: "done";
      readonly value: number;
      readonly gradient: readonly number[];
      readonly agrees: boolean;
    }
  | { readonly phase: "refused"; readonly reason: string };

const format = (value: number) => value.toPrecision(16);

/**
 * Run one committed unit through the shipped WASM kernel.
 *
 * The replay input must hash to the committed digest before the kernel loads;
 * the result is compared with the source within the frozen float64 budget.
 */
async function recompute(
  unit: GradientUnit,
  loadKernel: () => Promise<KernelReplay>,
): Promise<Recompute> {
  const bytes = hexToBytes(unit.inputHex);
  if (bytes === null)
    return { phase: "refused", reason: "Committed replay input is not valid hex." };
  if ((await digestProgramAdInput(bytes)) !== unit.inputSha256)
    return {
      phase: "refused",
      reason: "Replay input does not match its committed SHA-256; nothing ran.",
    };
  let replay: KernelReplay;
  try {
    replay = await loadKernel();
  } catch (cause: unknown) {
    return { phase: "refused", reason: `WASM kernel unavailable: ${(cause as Error).message}` };
  }
  const result = replay(bytes, unit.parameters.length);
  if (!result.ok)
    return { phase: "refused", reason: `WASM kernel refused the program (code ${result.code}).` };
  return {
    phase: "done",
    value: result.value,
    gradient: result.gradient,
    agrees: withinBudget(
      [result.value, ...result.gradient],
      [unit.expected.value, ...unit.expected.gradient],
    ),
  };
}

function PlanFacts({ choice }: { readonly choice: DebuggerChoice }) {
  const cell = choice.cell;
  if (cell === null)
    return (
      <p>A committed program unit outside the plan table; it runs only in the browser kernel.</p>
    );
  return (
    <dl className="qsp-gradient-facts" aria-label="Declared plan">
      <dt>Method</dt>
      <dd>
        {cell.selectedMethod} ({cell.methodFamily}, {cell.evaluationMode})
      </dd>
      <dt>Framework and backend</dt>
      <dd>
        {cell.framework} · {cell.backend} ({cell.backendFamily})
      </dd>
      <dt>Cost</dt>
      <dd>
        {cell.backendEvaluations} backend evaluations per gradient
        {cell.shots === null ? "" : ` · ${cell.shots} shots`}
      </dd>
      <dt>Finite-shot variance</dt>
      <dd>
        {cell.requiresFiniteShotVariance ? "Required by this method's error bound" : "Not involved"}
      </dd>
      {cell.warnings.length > 0 && (
        <>
          <dt>Warnings</dt>
          <dd>{cell.warnings.join(" ")}</dd>
        </>
      )}
    </dl>
  );
}

function RunnerRequest({ unit }: { readonly unit: GradientUnit | null }) {
  const request = localRunnerRequest(unit);
  return (
    <section aria-label="Local runner request">
      <h5>Run it locally</h5>
      <p>
        Nothing is submitted from the browser. The local runner differentiates the same objective
        with the package's own backends; it does not run another framework. Run this with the
        installed package:
      </p>
      <pre className="qsp-script">
        <code>{request.command}</code>
      </pre>
      {request.paramsJson !== null && (
        <>
          <p>Save the parameters beside it as {`${unit?.localRunner.actionId}.json`}:</p>
          <pre className="qsp-script">
            <code>{request.paramsJson}</code>
          </pre>
        </>
      )}
    </section>
  );
}

function Flow({ unit }: { readonly unit: GradientUnit }) {
  const name = (target: string) =>
    unit.parameters.find((parameter) => parameter.target === target)?.name ?? target;
  return (
    <section className="qsp-gradient-flow" aria-label="Forward and reverse flow">
      <h5>Flow</h5>
      <ol aria-label="Forward pass">
        {unit.steps.map((step) => (
          <li key={step.target}>
            <code>{step.target}</code> ={" "}
            {step.operation === "parameter"
              ? `parameter ${step.inputs.join(", ")}`
              : `${step.operation}(${step.inputs.map(name).join(", ")})`}
          </li>
        ))}
      </ol>
      <ol aria-label="Reverse pass">
        {[...unit.steps].reverse().map((step) => (
          <li key={step.target}>
            adjoint of <code>{step.target}</code>
            {step.operation === "parameter"
              ? ` gives ∂f/∂${step.inputs.join(", ")}`
              : ` flows into ${step.inputs.join(", ")}`}
          </li>
        ))}
      </ol>
    </section>
  );
}

function UnitDebugger({
  unit,
  loadKernel,
}: {
  readonly unit: GradientUnit;
  readonly loadKernel: () => Promise<KernelReplay>;
}) {
  const [trainable, setTrainable] = useState<boolean[]>(() => unit.parameters.map(() => true));
  const [run, setRun] = useState<Recompute>({ phase: "idle" });
  const masked = maskedGradient(unit, trainable);
  const active = unit.parameters.filter((_parameter, index) => trainable[index]);
  return (
    <section aria-label={`Unit ${unit.title}`}>
      <h5>{unit.objective}</h5>
      <table aria-label="Parameters and trainable mask">
        <thead>
          <tr>
            <th scope="col">Parameter</th>
            <th scope="col">Value</th>
            <th scope="col">Trainable</th>
            <th scope="col">Gradient under mask</th>
          </tr>
        </thead>
        <tbody>
          {unit.parameters.map((parameter, index) => (
            <tr key={parameter.target}>
              <th scope="row">{parameter.name}</th>
              <td className="qsp-mono">{format(parameter.value)}</td>
              <td>
                <input
                  type="checkbox"
                  aria-label={`${parameter.name} trainable`}
                  checked={trainable[index] === true}
                  onChange={(event) =>
                    setTrainable((previous) =>
                      previous.map((flag, at) => (at === index ? event.target.checked : flag)),
                    )
                  }
                />
              </td>
              <td className="qsp-mono">
                {masked === null ? "not stated by the source" : format(masked[index] as number)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p role="status" aria-label="Active inputs">
        Active inputs:{" "}
        {active.length === 0 ? "none" : active.map((parameter) => parameter.name).join(", ")}
      </p>
      <Flow unit={unit} />
      <table aria-label="Gradient comparison">
        <thead>
          <tr>
            <th scope="col">Source</th>
            <th scope="col">Value</th>
            <th scope="col">Gradient</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">differentiate action (reverse mode)</th>
            <td className="qsp-mono">{format(unit.expected.value)}</td>
            <td className="qsp-mono">{unit.expected.gradient.map(format).join(", ")}</td>
          </tr>
          <tr>
            <th scope="row">
              central finite differences (step {unit.finiteDifference.step}, max error{" "}
              {unit.finiteDifference.maxAbsError.toExponential(2)} ≤{" "}
              {unit.finiteDifference.tolerance})
            </th>
            <td>—</td>
            <td className="qsp-mono">{unit.finiteDifference.gradient.map(format).join(", ")}</td>
          </tr>
          {unit.reference !== null && (
            <tr>
              <th scope="row">owner reference ({unit.reference.caseId})</th>
              <td className="qsp-mono">{format(unit.reference.value)}</td>
              <td className="qsp-mono">{unit.reference.gradient.map(format).join(", ")}</td>
            </tr>
          )}
          {run.phase === "done" && (
            <tr>
              <th scope="row">browser WASM recompute</th>
              <td className="qsp-mono">{format(run.value)}</td>
              <td className="qsp-mono">{run.gradient.map(format).join(", ")}</td>
            </tr>
          )}
        </tbody>
      </table>
      {unit.reference !== null && (
        <ul aria-label="Framework fingerprints">
          {unit.reference.frameworks.map((framework) => (
            <li key={framework.backend}>
              {framework.backend}{" "}
              {Object.entries(framework.versions)
                .map(([name, version]) => `${name} ${version}`)
                .join(", ")}{" "}
              · {framework.evaluations} evaluations · circuit{" "}
              <code>{framework.circuitFingerprint.slice(0, 16)}</code>
            </li>
          ))}
        </ul>
      )}
      <button
        type="button"
        disabled={run.phase === "running"}
        onClick={() => {
          setRun({ phase: "running" });
          void recompute(unit, loadKernel).then(setRun);
        }}
      >
        Recompute in browser
      </button>
      <p role="status" aria-label="Browser recompute">
        {run.phase === "idle" && "Not recomputed yet."}
        {run.phase === "running" && "Recomputing in the WASM kernel…"}
        {run.phase === "done" &&
          (run.agrees
            ? "The browser kernel agrees with the source within the float64 budget."
            : "The browser kernel disagrees with the source beyond the float64 budget.")}
        {run.phase === "refused" && run.reason}
      </p>
    </section>
  );
}

/** Props of the gradient debugger. */
export interface GradientDebuggerProps {
  /** Declared plan cells. */
  readonly plan: Loaded<GradientPlanExplanationView>;
  /** Committed units the browser can replay. */
  readonly units: Loaded<GradientUnits>;
  /** How the WASM kernel is obtained; tests pass a built kernel. */
  readonly loadKernel?: () => Promise<KernelReplay>;
}

/**
 * Select a declared objective/parameter/transform/backend combination and see
 * its plan, cost, blockers, mask, flow and gradient checks.
 *
 * Selection is limited to declared plan cells and committed units. A blocked
 * cell is refused before anything loads; a framework the browser cannot run
 * is never emulated and gets a local-runner request instead.
 *
 * @param props - Plan, units and kernel loader.
 * @returns The debugger.
 */
export function GradientDebugger({
  plan,
  units,
  loadKernel = fetchProgramAd,
}: GradientDebuggerProps) {
  const selectId = useId();
  const choices = useMemo(
    () => (plan.ok && units.ok ? debuggerChoices(plan.value.rows, units.value.units) : []),
    [plan, units],
  );
  const [selected, setSelected] = useState(
    () => choices.find((choice) => choice.route.kind === "browser")?.id ?? "",
  );
  if (!plan.ok)
    return <Unverifiable surface="gradient_plan_explanations_20260709.json" reason={plan.reason} />;
  if (!units.ok)
    return <Unverifiable surface="gradient_debugger_units_20261030.json" reason={units.reason} />;
  const choice = choices.find((candidate) => candidate.id === selected) ?? choices[0];
  if (choice === undefined)
    return (
      <p role="status" className="qsp-empty">
        No declared gradient combination or committed unit to debug.
      </p>
    );
  return (
    <section className="qsp-gradient-debugger" aria-label="Gradient debugger">
      <h4>Gradient debugger</h4>
      <label htmlFor={selectId}>Objective, transform, backend and framework</label>
      <select id={selectId} value={choice.id} onChange={(event) => setSelected(event.target.value)}>
        <optgroup label="Declared plan cells">
          {choices
            .filter((candidate) => candidate.cell !== null)
            .map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.label}
              </option>
            ))}
        </optgroup>
        <optgroup label="Program units">
          {choices
            .filter((candidate) => candidate.cell === null)
            .map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.label}
              </option>
            ))}
        </optgroup>
      </select>
      <PlanFacts choice={choice} />
      {choice.route.kind === "refused" && (
        <section aria-label="Refused before execution">
          <h5>Refused before execution</h5>
          <ul>
            {choice.route.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </section>
      )}
      {choice.route.kind === "unavailable" && (
        <>
          <p role="note">{choice.route.reason}</p>
          <RunnerRequest unit={choice.route.unit} />
        </>
      )}
      {choice.route.kind === "browser" && (
        <UnitDebugger key={choice.id} unit={choice.route.unit} loadKernel={loadKernel} />
      )}
      <p className="qsp-boundary">{units.value.claimBoundary}</p>
    </section>
  );
}
