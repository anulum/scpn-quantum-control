// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the gradient debugger shows source values and refuses before execution

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import committedUnitsJson from "../../../../data/studio/gradient_debugger_units_20261030.json";
import { gradientPlanExplanations } from "../../shared/browser/data";
import type { GradientPlanExplanationView, Loaded } from "../../shared/browser/data";
import { instantiateProgramAd } from "../../shared/browser/programAd";
import type { KernelReplay } from "../../shared/browser/programAd";
import { GradientDebugger } from "./GradientDebugger";
import {
  debuggerChoices,
  gradientUnits,
  localRunnerRequest,
  maskedGradient,
  parseGradientUnits,
  withinBudget,
} from "./gradientDebugger";
import type { GradientUnits } from "./gradientDebugger";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_program_ad_wasm/target/wasm32-unknown-unknown/release/scpn_quantum_studio_program_ad_wasm.wasm",
);
const RY_GRAD = "ry::pauli_expectation::statevector::grad::native";
const HESSIAN = "ry::pauli_expectation::qasm_simulator::hessian::native";
const JAX = "ry::pauli_expectation::statevector::value_and_grad::jax";
const NO_UNIT = "rz::kuramoto_xy_energy::qasm_simulator::grad::native";

let kernel: KernelReplay;

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  kernel = await instantiateProgramAd(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer,
  );
});
afterEach(cleanup);

function units(): GradientUnits {
  if (!gradientUnits.ok) throw new Error(gradientUnits.reason);
  return gradientUnits.value;
}

function plan(): GradientPlanExplanationView {
  if (!gradientPlanExplanations.ok) throw new Error(gradientPlanExplanations.reason);
  return gradientPlanExplanations.value;
}

/** Render the debugger with a counting loader around the real built kernel. */
function renderDebugger(loaded: Loaded<GradientUnits> = gradientUnits) {
  const loadKernel = vi.fn(async () => kernel);
  render(
    <GradientDebugger plan={gradientPlanExplanations} units={loaded} loadKernel={loadKernel} />,
  );
  return loadKernel;
}

function choose(id: string): void {
  fireEvent.change(screen.getByLabelText("Objective, transform, backend and framework"), {
    target: { value: id },
  });
}

function recomputeStatus(): string {
  return screen.getByRole("status", { name: "Browser recompute" }).textContent ?? "";
}

it("test_gradient_debugger_01: the supported RY/Z example recomputes -sin(theta) in the WASM kernel", async () => {
  const loadKernel = renderDebugger();
  expect(
    (screen.getByLabelText("Objective, transform, backend and framework") as HTMLSelectElement)
      .value,
  ).toBe(RY_GRAD);
  const unit = screen.getByRole("region", { name: "Unit RY(θ) then measure Z" });
  expect(within(unit).getByText("⟨Z⟩ = cos θ")).toBeTruthy();
  const committed = units().units[0];
  expect(withinBudget(committed?.expected.gradient ?? [], [-Math.sin(0.4)])).toBe(true);
  expect(withinBudget(committed?.reference?.gradient ?? [], [-Math.sin(0.4)])).toBe(true);
  fireEvent.click(within(unit).getByRole("button", { name: "Recompute in browser" }));
  await waitFor(() =>
    expect(recomputeStatus()).toBe(
      "The browser kernel agrees with the source within the float64 budget.",
    ),
  );
  expect(loadKernel).toHaveBeenCalledTimes(1);
  const row = within(screen.getByRole("table", { name: "Gradient comparison" }))
    .getByText("browser WASM recompute")
    .closest("tr") as HTMLTableRowElement;
  const [value, gradient] = [...row.querySelectorAll("td")].map((cell) => Number(cell.textContent));
  expect(withinBudget([value as number, gradient as number], [Math.cos(0.4), -Math.sin(0.4)])).toBe(
    true,
  );
  const fingerprints = screen.getByRole("list", { name: "Framework fingerprints" }).textContent;
  expect(fingerprints).toContain("qiskit 2.4.1");
  expect(fingerprints).toContain("pennylane 0.45.0");
  expect(screen.getByRole("list", { name: "Forward pass" }).textContent).toContain("cos(theta)");
  expect(screen.getByRole("list", { name: "Reverse pass" }).textContent).toContain("∂f/∂theta");
});

it("refuses a replay input that no longer hashes to its committed digest, before the kernel loads", async () => {
  const tampered = structuredClone(committedUnitsJson) as { units: { input_sha256: string }[] };
  (tampered.units[0] as { input_sha256: string }).input_sha256 = `sha256:${"0".repeat(64)}`;
  const loadKernel = renderDebugger(parseGradientUnits(tampered));
  fireEvent.click(screen.getByRole("button", { name: "Recompute in browser" }));
  await waitFor(() => expect(recomputeStatus()).toContain("does not match its committed SHA-256"));
  expect(loadKernel).not.toHaveBeenCalled();
});

it("test_gradient_debugger_02: an unsupported higher order is refused before anything executes", () => {
  const loadKernel = renderDebugger();
  choose(HESSIAN);
  const refusal = screen.getByRole("region", { name: "Refused before execution" });
  expect(refusal.textContent).toContain(
    "hessian support is limited to deterministic local backends",
  );
  expect(screen.queryByRole("button", { name: "Recompute in browser" })).toBeNull();
  expect(screen.queryByRole("region", { name: "Local runner request" })).toBeNull();
  expect(loadKernel).not.toHaveBeenCalled();
});

it("test_gradient_debugger_03: the trainable mask changes the active inputs and the source-stated gradient", () => {
  renderDebugger();
  choose("two-angle-mix");
  const active = () => screen.getByRole("status", { name: "Active inputs" }).textContent;
  const masked = (name: string) =>
    (screen.getByRole("rowheader", { name }).closest("tr") as HTMLTableRowElement).lastElementChild
      ?.textContent;
  expect(active()).toBe("Active inputs: a, b");
  expect(Number(masked("b"))).toBeCloseTo(0.25 * Math.cos(-1.1), 12);
  fireEvent.click(screen.getByLabelText("b trainable"));
  expect(active()).toBe("Active inputs: a");
  expect(Number(masked("b"))).toBe(0);
  expect(Number(masked("a"))).toBeCloseTo(-Math.sin(0.3), 12);
  fireEvent.click(screen.getByLabelText("a trainable"));
  expect(active()).toBe("Active inputs: none");
  expect(Number(masked("a"))).toBe(0);
});

it("test_gradient_debugger_04: JAX is unavailable in the browser and gets a local-runner request, never an emulated value", () => {
  const loadKernel = renderDebugger();
  choose(JAX);
  expect(screen.getByRole("note").textContent).toBe(
    "No jax runtime runs in this browser; it is not emulated.",
  );
  const request = screen.getByRole("region", { name: "Local runner request" });
  expect(request.textContent).toContain(
    "scpn-studio-run differentiate --action-id gradient-debugger-ry-z-expectation --params-file gradient-debugger-ry-z-expectation.json",
  );
  expect(request.textContent).toContain('"op": "cos"');
  expect(request.textContent).toContain("it does not run another framework");
  expect(screen.queryByRole("table", { name: "Gradient comparison" })).toBeNull();
  expect(loadKernel).not.toHaveBeenCalled();
});

it("says when the source states no gradient for a mask", () => {
  const unstated = structuredClone(committedUnitsJson) as { units: { masks: unknown[] }[] };
  (unstated.units[0] as { masks: unknown[] }).masks = [];
  renderDebugger(parseGradientUnits(unstated));
  const row = screen.getByRole("rowheader", { name: "theta" }).closest("tr") as HTMLTableRowElement;
  expect(row.lastElementChild?.textContent).toBe("not stated by the source");
});

it("prepares a generic local-runner request for a supported cell without a committed unit", () => {
  renderDebugger();
  choose(NO_UNIT);
  expect(screen.getByRole("note").textContent).toBe(
    "No committed unit replays this cell in the browser.",
  );
  expect(screen.getByRole("region", { name: "Local runner request" }).textContent).toContain(
    "--params-file <program.json> --preview",
  );
});

describe("kernel outcomes", () => {
  it("reports a kernel that fails to load", async () => {
    render(
      <GradientDebugger
        plan={gradientPlanExplanations}
        units={gradientUnits}
        loadKernel={async () => {
          throw new Error("no network");
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Recompute in browser" }));
    await waitFor(() => expect(recomputeStatus()).toBe("WASM kernel unavailable: no network"));
  });

  it("reports a kernel refusal and a disagreement", async () => {
    render(
      <GradientDebugger
        plan={gradientPlanExplanations}
        units={gradientUnits}
        loadKernel={async () => () => ({ ok: false, code: -5 })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Recompute in browser" }));
    await waitFor(() =>
      expect(recomputeStatus()).toBe("WASM kernel refused the program (code -5)."),
    );
    cleanup();
    render(
      <GradientDebugger
        plan={gradientPlanExplanations}
        units={gradientUnits}
        loadKernel={async () => () => ({ ok: true, value: 0.5, gradient: [0.5] })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Recompute in browser" }));
    await waitFor(() =>
      expect(recomputeStatus()).toBe(
        "The browser kernel disagrees with the source beyond the float64 budget.",
      ),
    );
  });

  it("refuses committed replay input that is not hex", async () => {
    const broken = structuredClone(committedUnitsJson) as { units: { input_hex: string }[] };
    (broken.units[0] as { input_hex: string }).input_hex = "zz";
    renderDebugger(parseGradientUnits(broken));
    fireEvent.click(screen.getByRole("button", { name: "Recompute in browser" }));
    await waitFor(() => expect(recomputeStatus()).toBe("Committed replay input is not valid hex."));
  });
});

describe("model", () => {
  it("refuses malformed artefacts with the first reason", () => {
    const base = structuredClone(committedUnitsJson) as Record<string, unknown>;
    expect(parseGradientUnits([])).toEqual({ ok: false, reason: "artefact must be an object" });
    expect(parseGradientUnits({ ...base, schema: "other" })).toMatchObject({ ok: false });
    type Draft = { units: Record<string, unknown>[] };
    const first = (draft: Draft) => draft.units[0] as Record<string, unknown>;
    const cases: [string, (draft: Draft) => void][] = [
      [
        "units[0].title must be a non-empty string",
        (draft) => {
          first(draft)["title"] = "";
        },
      ],
      [
        "units[0].expected.value must be a finite number",
        (draft) => {
          (first(draft)["expected"] as Record<string, unknown>)["value"] = "x";
        },
      ],
      [
        "units[0].masks[0] flags must be true or false",
        (draft) => {
          (
            (first(draft)["masks"] as { trainable: unknown[] }[])[0] as { trainable: unknown[] }
          ).trainable[0] = "yes";
        },
      ],
      [
        "units[0].masks must be a list",
        (draft) => {
          first(draft)["masks"] = {};
        },
      ],
      [
        "units[0].parameters must be a list",
        (draft) => {
          first(draft)["parameters"] = null;
        },
      ],
      [
        "units[0].expected.gradient must be a list",
        (draft) => {
          (first(draft)["expected"] as Record<string, unknown>)["gradient"] = 1;
        },
      ],
    ];
    for (const [reason, mutate] of cases) {
      const draft = structuredClone(committedUnitsJson) as { units: Record<string, unknown>[] };
      mutate(draft);
      expect(parseGradientUnits(draft)).toEqual({ ok: false, reason });
    }
  });

  it("shows unverifiable sources instead of a debugger", () => {
    render(<GradientDebugger plan={{ ok: false, reason: "plan refused" }} units={gradientUnits} />);
    expect(screen.getByText(/plan refused/)).toBeTruthy();
    cleanup();
    render(
      <GradientDebugger
        plan={gradientPlanExplanations}
        units={{ ok: false, reason: "units refused" }}
      />,
    );
    expect(screen.getByText(/units refused/)).toBeTruthy();
  });

  it("opens on the first declared choice when none runs in the browser, and says when there is none", () => {
    const blocked = plan().rows.filter((row) => !row.supported);
    const none = { ...units(), units: [] };
    render(
      <GradientDebugger
        plan={{ ok: true, value: { ...plan(), rows: blocked } }}
        units={{ ok: true, value: none }}
      />,
    );
    expect(screen.getByRole("region", { name: "Refused before execution" })).toBeTruthy();
    cleanup();
    render(
      <GradientDebugger
        plan={{ ok: true, value: { ...plan(), rows: [] } }}
        units={{ ok: true, value: none }}
      />,
    );
    expect(screen.getByRole("status").textContent).toBe(
      "No declared gradient combination or committed unit to debug.",
    );
  });

  it("states masks, budgets and requests only from the source", () => {
    const ry = units().units[0];
    if (ry === undefined) throw new Error("missing unit");
    expect(maskedGradient(ry, [true, true])).toBeNull();
    expect(maskedGradient(ry, [false])).toEqual([0]);
    expect(withinBudget([1], [1, 2])).toBe(false);
    expect(localRunnerRequest(null).paramsJson).toBeNull();
    const choices = debuggerChoices(plan().rows, units().units);
    expect(choices.map((choice) => choice.route.kind)).toEqual([
      "browser",
      "unavailable",
      "unavailable",
      "unavailable",
      "unavailable",
      "refused",
      "refused",
      "refused",
      "refused",
      "refused",
      "browser",
    ]);
  });
});
