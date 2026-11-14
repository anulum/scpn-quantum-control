// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — relabelling, sectors, derivative validity and display-only layout

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import committed from "../../../../data/studio/topology_instrument_20261114.json";
import { TopologyInspector } from "./TopologyInspector";
import { parseTopologyInstrument, topologyInstrument } from "./topologyInstrument";
import { projectNodes } from "./TopologyProjection";

afterEach(cleanup);

type Raw = ReturnType<typeof copy>;
const copy = () => structuredClone(committed);
const caseNamed = (data: Raw, name: string) =>
  data.templates.sensitivity.cases.find(
    (item) => item.case === name,
  ) as Raw["templates"]["sensitivity"]["cases"][number];
const physics = () =>
  [
    "Order parameter under relabelling",
    "Symmetry sectors",
    "Cluster order parameters",
    "Persistence summaries",
  ].map((name) => screen.getAllByLabelText(name)[0]?.textContent);

it("test_topology_symmetry_instrument_01: relabelling the nodes keeps the order parameter", () => {
  render(<TopologyInspector instrument={topologyInstrument} />);
  const table = screen.getByRole("table", { name: "Order parameter by labelling" });
  const [original, relabelled] = within(table).getAllByRole("row").slice(1);
  expect(original?.textContent?.replace("Original", "")).toBe(
    `${committed.templates.relabelling.original.magnitude}${committed.templates.relabelling.original.phase}`,
  );
  expect(relabelled?.textContent).toContain(
    String(committed.templates.relabelling.relabelled.magnitude),
  );
  expect(screen.getByLabelText("Relabelling").textContent).toContain("0→2, 1→0, 2→3, 3→1");
  const changed = copy();
  changed.templates.relabelling.relabelled.magnitude += 0.01;
  changed.templates.relabelling.magnitude_difference = 0.01;
  expect(parseTopologyInstrument(changed)).toEqual({
    ok: false,
    reason: "A relabelling changed the order parameter",
  });
  const hidden = copy();
  hidden.templates.relabelling.relabelled.magnitude += 0.01;
  expect(parseTopologyInstrument(hidden).ok).toBe(false);
  const duplicated = copy();
  duplicated.templates.relabelling.permutation = [0, 0, 1, 2];
  expect(parseTopologyInstrument(duplicated)).toEqual({
    ok: false,
    reason: "The relabelling must be a permutation of the nodes",
  });
});

it("test_topology_symmetry_instrument_02: finite symmetric sectors match the dense reference", () => {
  render(<TopologyInspector instrument={topologyInstrument} />);
  const spectra = screen.getByLabelText("Sector spectra against the dense reference");
  expect(spectra.textContent).toContain("even 8, odd 8");
  expect(spectra.textContent).toContain("M=-4: 1, M=-2: 4, M=0: 6, M=2: 4, M=4: 1");
  const weights = screen.getByRole("table", {
    name: "Parity weights before and after the evolution",
  });
  expect(within(weights).getByText("mixed product state")).toBeTruthy();
  const drifted = copy();
  drifted.templates.sectors.spectra.parity_max_deviation = 1e-3;
  expect(parseTopologyInstrument(drifted)).toEqual({
    ok: false,
    reason: "Sector spectra must agree with the dense reference",
  });
  const leaking = copy();
  const mixed = leaking.templates.sectors.weights.mixed_product_state;
  mixed.after.odd += 0.1;
  mixed.after.even -= 0.1;
  expect(parseTopologyInstrument(leaking)).toEqual({
    ok: false,
    reason: "Parity weights changed under a parity-conserving evolution",
  });
  const unnormalised = copy();
  unnormalised.templates.sectors.weights.even_basis_state.before.even = 0.5;
  expect(parseTopologyInstrument(unnormalised).ok).toBe(false);
});

it("test_topology_symmetry_instrument_03: a clipping kink or discrete edit shows no derivative", () => {
  render(<TopologyInspector instrument={topologyInstrument} />);
  expect(screen.getByRole("table", { name: "Local JVP for the fixed active set" })).toBeTruthy();
  const select = screen.getByLabelText("Primal point");
  for (const [at, blocker] of [
    ["1", "uniform_bounds"],
    ["2", "sign_policy, uniform_bounds"],
    ["3", "algebraic_connectivity_threshold"],
  ] as const) {
    fireEvent.change(select, { target: { value: at } });
    expect(screen.queryByRole("table", { name: "Local JVP for the fixed active set" })).toBeNull();
    expect(screen.getByRole("status").textContent).toBe(
      `No derivative is shown: projection derivative is unsupported for: ${blocker}.`,
    );
  }
  const invented = copy();
  caseNamed(invented, "coupling at the clipping bound").projected_tangent = [[0]] as never;
  expect(parseTopologyInstrument(invented)).toEqual({
    ok: false,
    reason: "An unsupported branch cannot display a derivative",
  });
  const contradicted = copy();
  caseNamed(contradicted, "edge removed").derivative_supported = true;
  expect(parseTopologyInstrument(contradicted)).toEqual({
    ok: false,
    reason: "Support rows contradict the derivative decision",
  });
});

it("test_topology_symmetry_instrument_04: rotating the 3D view changes no physics value", () => {
  render(<TopologyInspector instrument={topologyInstrument} />);
  const before = physics();
  const view = screen.getByRole("img", { name: /positions carry no physical meaning/ });
  const node = () => view.querySelector('[data-node="0"]')?.getAttribute("cx");
  const start = node();
  fireEvent.change(screen.getByLabelText("Rotate view (degrees)"), { target: { value: "200" } });
  fireEvent.change(screen.getByLabelText("Tilt view (degrees)"), { target: { value: "-60" } });
  expect(node()).not.toBe(start);
  expect(physics()).toEqual(before);
  expect(view.querySelectorAll('[data-cluster="0"]').length).toBe(8);
  expect(screen.getByLabelText("Display projection").textContent).toContain(
    "carry no physical meaning",
  );
  const placed = projectNodes(4, 90, 0);
  expect(placed.map((point) => point.node).sort()).toEqual([0, 1, 2, 3]);
});

it("shows a refusal and recovers, and falls back to the first case after a refresh", () => {
  const { rerender } = render(
    <TopologyInspector instrument={{ ok: false, reason: "unsupported source schema" }} />,
  );
  expect(screen.getByRole("alert").textContent).toContain("unsupported source schema");
  rerender(<TopologyInspector instrument={topologyInstrument} />);
  fireEvent.change(screen.getByLabelText("Primal point"), { target: { value: "3" } });
  const fewer = copy();
  fewer.templates.sensitivity.cases = fewer.templates.sensitivity.cases.slice(0, 1);
  rerender(<TopologyInspector instrument={parseTopologyInstrument(fewer)} />);
  expect(screen.getByRole("table", { name: "Local JVP for the fixed active set" })).toBeTruthy();
});

it.each([
  (data: Raw) => {
    data.schema = "future";
  },
  (data: Raw) => {
    (data as Record<string, unknown>)["global_random_state_unchanged"] = false;
  },
  (data: Raw) => {
    data.templates.sectors.spectra.magnetisation_max_deviation = 1e-3;
  },
  (data: Raw) => {
    data.templates.chimera.clusters[1]?.nodes.pop();
  },
  (data: Raw) => {
    data.templates.persistence.exact_example.summary.H0.finite_deaths = [];
  },
  (data: Raw) => {
    data.templates.persistence.exact_example.analytic_spacing = 2;
  },
  (data: Raw) => {
    data.templates.sensitivity.cases = [];
  },
  (data: Raw) => {
    data.templates.sensitivity.tangent = [[0, 1]] as never;
  },
])("refuses malformed or contradictory source data", (mutate) => {
  const data = copy();
  mutate(data);
  expect(parseTopologyInstrument(data).ok).toBe(false);
});
