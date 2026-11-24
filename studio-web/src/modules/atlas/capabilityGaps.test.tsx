// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — capability gap report admission and panel tests

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import reportText from "../../../../data/studio/atlas_capability_gaps_20261123.json?raw";
import AtlasView from "./AtlasView";
import { CapabilityGaps } from "./CapabilityGaps";
import {
  type CapabilityEvidence,
  admitCapabilityGaps,
  admitCommittedCapabilityGaps,
  capabilityGapsSchema,
} from "./capabilityGaps";

afterEach(cleanup);

type Json = Record<string, unknown>;

function report(change: (body: Json) => void = () => {}): string {
  const body: Json = {
    schema: capabilityGapsSchema,
    capabilities: [
      {
        capability: "core",
        node_id: "capability:atlas.architecture/core",
        title: "Core",
        tests: [
          {
            path: "tests/test_core.py",
            present: true,
            behaviour: true,
            ci_state: "explicit",
            ci_owners: ["a.yml#one"],
          },
          {
            path: "tests/test_dup.py",
            present: true,
            behaviour: true,
            ci_state: "duplicate",
            ci_owners: ["a.yml#one", "b.yml#two"],
          },
          {
            path: "engine/core.jl",
            present: true,
            behaviour: null,
            ci_state: "unknown",
            ci_owners: [],
          },
          {
            path: "tests/test_gone.py",
            present: false,
            behaviour: null,
            ci_state: "unknown",
            ci_owners: [],
          },
          {
            path: "tests/test_empty.py",
            present: true,
            behaviour: false,
            ci_state: "unknown",
            ci_owners: [],
          },
        ],
        docs: [
          { path: "docs/core.md", present: true },
          { path: "docs/gone.md", present: false },
        ],
        ui: [
          { path: "studio-web/src/Core.tsx", present: false },
          { path: "studio-web/src/CoreView.tsx", present: true },
        ],
        code_coverage: "unmeasured",
        modules: [{ path: "src/core.py", percent: null }],
        support_ready: false,
        support_blockers: ["still Python-first"],
        scientific: [{ claim_id: "core_claim", promotion_status: "bounded_candidate" }],
      },
      {
        capability: "quiet",
        node_id: "capability:atlas.architecture/quiet",
        title: "Quiet",
        tests: [],
        docs: [],
        ui: [],
        code_coverage: "unmeasured",
        modules: [],
        support_ready: true,
        support_blockers: [],
        scientific: [],
      },
      {
        capability: "settled",
        node_id: "capability:atlas.architecture/settled",
        title: "Settled",
        tests: [],
        docs: [],
        ui: [{ path: "studio-web/src/Settled.tsx", present: true }],
        code_coverage: "measured",
        modules: [],
        support_ready: true,
        support_blockers: [],
        scientific: [],
      },
    ],
    gaps: [
      {
        capability: "core",
        kind: "duplicate_ci_owner",
        declared_in: "map.json",
        surface: "tests/test_dup.py",
        detail: "executed by name by 2 jobs",
      },
      {
        capability: "core",
        kind: "missing_doc",
        declared_in: "map.json",
        surface: "docs/gone.md",
        detail: "the declared documentation does not exist",
      },
    ],
    unknowns: [
      { capability: "core", statement: "code coverage: unmeasured" },
      { capability: "quiet", statement: "user-route availability: no UI surface is declared" },
    ],
  };
  change(body);
  return JSON.stringify(body);
}
function evidence(node: string, source = report()): CapabilityEvidence {
  const admission = admitCapabilityGaps(source);
  if (!admission.ok) throw new Error(admission.reason);
  const found = admission.byNode.get(node);
  if (found === undefined) throw new Error("expected the capability");
  return found;
}
function reason(source: string): string {
  const admission = admitCapabilityGaps(source);
  if (admission.ok) throw new Error("expected a refusal");
  return admission.reason;
}
function first(body: Json): Json {
  return (body["capabilities"] as Json[])[0] as Json;
}

describe("Capability gap report admission", () => {
  it("admits the committed report with one entry per architecture capability", () => {
    const admission = admitCommittedCapabilityGaps();
    if (!admission.ok) throw new Error(admission.reason);
    const committed = JSON.parse(reportText) as { capabilities: { node_id: string }[] };
    expect([...admission.byNode.keys()]).toEqual(
      committed.capabilities.map((item) => item.node_id),
    );
  });

  it("joins each capability with its own gaps and unknowns", () => {
    const core = evidence("capability:atlas.architecture/core");
    expect(core.gaps.map((gap) => gap.kind)).toEqual(["duplicate_ci_owner", "missing_doc"]);
    expect(core.unknowns).toEqual(["code coverage: unmeasured"]);
    expect(core.scientific).toEqual([["core_claim", "bounded_candidate"]]);
    expect(core.tests[2]?.behaviour).toBeNull();
    const quiet = evidence("capability:atlas.architecture/quiet");
    expect(quiet.gaps).toEqual([]);
    expect(quiet.unknowns).toEqual(["user-route availability: no UI surface is declared"]);
    const settled = evidence("capability:atlas.architecture/settled");
    expect([settled.gaps, settled.unknowns, settled.codeCoverage]).toEqual([[], [], "measured"]);
  });

  it("refuses text, schemas and every malformed field", () => {
    expect(reason("{not json")).toBe("The capability gap report was refused: it is not JSON.");
    expect(reason("[]")).toContain("$ must be an object");
    expect(reason(report((body) => (body["schema"] = "other")))).toContain("unknown schema");
    expect(reason(report((body) => (body["gaps"] = {})))).toContain("$.gaps must be an array");
    expect(
      reason(
        report((body) => ((body["gaps"] as Json[])[0] = { capability: "core", kind: "guess" })),
      ),
    ).toContain("$.gaps[0].kind is not one of");
    expect(
      reason(report((body) => ((body["unknowns"] as Json[])[0] = { capability: "core" }))),
    ).toContain("$.unknowns[0].statement must be text");
    expect(reason(report((body) => (first(body)["node_id"] = "")))).toContain(
      "$.capabilities[0].node_id must be text",
    );
    expect(
      reason(
        report((body) => (((first(body)["tests"] as Json[])[0] as Json)["behaviour"] = "yes")),
      ),
    ).toContain("behaviour must be a boolean");
    expect(
      reason(report((body) => (((first(body)["tests"] as Json[])[0] as Json)["ci_owners"] = [3]))),
    ).toContain("ci_owners[0] must be text");
    expect(
      reason(report((body) => (first(body)["docs"] = [{ path: "x", present: "no" }]))),
    ).toContain("$.capabilities[0].docs[0].present must be a boolean");
    expect(reason(report((body) => (first(body)["code_coverage"] = "full")))).toContain(
      "code_coverage is not one of measured, unmeasured",
    );
    expect(reason(report((body) => (first(body)["support_blockers"] = [""])))).toContain(
      "support_blockers[0] must be text",
    );
    expect(reason(report((body) => (first(body)["scientific"] = [{ claim_id: "c" }])))).toContain(
      "scientific[0].promotion_status must be text",
    );
  });
});

describe("Capability gaps panel", () => {
  it("keeps each kind of evidence apart and names every gap with its owner", () => {
    render(<CapabilityGaps evidence={evidence("capability:atlas.architecture/core")} />);
    const kinds = screen.getByLabelText("Kinds of evidence");
    expect(kinds.textContent).toBe(
      "Code coverageunmeasuredCapability supportnot ready; blockers: still Python-firstScientific validationcore_claim: bounded_candidateUser routesstudio-web/src/Core.tsx (missing); studio-web/src/CoreView.tsx (present)",
    );
    const tests = screen.getByRole("table", { name: "Declared tests" });
    expect(
      within(tests)
        .getAllByRole("row")
        .slice(1)
        .map((row) => row.textContent),
    ).toEqual([
      "tests/test_core.pydefines tests; run by a.yml#one",
      "tests/test_dup.pydefines tests; run by 2 jobs: a.yml#one, b.yml#two",
      "engine/core.jlbehaviour not judged; no CI job names it",
      "tests/test_gone.pymissing",
      "tests/test_empty.pydefines no test; no CI job names it",
    ]);
    fireEvent.click(screen.getByRole("button", { name: "Scroll Declared tests" }));
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "Declared tests table scrolling" }),
    );
    expect(screen.getByLabelText("Declared documentation").textContent).toContain(
      "docs/gone.md (missing)",
    );
    expect(screen.getByLabelText("Gaps").textContent).toContain(
      "missing_doc: docs/gone.md, declared in map.json. the declared documentation does not exist.",
    );
    expect(screen.getByLabelText("Unknown").textContent).toBe("code coverage: unmeasured");
    expect(screen.queryByText(/score/i)).toBeNull();
  });

  it("says when a capability has no gap, no UI surface and no assessed science", () => {
    render(<CapabilityGaps evidence={evidence("capability:atlas.architecture/quiet")} />);
    expect(screen.getByText("No gap in the declared mappings.")).toBeTruthy();
    const kinds = screen.getByLabelText("Kinds of evidence").textContent;
    expect(kinds).toContain("Capability supportready");
    expect(kinds).toContain("Scientific validationunassessed");
    expect(kinds).toContain("User routesunknown: no UI surface is declared");
  });
});

describe("Capability gaps in the Atlas", () => {
  it("opens a capability's coverage and gaps from the architecture lens", async () => {
    render(<AtlasView instrument="architecture" contextSuffix="" />);
    const table = await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    const committed = JSON.parse(reportText) as {
      capabilities: { title: string; node_id: string }[];
    };
    const capability = committed.capabilities[0];
    if (capability === undefined) throw new Error("expected a capability");
    const row = within(table)
      .getAllByRole("button")
      .find((button) => button.textContent === capability.title);
    if (row === undefined) throw new Error("expected the capability row");
    fireEvent.click(row);
    expect(
      within(screen.getByRole("region", { name: "Atlas inspector" })).getByRole("region", {
        name: "Coverage and gaps",
      }),
    ).toBeTruthy();
  }, 30_000);

  it("reports a refused report and shows no panel", async () => {
    render(
      <AtlasView
        instrument="architecture"
        contextSuffix=""
        capabilityGaps={() => ({
          ok: false,
          reason: "The capability gap report was refused: fixture.",
        })}
      />,
    );
    expect(await screen.findByText("The capability gap report was refused: fixture.")).toBeTruthy();
    const table = await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    fireEvent.click(within(table).getAllByRole("button")[0] as HTMLElement);
    expect(screen.queryByRole("region", { name: "Coverage and gaps" })).toBeNull();
  }, 30_000);
});
