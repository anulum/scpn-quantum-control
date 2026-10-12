// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of the workbench module contract and registry

import { describe, expect, it } from "vitest";
import { createModuleRegistry } from "./module";
import type { StudioModule } from "./module";

type Services = { readonly word: string };

function module(
  id: string,
  stages: ReadonlyArray<readonly [string, readonly string[]]>,
  requires: StudioModule<Services>["requires"] = { verbs: [], schemas: [] },
): StudioModule<Services> {
  return {
    id,
    version: "1",
    requires,
    stages: stages.map(([stage, instruments]) => ({
      id: stage,
      label: stage.toUpperCase(),
      group: "primary" as const,
      instruments,
      render: (services, instrument) => `${stage}:${services.word}:${instrument ?? "landing"}`,
    })),
  };
}

const first = module("first", [["alpha", ["probe"]]]);
const second = module("second", [
  ["beta", []],
  ["gamma", ["one", "two"]],
]);

describe("createModuleRegistry", () => {
  it("lists stages in registration order and derives exactly their addresses", () => {
    const registry = createModuleRegistry([first, second]);
    expect(registry.modules.map((entry) => entry.id)).toEqual(["first", "second"]);
    expect(registry.stages.map((stage) => stage.id)).toEqual(["alpha", "beta", "gamma"]);
    expect([...registry.paths]).toEqual([
      ["/alpha", { view: "alpha", instrument: null }],
      ["/alpha/probe", { view: "alpha", instrument: "probe" }],
      ["/beta", { view: "beta", instrument: null }],
      ["/gamma", { view: "gamma", instrument: null }],
      ["/gamma/one", { view: "gamma", instrument: "one" }],
      ["/gamma/two", { view: "gamma", instrument: "two" }],
    ]);
    expect(registry.stage("gamma")?.render({ word: "w" }, "two")).toBe("gamma:w:two");
    expect(registry.stage("alpha")?.render({ word: "w" }, null)).toBe("alpha:w:landing");
    expect(registry.stage("absent")).toBeUndefined();
    expect(registry.owner("alpha")).toBe(first);
    expect(registry.owner("gamma")).toBe(second);
    expect(registry.owner("absent")).toBeUndefined();
  });

  it("adds a module without touching the others and removes it the same way", () => {
    const without = createModuleRegistry([first]);
    const withSecond = createModuleRegistry([first, second]);
    expect([...withSecond.paths].slice(0, without.paths.size)).toEqual([...without.paths]);
    expect(withSecond.stage("alpha")).toBe(without.stage("alpha"));
    expect(without.stage("beta")).toBeUndefined();
    expect(without.paths.has("/gamma/one")).toBe(false);
  });

  it("is immutable and does not alias the caller's list", () => {
    const source = [first];
    const registry = createModuleRegistry(source);
    source.push(second);
    expect(registry.modules).toHaveLength(1);
    expect(Object.isFrozen(registry)).toBe(true);
    expect(Object.isFrozen(registry.modules)).toBe(true);
    expect(Object.isFrozen(registry.stages)).toBe(true);
    expect(Object.isFrozen(registry.paths.get("/alpha"))).toBe(true);
  });

  it.each([
    ["a repeated module", [first, module("first", [["other", []]])], "Repeated module identifier"],
    ["a repeated stage", [first, module("third", [["alpha", []]])], "Repeated stage identifier"],
    ["a repeated instrument", [module("m", [["s", ["x", "x"]]])], "Repeated instrument identifier"],
    ["an upper-case module", [module("First", [["s", []]])], "Invalid module identifier"],
    ["a stage with a slash", [module("m", [["a/b", []]])], "Invalid stage identifier"],
    ["an empty instrument", [module("m", [["s", [""]]])], "Invalid instrument identifier"],
    ["an overlong stage", [module("m", [["a".repeat(41), []]])], "Invalid stage identifier"],
  ])("refuses %s", (_label, modules, message) => {
    expect(() => createModuleRegistry(modules)).toThrow(message);
  });

  it("accepts an identifier of exactly forty characters", () => {
    const longest = `a${"b".repeat(39)}`;
    expect(createModuleRegistry([module("m", [[longest, []]])]).paths.has(`/${longest}`)).toBe(
      true,
    );
  });

  it("names every requirement the manifest does not declare, and nothing else", () => {
    const needy = module("needy", [["delta", []]], {
      verbs: ["compile", "simulate"],
      schemas: ["studio.quantum-evolution.v1"],
    });
    const registry = createModuleRegistry([first, needy]);
    expect(registry.missing({ verbs: ["compile"], schemas: [] })).toEqual([
      { module: "needy", kind: "verb", name: "simulate" },
      { module: "needy", kind: "schema", name: "studio.quantum-evolution.v1" },
    ]);
    expect(
      registry.missing({
        verbs: ["compile", "simulate"],
        schemas: ["studio.quantum-evolution.v1"],
      }),
    ).toEqual([]);
  });
});
