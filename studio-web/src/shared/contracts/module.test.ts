// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of the workbench module contract and registry

// @vitest-environment node
import { requirePresent } from "../../../test-support/requirePresent";
import { describe, expect, it } from "vitest";
import settingsFixtureText from "../../../../tests/data/studio_workspace/settings.json?raw";
import { readJson, writeJson } from "./jsonTransport";
import { documentDigest, parseResolvedSettings } from "./workspace";
import { createModuleRegistry } from "./module";
import type {
  ModuleDocument,
  ModuleDocumentView,
  ModuleJob,
  ModuleSlot,
  StudioModule,
} from "./module";

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

const settingsFixture = readJson(settingsFixtureText) as { readonly document: unknown };
const admittedSettings = parseResolvedSettings(settingsFixture.document);
if (!admittedSettings.ok) throw new Error(admittedSettings.message);
const retainedDocument: ModuleDocument = {
  schema: admittedSettings.value.schema,
  sha256: await documentDigest(admittedSettings.value),
  content: writeJson(admittedSettings.value),
};

const documentView: ModuleDocumentView<Services> = {
  kind: "document",
  id: "settings-source",
  label: "Original settings",
  schemas: [retainedDocument.schema],
  render: (services, document) => `${services.word}:${document.content}`,
};

const localJob: ModuleJob<Services> = {
  id: "local-worker",
  label: "Local worker",
  runner: "worker",
  stage: "alpha",
  estimate: async () => ({
    kind: "declared-resources",
    plan: {
      backend: "browser",
      method: "declared-test-buffer",
      buffers: [{ name: "input", role: "transfer", shape: [2n], dtype: "float64", count: 1n }],
      concurrency: 1n,
      workUnits: 2n,
    },
  }),
  run: async () => {
    throw new Error("Registration must not execute a job");
  },
};

describe("createModuleRegistry", () => {
  it("registers document, explorer, command and job contributions without changing another module", () => {
    const extension: StudioModule<Services> = {
      ...module("archive-tools", []),
      explorer: [
        {
          id: "saved-documents",
          label: "Saved documents",
          kinds: ["settings"],
          nodes: () => [
            {
              id: "selected-settings",
              kind: "settings",
              label: "Settings",
              document: retainedDocument,
            },
          ],
        },
      ],
      editors: [documentView],
      inspectors: [documentView],
      settings: [documentView],
      commands: [
        {
          id: "open-source",
          label: "Open source",
          action: { kind: "navigate", stage: "alpha", instrument: "probe" },
        },
        { id: "run-local", label: "Run local", action: { kind: "job", job: localJob.id } },
      ],
      jobs: [localJob],
    };
    const base = createModuleRegistry([first]);
    const registry = createModuleRegistry([first, extension]);
    expect([...registry.paths]).toEqual([...base.paths]);
    expect(registry.explorer[0]?.nodes({ word: "source" })[0]?.document).toBe(retainedDocument);
    expect(registry.editors[0]?.render({ word: "source" }, retainedDocument)).toBe(
      `source:${retainedDocument.content}`,
    );
    expect(registry.inspectors).toEqual([documentView]);
    expect(registry.settings).toEqual([documentView]);
    expect(registry.commands.map((command) => command.id)).toEqual(["open-source", "run-local"]);
    expect(registry.jobs).toEqual([localJob]);
    for (const slot of [
      "explorer",
      "editors",
      "inspectors",
      "commands",
      "jobs",
      "settings",
    ] as const) {
      expect(registry.contributionOwner(slot, requirePresent(registry[slot][0]).id)).toBe(
        extension,
      );
      expect(registry.contributionOwner(slot, "not-registered")).toBeUndefined();
      expect(Object.isFrozen(registry[slot])).toBe(true);
      expect(base[slot]).toEqual([]);
    }
    expect(retainedDocument.content).toBe(writeJson(admittedSettings.value));
  });

  it.each(["explorer", "editors", "inspectors", "commands", "jobs", "settings"] as const)(
    "refuses colliding identifiers in the %s slot before a view or job can run",
    (slot: ModuleSlot) => {
      const contributors =
        slot === "explorer"
          ? [{ id: "same", label: "Documents", kinds: [], nodes: () => [] }]
          : slot === "commands"
            ? [
                {
                  id: "same",
                  label: "Open",
                  action: { kind: "navigate", stage: "alpha", instrument: null },
                },
              ]
            : slot === "jobs"
              ? [{ ...localJob, id: "same" }]
              : [{ ...documentView, id: "same" }];
      const extension = { ...module("second", []), [slot]: contributors };
      const duplicate = { ...module("third", []), [slot]: contributors };
      expect(() => createModuleRegistry([first, extension, duplicate])).toThrow(
        `Repeated ${slot} identifier`,
      );
      const malformed = {
        ...module("bad", []),
        [slot]: contributors.map((entry) => ({ ...entry, id: "bad/name" })),
      };
      expect(() => createModuleRegistry([first, malformed])).toThrow(`Invalid ${slot} identifier`);
    },
  );

  it("refuses commands for missing stages, instruments and job kinds", () => {
    for (const action of [
      { kind: "navigate", stage: "missing", instrument: null },
      { kind: "navigate", stage: "alpha", instrument: "missing" },
      { kind: "job", job: "missing" },
    ] as const) {
      const extension = {
        ...module("commands", []),
        commands: [{ id: "open", label: "Open", action }],
      };
      expect(() => createModuleRegistry([first, extension])).toThrow(
        "Command targets an unregistered",
      );
    }
  });
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
