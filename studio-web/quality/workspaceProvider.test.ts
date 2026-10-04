// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public coverage provider lifecycle tests

// @vitest-environment node
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import v8 from "@vitest/coverage-v8";
import { expect, it } from "vitest";
import { createVitest } from "vitest/node";
import type { Vitest } from "vitest/node";
import { browserOwners, coverageObject, experimentBrowserOwners, parameterBrowserOwners, workbenchBrowserOwners } from "./browserRecord";
import provider from "./workspaceProvider";

it("refuses generation before real browser admission initializes the provider", async () => {
  const instance = await provider.getProvider();
  await expect(instance.generateCoverage({ allTestsRun: false })).rejects.toThrow("Native browser coverage was not initialized");
});

it("refuses missing browser evidence with an actual public Vitest context", async () => {
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-missing-"));
  const previous = process.env["STUDIO_WORKSPACE_COVERAGE"];
  let context: Vitest | null = null;
  try {
    context = await createVitest("test", { config: false, root: process.cwd(), watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory } });
    delete process.env["STUDIO_WORKSPACE_COVERAGE"];
    const instance = await provider.getProvider();
    await expect(instance.initialize(context)).rejects.toThrow("STUDIO_WORKSPACE_COVERAGE must name the actual browser journey evidence");
    await expect(instance.generateCoverage({ allTestsRun: false })).rejects.toThrow("Native browser coverage was not initialized");
  } finally {
    if (previous === undefined) delete process.env["STUDIO_WORKSPACE_COVERAGE"];
    else process.env["STUDIO_WORKSPACE_COVERAGE"] = previous;
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});

it("merges actual browser counters through the original public V8 provider before reporting", async () => {
  if (!process.env["STUDIO_WORKSPACE_COVERAGE"]) throw new Error("Run native workspace_recovery first and supply its actual evidence path");
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-native-"));
  const root = process.cwd();
  let context: Vitest | null = null;
  try {
    context = await createVitest("test", { config: false, root, watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory, thresholds: { statements: 100, branches: 100, functions: 100, lines: 100 } } });
    const instance = await provider.getProvider();
    const original = await v8.getProvider();
    expect(instance.reportCoverage).toBe(original.reportCoverage);
    expect(instance.onAfterSuiteRun).toBe(original.onAfterSuiteRun);
    await instance.initialize(context);
    expect(instance.resolveOptions().thresholds).toMatchObject({ statements: 100, branches: 100, functions: 100, lines: 100 });
    await instance.clean(true);
    const result = await instance.generateCoverage({ allTestsRun: false });
    const map = coverageObject(result);
    const files = map["files"];
    if (typeof files !== "function") throw new Error("Original provider did not return its public coverage map");
    const observed: unknown = files.call(result);
    expect(observed).toEqual(expect.arrayContaining([...browserOwners].map(owner => resolve(root, "." + owner))));
  } finally {
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});


it("keeps ordinary workspace evidence sufficient when no damaged-source cohort is supplied", async () => {
  if (!process.env["STUDIO_WORKSPACE_COVERAGE"]) throw new Error("Supply actual workspace_recovery evidence");
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-workspace-only-"));
  const panel = process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
  const workbench = process.env["STUDIO_WORKBENCH_COVERAGE"];
  const parameters = process.env["STUDIO_PARAMETER_COVERAGE"];
  const experiments = process.env["STUDIO_EXPERIMENT_COVERAGE"];
  let context: Vitest | null = null;
  try {
    delete process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
    delete process.env["STUDIO_WORKBENCH_COVERAGE"];
    delete process.env["STUDIO_PARAMETER_COVERAGE"];
    delete process.env["STUDIO_EXPERIMENT_COVERAGE"];
    context = await createVitest("test", { config: false, root: process.cwd(), watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory } });
    const instance = await provider.getProvider();
    await instance.initialize(context);
    await instance.clean(true);
    const actual = coverageObject(await instance.generateCoverage({ allTestsRun: false }));
    const files = actual["files"];
    if (typeof files !== "function") throw new Error("Original provider omitted its coverage map API");
    expect(files.call(actual)).toEqual(expect.arrayContaining([...browserOwners].map(owner => resolve(process.cwd(), "." + owner))));
  } finally {
    if (panel === undefined) delete process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
    else process.env["STUDIO_PANEL_REFUSAL_COVERAGE"] = panel;
    if (workbench === undefined) delete process.env["STUDIO_WORKBENCH_COVERAGE"];
    else process.env["STUDIO_WORKBENCH_COVERAGE"] = workbench;
    if (parameters === undefined) delete process.env["STUDIO_PARAMETER_COVERAGE"];
    else process.env["STUDIO_PARAMETER_COVERAGE"] = parameters;
    if (experiments === undefined) delete process.env["STUDIO_EXPERIMENT_COVERAGE"];
    else process.env["STUDIO_EXPERIMENT_COVERAGE"] = experiments;
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});

it("merges all ten genuine experiment owners through the unchanged public provider", async () => {
  if (!process.env["STUDIO_WORKSPACE_COVERAGE"] || !process.env["STUDIO_EXPERIMENT_COVERAGE"]) throw new Error("Supply actual workspace and experiment evidence");
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-experiments-"));
  let context: Vitest | null = null;
  try {
    context = await createVitest("test", { config: false, root: process.cwd(), watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory } });
    const instance = await provider.getProvider();
    await instance.initialize(context);
    await instance.clean(true);
    const actual = coverageObject(await instance.generateCoverage({ allTestsRun: false }));
    const files = actual["files"];
    if (typeof files !== "function") throw new Error("Original provider omitted its coverage map API");
    expect(files.call(actual)).toEqual(expect.arrayContaining([...experimentBrowserOwners].map(owner => resolve(process.cwd(), "." + owner))));
  } finally {
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});

it("merges every actual workbench owner through the original public provider", async () => {
  if (!process.env["STUDIO_WORKSPACE_COVERAGE"] || !process.env["STUDIO_WORKBENCH_COVERAGE"]) throw new Error("Supply actual workspace and workbench evidence");
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-workbench-"));
  let context: Vitest | null = null;
  try {
    context = await createVitest("test", { config: false, root: process.cwd(), watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory } });
    const instance = await provider.getProvider();
    await instance.initialize(context);
    await instance.clean(true);
    const actual = coverageObject(await instance.generateCoverage({ allTestsRun: false }));
    const files = actual["files"];
    if (typeof files !== "function") throw new Error("Original provider omitted its coverage map API");
    expect(files.call(actual)).toEqual(expect.arrayContaining([...workbenchBrowserOwners].map(owner => resolve(process.cwd(), "." + owner))));
  } finally {
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});

it("merges all eight actual parameter owners through the original public provider", async () => {
  if (!process.env["STUDIO_WORKSPACE_COVERAGE"] || !process.env["STUDIO_PARAMETER_COVERAGE"]) throw new Error("Supply actual workspace and parameter evidence");
  const directory = await mkdtemp(join(tmpdir(), "studio-provider-parameters-"));
  let context: Vitest | null = null;
  try {
    context = await createVitest("test", { config: false, root: process.cwd(), watch: false, coverage: { enabled: true, provider: "v8", reportsDirectory: directory } });
    const instance = await provider.getProvider();
    await instance.initialize(context);
    await instance.clean(true);
    const actual = coverageObject(await instance.generateCoverage({ allTestsRun: false }));
    const files = actual["files"];
    if (typeof files !== "function") throw new Error("Original provider omitted its coverage map API");
    expect(files.call(actual)).toEqual(expect.arrayContaining([...parameterBrowserOwners].map(owner => resolve(process.cwd(), "." + owner))));
  } finally {
    try { await context?.close(); }
    finally { await rm(directory, { recursive: true }); }
  }
});
