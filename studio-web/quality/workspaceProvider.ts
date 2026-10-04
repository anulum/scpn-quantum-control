// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public Vitest native coverage adapter

import v8 from "@vitest/coverage-v8";
import type { V8CoverageProvider } from "@vitest/coverage-v8/dist/provider.js";
import type { CoverageMapData } from "istanbul-lib-coverage";
import type { CoverageProviderModule } from "vitest/node";

import { readBrowserCoverage } from "./browserCoverage";

/** Preserve the locked V8 provider lifecycle and add real browser counters before its original gates. */
const provider: CoverageProviderModule = {
  ...v8,
  async getProvider() {
    // The locked factory constructs this exported SDK class; its map remains SDK-owned.
    const delegate = await v8.getProvider() as V8CoverageProvider;
    const initialize = delegate.initialize.bind(delegate);
    const generateCoverage = delegate.generateCoverage.bind(delegate);
    let browser: CoverageMapData[] | undefined;
    delegate.initialize = async context => {
      const filename = process.env["STUDIO_WORKSPACE_COVERAGE"];
      if (!filename) throw new Error("STUDIO_WORKSPACE_COVERAGE must name the actual browser journey evidence");
      browser = await readBrowserCoverage(filename, context.config.root);
      const panelFilename = process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
      if (panelFilename) browser.push(...await readBrowserCoverage(panelFilename, context.config.root));
      const workbenchFilename = process.env["STUDIO_WORKBENCH_COVERAGE"];
      if (workbenchFilename) browser.push(...await readBrowserCoverage(workbenchFilename, context.config.root));
      const parameterFilename = process.env["STUDIO_PARAMETER_COVERAGE"];
      if (parameterFilename) browser.push(...await readBrowserCoverage(parameterFilename, context.config.root));
      const experimentFilename = process.env["STUDIO_EXPERIMENT_COVERAGE"];
      if (experimentFilename) browser.push(...await readBrowserCoverage(experimentFilename, context.config.root));
      const resultFilename = process.env["STUDIO_RESULT_COVERAGE"];
      if (resultFilename) browser.push(...await readBrowserCoverage(resultFilename, context.config.root));
      await initialize(context);
    };
    delegate.generateCoverage = async context => {
      if (browser === undefined) throw new Error("Native browser coverage was not initialized");
      const node = await generateCoverage(context);
      for (const map of browser) node.merge(map);
      return node;
    };
    return delegate;
  },
};

export default provider;
