// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Vite build (portal + MF remote)

/// <reference types="vitest/config" />
import { federation } from "@module-federation/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { moduleFederationConfig } from "./module-federation.config";

export default defineConfig({
  plugins: [react(), federation({ ...moduleFederationConfig })],
  build: {
    target: "esnext",
    // The committed evidence artefacts live one level above this workspace;
    // Vite inlines them at build time so the served panel needs no API.
    assetsInlineLimit: 0,
  },
  server: {
    fs: {
      // Allow build-time imports of the committed repo artefacts.
      allow: [".."],
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/**/*.test.{ts,tsx}"],
      thresholds: {
        statements: 95,
        branches: 95,
        functions: 95,
        lines: 95,
      },
    },
  },
});
