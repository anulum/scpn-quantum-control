// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — independent federation consumer build

import { fileURLToPath } from "node:url";
import { federation } from "@module-federation/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { moduleFederationConfig } from "../module-federation.config";

/** Separate host renderer/provider bundle; production remote is loaded through its real get/init API. */
export default defineConfig({
  base: "/acceptance-host/",
  plugins: [react(), federation({
    name: "quantum_studio_acceptance_host",
    filename: "acceptanceHostEntry.js",
    shared: { ...moduleFederationConfig.shared },
  })],
  publicDir: false,
  build: { target: "esnext", rollupOptions: { input: fileURLToPath(new URL("./federation.html", import.meta.url)) } },
});
