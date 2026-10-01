// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual federated export and host singleton consumer

import * as React from "react";
import * as ReactDOM from "react-dom";
import { createRoot } from "react-dom/client";
import type { QuantumStudioPanelProps } from "../src/QuantumStudioPanel";
import { conformanceCodecs } from "./workspaceFixture";

interface SharedModule {
  get(): Promise<() => unknown>;
  readonly from: string;
  readonly loaded: boolean;
}
interface Remote {
  init(scope: Record<string, Record<string, SharedModule>>): Promise<void>;
  get(name: string): Promise<() => unknown>;
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null) throw new Error("Federation module object required");
  return value as Record<string, unknown>;
}

async function boot(): Promise<void> {
  const url = "/remoteEntry.js";
  const imported: unknown = await import(/* @vite-ignore */ url);
  const record = object(imported);
  if (typeof record["init"] !== "function" || typeof record["get"] !== "function") throw new Error("Original federation container contract missing");
  const remote = record as unknown as Remote;
  const scope = {
    react: { "19.2.7": { get: async () => () => React, from: "workbench-acceptance-host", loaded: true } },
    "react-dom": { "19.2.7": { get: async () => () => ReactDOM, from: "workbench-acceptance-host", loaded: true } },
  };
  await remote.init(scope);
  const factory = await remote.get("./QuantumStudioPanel");
  const exposed = object(factory());
  if (typeof exposed["default"] !== "function" || exposed["default"] !== exposed["QuantumStudioPanel"]) throw new Error("Original named/default panel identity missing");
  const Panel = exposed["default"] as React.ComponentType<QuantumStudioPanelProps>;
  function Host() {
    const [count, setCount] = React.useState(0);
    return <><button onClick={() => setCount(value => value + 1)}>Host singleton counter</button><output aria-label="Federation consumer state">{count}</output><div style={{ width: "100%", maxWidth: "720px" }}><Panel mode="embedded" rawCodecs={conformanceCodecs} /></div></>;
  }
  const container = document.getElementById("consumer");
  if (container === null) throw new Error("Native consumer root missing");
  const root = createRoot(container);
  root.render(<Host />);
  window.addEventListener("pagehide", () => root.unmount(), { once: true });
}

void boot().catch(() => {
  const alert = document.createElement("p");
  alert.setAttribute("role", "alert");
  alert.textContent = "Federated panel unavailable";
  document.body.append(alert);
});
