// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — schema-bound original settings document inspector

import { useEffect, useState } from "react";
import { readJson } from "../../shared/contracts";
import type { ModuleDocument } from "../../shared/contracts/module";
import { documentDigest, parseResolvedSettings } from "../../shared/contracts/workspace";
import type { ResolvedSettings } from "../../shared/contracts/workspace";
import { SettingsRecord } from "./SettingsInspector";

/** Parse and bind an original settings document before displaying its source values. */
export function SettingsDocumentInspector({
  document,
}: {
  /** Unchanged admitted source record. */ readonly document: ModuleDocument;
}) {
  const [state, setState] = useState<{
    readonly content: string;
    readonly digest: string;
    readonly value: ResolvedSettings | null;
  } | null>(null);
  useEffect(() => {
    let live = true;
    const inspect = async () => {
      try {
        const parsed = parseResolvedSettings(readJson(document.content));
        if (!parsed.ok || (await documentDigest(parsed.value)) !== document.sha256)
          throw new Error("Settings identity refused");
        if (live)
          setState({ content: document.content, digest: document.sha256, value: parsed.value });
      } catch {
        if (live) setState({ content: document.content, digest: document.sha256, value: null });
      }
    };
    void inspect();
    return () => {
      live = false;
    };
  }, [document.content, document.sha256]);
  if (state?.content !== document.content || state.digest !== document.sha256)
    return <p role="status">Reading original settings document…</p>;
  if (state.value === null)
    return <p role="alert">Settings document identity refused; original source retained.</p>;
  return (
    <section aria-label="Settings document inspector">
      <h3>Resolved settings</h3>
      <SettingsRecord digest={state.digest} document={state.value} />
    </section>
  );
}
