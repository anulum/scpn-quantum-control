// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — local evidence text viewer


import { useState } from "react";
import { ProgramADReplayCard } from "../../panel/ProgramADReplayCard";
import { PROGRAM_AD_SCHEMA, parseProgramAdUnit } from "../../panel/programAd";
import type { ProgramAdUnit } from "../../panel/programAd";
import { readJson } from "../contracts";
import { EvidenceInspector } from "./EvidenceInspector";
import { projectEvidenceBundle } from "./evidence";
import type { EvidenceView } from "./evidence";

type Snapshot = { readonly kind: "replay"; readonly unit: ProgramAdUnit; readonly revision: string }
  | { readonly kind: "evidence"; readonly view: EvidenceView; readonly revision: string };

/** Inspect pasted source metadata locally; imported data never selects executable code. */
export function EvidenceViewer() {
  const [text, setText] = useState("");
  const [loaded, setLoaded] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inspect = (): void => {
    try {
      const source = readJson(text);
      if (typeof source === "object" && source !== null && "schema" in source && source.schema === PROGRAM_AD_SCHEMA) {
        // This producer defines JSON numbers as binary64; workspace values remain lossless.
        const replay = parseProgramAdUnit(JSON.parse(text) as unknown);
        if (!replay.ok) {
          setLoaded(null);
          setError("Cannot replay evidence — " + replay.reason);
          return;
        }
        setLoaded({ kind: "replay", unit: replay.value, revision: text });
      } else {
        setLoaded({ kind: "evidence", view: projectEvidenceBundle(source), revision: text });
      }
      setError(null);
    } catch {
      setLoaded(null);
      setError("Cannot inspect evidence — invalid JSON or unsupported value");
    }
  };
  return <section className="qsp-evidence-viewer" aria-label="Inspect evidence JSON">
    <h3>Inspect evidence JSON</h3>
    <p>Paste a Studio evidence bundle or the original bounded program-AD replay artefact. Source references are displayed without fetching them.</p>
    <label>Evidence JSON<textarea value={text} onChange={event => setText(event.target.value)} rows={8} /></label>
    <button type="button" onClick={inspect}>Inspect snapshot</button>
    {error !== null && <p role="alert">{error}</p>}
    {loaded !== null && (loaded.kind === "replay" ? <ProgramADReplayCard unit={loaded.unit} inputRevision={loaded.revision} />
      : <EvidenceInspector view={loaded.view} inputRevision={loaded.revision} />)}
  </section>;
}
