// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — explicit unimplemented browser workflows

/** Navigable unavailable views with an explicit reason and reachable existing workspace. */
export default function UnavailableView({ view, workspaceHref }: {
  view: "experiments" | "atlas";
  workspaceHref: string;
}) {
  return <article className="qsp-panel">
    <h3>{view === "atlas" ? "Atlas unavailable" : "Experiments unavailable"}</h3>
    <p>{view === "atlas" ? "No Atlas snapshot browser is available in this build. Navigation does not extract or load a snapshot." : "No general experiment builder or job submission is available in this build. Existing committed browser instruments are in Build and Results."}</p>
    <p>Your workspace draft and saved references are retained.</p>
    <a href={workspaceHref}>Open Workspace</a>
  </article>;
}
