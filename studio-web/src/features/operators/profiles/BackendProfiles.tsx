// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — offline provider and device profiles

import { useEffect, useId, useRef, useState } from "react";
import declaredExample from "../../../../../data/studio/backend_profiles.json?raw";
import { declaredCapabilities, parseBackendProfiles, profileAge, selectBackendProfile } from "./backendProfiles";
import type { BackendProfile, BackendProfileSnapshot, ProfileBinding } from "./backendProfiles";

/** Render source declarations and observations without treating either as approval. */
function ProfileDetails({ profile, binding }: {
  /** Exact admitted dated profile. */ profile: BackendProfile;
  /** Optional metadata references for this exact row. */ binding: ProfileBinding;
}) {
  const observed = profile.observed;
  const prefix = useId(), pulseReason = useRef<HTMLParagraphElement>(null), analogReason = useRef<HTMLParagraphElement>(null);
  const reasons = { pulse: pulseReason, analog: analogReason };
  return <section aria-label="Selected backend profile">
    <h4>{profile.provider} · {profile.device}</h4>
    <dl><dt>Route</dt><dd>{profile.route_id}</dd><dt>Broker</dt><dd>{profile.broker ?? "direct"}</dd>
      <dt>HAL backend</dt><dd>{profile.backend_id}</dd><dt>Modality</dt><dd>{profile.modality}</dd>
      <dt>Region</dt><dd>{profile.region ?? "unknown"}</dd><dt>SDK declaration</dt><dd>{profile.sdk_package}</dd>
      <dt>IR declarations</dt><dd>{profile.ir_formats.join(", ")}</dd><dt>Profile SHA-256</dt><dd><code>{profile.sha256}</code></dd>
      <dt>Online observation</dt><dd aria-label="Online observation">{observed.online === null ? "unknown" : observed.online ? "online (supplied observation)" : "offline (supplied observation)"}</dd>
      <dt>Observed IR formats</dt><dd>{observed.ir_formats?.join(", ") ?? "unknown"}</dd>
      <dt>Observed basis gates</dt><dd>{observed.basis_gates?.join(", ") ?? "unknown"}</dd>
      <dt>Observed native features</dt><dd>{observed.native_features?.join(", ") ?? "unknown"}</dd>
      <dt>Observed qubits</dt><dd>{observed.n_qubits?.toString() ?? "unknown"}</dd>
      <dt>Observed shot ceiling</dt><dd aria-label="Observed shot ceiling">{observed.max_shots?.toString() ?? "unknown"}</dd>
      <dt>Observed circuit ceiling</dt><dd>{observed.max_circuits?.toString() ?? "unknown"}</dd>
      <dt>Observed queue depth</dt><dd>{observed.queue_depth?.toString() ?? "unknown"}</dd>
      <dt>Calibration timestamp</dt><dd>{observed.calibration_timestamp ?? "unknown"}</dd>
      <dt>Calibration source reference</dt><dd>{observed.calibration_ref ?? "unknown"}</dd>
      <dt>Credential configuration references</dt><dd>{profile.credential_refs === null ? "unknown" : profile.credential_refs.length === 0 ? "none declared" : profile.credential_refs.join(", ")}</dd>
    </dl>
    <p>Supplied observations are producer declarations. Snapshot age does not certify calibration freshness, authentication or runtime readiness.</p>
    <table aria-label="Declared HAL capabilities"><thead><tr><th>Capability</th><th>Declared</th></tr></thead><tbody>
      {declaredCapabilities.map(key => <tr key={key}><td>{key}</td><td>{profile.declared[key] ? "supported" : "unsupported"}</td></tr>)}
      <tr><td>max_qubits</td><td>{profile.declared.max_qubits?.toString() ?? "unknown"}</td></tr>
    </tbody></table>
    <table aria-label="Route operation support"><thead><tr><th>Operation</th><th>Declared</th><th>Observed</th><th>Provenance</th></tr></thead><tbody>
      {profile.verbs.map(verb => <tr key={verb.verb}><td>{verb.verb}</td><td>{verb.declared === null ? "unknown" : verb.declared ? "supported" : "unsupported"}</td><td>{verb.observed === null ? "unknown" : verb.observed ? "observed" : "not observed"}</td><td>{verb.declared_source ?? "unknown"} / {verb.declared_on ?? "unknown"}; {verb.conformance_owner ?? "unknown"} / {verb.observed_on ?? "unknown"}</td></tr>)}
    </tbody></table>
    {(["pulse", "analog"] as const).map(name => <div key={name}>
      <button type="button" disabled={!profile.options[name].supported} aria-describedby={`${prefix}-${name}-reason`} onClick={() => reasons[name].current!.focus()}>{name === "pulse" ? "Pulse options" : "Analog options"}</button>
      <p tabIndex={-1} id={`${prefix}-${name}-reason`} ref={reasons[name]}>{profile.options[name].reason}</p>
    </div>)}
    <section aria-label="Profile-bound metadata references"><h5>Imported references — metadata only</h5>
      <p>These digests do not approve a run. Switching device, broker, route or dated metadata clears all dependent references.</p>
      <dl><dt>Plan reference</dt><dd aria-label="Plan reference">{binding.plan_ref ?? "none"}</dd><dt>Bound calibration reference</dt><dd aria-label="Bound calibration reference">{binding.calibration_ref ?? "none"}</dd><dt>Review reference</dt><dd aria-label="Review reference">{binding.approval_ref ?? "none"}</dd></dl>
    </section>
  </section>;
}

/** Inspect and select dated offline profiles without network calls or secret storage. */
export function BackendProfiles() {
  const [draft, setDraft] = useState(""), [snapshot, setSnapshot] = useState<BackendProfileSnapshot | null>(null);
  const [selectedRoute, setSelectedRoute] = useState(""), [binding, setBinding] = useState<ProfileBinding | null>(null);
  const [error, setError] = useState<string | null>(null), [pending, setPending] = useState(false);
  const ticket = useRef(0), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; ticket.current++; }; }, []);
  function change(value: string) { ticket.current++; setDraft(value); setError(null); setPending(false); }
  async function inspect(value: string) {
    const current = ++ticket.current; setPending(true); setError(null);
    const result = await parseBackendProfiles(value);
    if (!mounted.current || current !== ticket.current) return;
    setPending(false);
    if (!result.ok) { setError(result.message); return; }
    const imported = result.value;
    const profile = imported.binding === null ? imported.profiles.find(row => row.route_id === selectedRoute) : imported.profiles.find(row => row.sha256 === imported.binding!.profile_sha256);
    setSnapshot(imported); setSelectedRoute(profile?.route_id ?? "");
    setBinding(profile === undefined ? null : imported.binding ?? selectBackendProfile(binding, profile));
  }
  function select(route: string) {
    ticket.current++; setPending(false); setSelectedRoute(route);
    const profile = snapshot!.profiles.find(row => row.route_id === route);
    setBinding(profile === undefined ? null : selectBackendProfile(binding, profile));
  }
  function download(value: BackendProfileSnapshot) {
    const url = URL.createObjectURL(new Blob([value.text], { type: "application/json;charset=utf-8" }));
    try { const anchor = document.createElement("a"); anchor.href = url; anchor.download = "backend-profiles.json"; anchor.click(); }
    finally { URL.revokeObjectURL(url); }
  }
  const profile = snapshot?.profiles.find(row => row.route_id === selectedRoute);
  return <section className="qsp-workspace" aria-label="Backend profiles">
    <h3>Provider and device profiles</h3>
    <p>Load dated native metadata or inspect the declared catalogue. This view does not refresh provider metadata, read credentials or submit jobs. Selection is local to this view.</p>
    <label>Backend profiles JSON<textarea aria-label="Backend profiles JSON" rows={6} value={draft} spellCheck={false} onChange={event => change(event.target.value)} /></label>
    <button type="button" disabled={pending} onClick={() => { void inspect(draft); }}>{pending ? "Inspecting profiles…" : "Inspect profiles"}</button>
    <button type="button" onClick={() => { change(declaredExample); void inspect(declaredExample); }}>Open declared profiles</button>
    <button type="button" disabled={snapshot === null} onClick={snapshot === null ? undefined : () => download(snapshot)}>Export admitted profiles</button>
    {error !== null && <p role="alert">{error}</p>}
    {snapshot === null && <p role="status">No profile snapshot admitted.</p>}
    {snapshot !== null && <section aria-label="Admitted backend profiles">
      <p>Snapshot date: {snapshot.observedAt}. Snapshot age: {profileAge(snapshot.observedAt)}.</p>
      <p>Envelope SHA-256: <code>{snapshot.sha256}</code>. Offline metadata — read-only.</p>
      <label>Backend profile<select aria-label="Backend profile" value={selectedRoute} onChange={event => select(event.target.value)}>
        <option value="">No profile selected</option>
        {snapshot.profiles.map(row => <option key={row.route_id} value={row.route_id}>{row.provider} / {row.broker ?? "direct"} / {row.device} / {row.route_id}</option>)}
      </select></label>
      {profile !== undefined && binding !== null && <ProfileDetails profile={profile} binding={binding} />}
    </section>}
  </section>;
}
