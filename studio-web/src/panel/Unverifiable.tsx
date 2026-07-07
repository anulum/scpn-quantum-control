// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web fail-closed evidence block

/**
 * Loud fail-closed block for a committed surface that failed its guard.
 * Rendering this instead of a blank or a partial card is compliance rule 7:
 * malformed evidence is `unverifiable`, never a silent downgrade.
 */
export function Unverifiable({ surface, reason }: { surface: string; reason: string }) {
  return (
    <section className="qsp-unverifiable" role="alert">
      <h3>unverifiable</h3>
      <p>
        The committed surface <code>{surface}</code> failed its fail-closed guard: {reason}
      </p>
    </section>
  );
}
