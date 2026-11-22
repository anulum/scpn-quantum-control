// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas snapshot pin admission for the browser

import { type AtlasSnapshot, parseAtlasSnapshot } from "../../shared/contracts/atlas";
import { rememberedAtlasSnapshotDigest } from "../../shared/contracts/atlasQuery";

/** Schema of the committed pin written by `tools/studio_atlas_snapshot.py`. */
export const atlasSnapshotPinSchema = "scpn_qc_studio_atlas_snapshot_pin_v1";

/** Admitted snapshot, or the reason it was refused. */
export type AtlasAdmission =
  | {
      /** The pin parsed, passed the contract and matched its digest. */
      readonly ok: true;
      /** The validated snapshot. */
      readonly snapshot: AtlasSnapshot;
      /** Its canonical digest, recomputed in the browser. */
      readonly digest: string;
    }
  | {
      /** The pin was refused; nothing is shown from it. */
      readonly ok: false;
      /** Fixed explanation of the refusal. */
      readonly reason: string;
    };

/**
 * Admit a snapshot pin: schema, contract and recomputed digest must all hold.
 *
 * @param text - The pin's JSON text.
 * @returns The admitted snapshot with its digest, or the refusal reason.
 */
export async function admitAtlasSnapshot(text: string): Promise<AtlasAdmission> {
  let pin: unknown;
  try {
    pin = JSON.parse(text);
  } catch {
    return { ok: false, reason: "The Atlas snapshot pin is not JSON." };
  }
  if (
    typeof pin !== "object" ||
    pin === null ||
    (pin as Record<string, unknown>)["schema"] !== atlasSnapshotPinSchema
  ) {
    return { ok: false, reason: "The Atlas snapshot pin has an unknown schema." };
  }
  const record = pin as Record<string, unknown>;
  const parsed = parseAtlasSnapshot(record["snapshot"]);
  if (!parsed.ok)
    return {
      ok: false,
      reason: `The Atlas snapshot breaks the contract: ${parsed.code} at ${parsed.path}.`,
    };
  // The query owner reuses this digest for the snapshot's first and later queries.
  const digest = await rememberedAtlasSnapshotDigest(parsed.value);
  if (digest !== record["snapshot_digest"]) {
    return { ok: false, reason: "The Atlas snapshot does not match its recorded digest." };
  }
  return { ok: true, snapshot: parsed.value, digest };
}
