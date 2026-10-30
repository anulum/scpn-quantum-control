// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — colour set chosen by the admitted display setting

import { settingsFromArchive } from "../shared/contracts/settings";
import type { WorkspaceArchivePreview } from "../shared/storage/workspaceArchive";

/** Colour set the workbench can be forced into; absent means the system preference. */
export type DisplayTheme = "light" | "dark";

/**
 * Read the display-only `theme` from an admitted archive's resolved settings.
 *
 * The setting is a display field: it never enters a numerical digest and this read
 * resolves no policy. Records that disagree, or a value other than `light` or
 * `dark`, leave the choice to the system preference.
 */
export function displayTheme(preview: WorkspaceArchivePreview | null): DisplayTheme | null {
  if (preview === null) return null;
  const settings = settingsFromArchive(preview);
  if (!settings.ok) return null;
  const themes = new Set(
    settings.value.map(({ document }) => {
      const effective = document.body["effective"] as Readonly<Record<string, unknown>>;
      return effective["theme"];
    }),
  );
  if (themes.size !== 1) return null;
  const [theme] = themes;
  return theme === "light" || theme === "dark" ? theme : null;
}
