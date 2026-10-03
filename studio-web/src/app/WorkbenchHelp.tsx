// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native workbench keyboard assistance

import { useEffect, useId, useRef } from "react";

/** Identity of the current route, used to close assistance when navigation changes. */
export interface WorkbenchHelpProps {
  /** Original admitted or refused route identity; no navigation authority. */
  readonly routeKey: string;
}

/** Explain native keyboard actions in a modal that restores the invoking control's focus. */
export function WorkbenchHelp({ routeKey }: WorkbenchHelpProps) {
  const dialog = useRef<HTMLDialogElement>(null);
  const origin = useRef<HTMLButtonElement>(null);
  const closeControl = useRef<HTMLButtonElement>(null);
  const restoreFocus = useRef(true);
  const titleId = useId();
  useEffect(() => {
    restoreFocus.current = false;
    if (dialog.current!.open) dialog.current!.close();
  }, [routeKey]);
  return <>
    <button type="button" ref={origin} onClick={() => {
      restoreFocus.current = true;
      dialog.current!.showModal();
    }}>Keyboard help</button>
    <dialog className="qsp-keyboard-help" ref={dialog} aria-labelledby={titleId} onKeyDown={event => {
      // This help dialog has one control. Keep Tab on it instead of Chromium's body sentinel.
      if (event.key === "Tab") { event.preventDefault(); closeControl.current!.focus(); }
    }} onClose={() => {
      // Native close already restores focus; a queued event must preserve a later action.
      if (restoreFocus.current && !dialog.current!.open
          && (document.activeElement === document.body || dialog.current!.contains(document.activeElement))) {
        origin.current!.focus();
      }
    }}>
      <h3 id={titleId}>Keyboard help</h3>
      <p>Use Tab and Shift+Tab to move between controls, Enter to follow links or activate a button, and Space to activate buttons and checkboxes.</p>
      <p>Use the arrow keys on sliders and selects. Navigation focuses the current view; Skip to current view bypasses the workbench navigation.</p>
      <p>Data tables contain the same source values as the charts. Focus a table's scrolling region to scroll horizontally, and use the sample page buttons to reach every snapshot.</p>
      <p>Claim status and verification are written as text. A local match does not certify a scientific claim. Imported workspace and evidence values stay local to this browser.</p>
      <p>Escape or Close keyboard help returns to the control that opened this dialog.</p>
      <button type="button" ref={closeControl} onClick={() => dialog.current!.close()}>Close keyboard help</button>
    </dialog>
  </>;
}
