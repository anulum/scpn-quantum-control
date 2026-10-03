# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real workbench accessibility acceptance
"""Check the original route controls, graph values and locked audit engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from playwright.sync_api import Page

AXE_VERSION = "4.13.0"
AXE_SHA256 = "c24f097bd2f451d4f933e8bc7d8d539f8672a2ebcb5cc9f9f3eec8ca9470a0c1"
REPOSITORY = Path(__file__).resolve().parents[1]


def read_auditor(path: Path | None = None) -> str:
    """Read the admitted axe-core bytes without installing or replacing a tool.

    Parameters
    ----------
    path
        Explicit existing asset, or the frozen Studio dependency when omitted.

    Returns
    -------
    str
        Exact UTF-8 auditor whose content digest matches the locked version.

    Raises
    ------
    OSError
        The required existing dependency is unavailable.
    ValueError
        The proposed auditor differs from the admitted source bytes.

    """
    source = path or REPOSITORY / "studio-web/node_modules/axe-core/axe.min.js"
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != AXE_SHA256:
        raise ValueError("Accessibility auditor differs from the locked axe-core source")
    return content.decode("utf-8")


def audit_page(page: Page, auditor: str, state: str, reports: list[dict[str, object]]) -> None:
    """Retain the full actual audit and refuse every serious or critical finding.

    Parameters
    ----------
    page
        Real production route in the owned isolated browser context.
    auditor
        Already content-verified axe-core source; no rules are disabled.
    state
        Explicit route, theme and data-state identity for the retained result.
    reports
        Caller-owned evidence including passes, incompletes and all violations.

    Raises
    ------
    AssertionError
        The loaded engine version or observed serious/critical findings fail.

    """
    page.add_script_tag(content=auditor)
    assert page.evaluate("axe.version") == AXE_VERSION
    report = cast(dict[str, object], page.evaluate("async () => await axe.run(document)"))
    reports.append({"state": state, "report": report})
    violations = cast(list[dict[str, object]], report["violations"])
    failures = [row for row in violations if row["impact"] in ("serious", "critical")]
    assert not failures, [(row["id"], row["nodes"]) for row in failures]


def keyboard_route(page: Page) -> dict[str, object]:
    """Traverse actual enabled controls with Tab and check native modal recovery.

    Parameters
    ----------
    page
        Loaded current production route with its original native controls.

    Returns
    -------
    dict[str, object]
        Real control count, focus traversal and native dialog close observations.

    """
    from playwright.sync_api import expect

    count = cast(
        int,
        page.evaluate("""() => {
      let index = 0;
      for (const element of document.querySelectorAll('a[href],button,input,select,textarea,[tabindex]')) {
        if (element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length) {
          element.dataset.accessibilityControl = String(index++);
        }
      }
      return index;
    }"""),
    )
    origin = page.get_by_role("button", name="Skip to current view", exact=True)
    origin.focus()
    seen: set[str] = set()
    # Chromium also tabs to implicit scrolling regions without a tabindex attribute.
    for _ in range(2 * count + 2):
        identity = cast(
            str | None,
            page.evaluate("document.activeElement.dataset.accessibilityControl ?? null"),
        )
        if identity is not None:
            seen.add(identity)
        if len(seen) == count:
            break
        page.keyboard.press("Tab")
    assert len(seen) == count, {"expected": count, "reached": sorted(seen)}
    help_control = page.get_by_role("button", name="Keyboard help", exact=True)
    help_control.focus()
    page.keyboard.press("Enter")
    dialog = page.get_by_role("dialog", name="Keyboard help", exact=True)
    expect(dialog).to_be_visible()
    close = dialog.get_by_role("button", name="Close keyboard help")
    expect(close).to_be_focused()
    page.keyboard.press("Tab")
    expect(close).to_be_focused()
    page.keyboard.press("Shift+Tab")
    expect(close).to_be_focused()
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    expect(help_control).to_be_focused()
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")
    expect(help_control).to_be_focused()
    origin.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("region", name="Workbench view", exact=True)).to_be_focused()
    return {
        "enabled_controls": count,
        "tab_reached": len(seen),
        "native_modal_trap_and_origin": True,
    }


def chart_table_identity(page: Page) -> dict[str, object]:
    """Compare every rendered sample with the plot and original Python reference.

    Parameters
    ----------
    page
        Genuine built Workspace route with the original two WASM instruments.

    Returns
    -------
    dict[str, object]
        Complete sample count, original numeric boundary and exact cross-view identity.

    """
    from playwright.sync_api import expect

    fixture = json.loads(
        (REPOSITORY / "data/studio/kuramoto_scenario_meanfield_20260708.json").read_text()
    )
    from scpn_quantum_control.studio.kuramoto_reference import simulate

    # The live instruments already use dt=.05; the committed replay fixture uses dt=.02.
    # Exercise the unchanged live request with its independent original reference.
    expected = simulate(
        "mean-field",
        fixture["scenario"]["omega"],
        fixture["scenario"]["theta0"],
        steps=300,
        dt=0.05,
        coupling=2.5,
    ).order_parameter.tolist()
    samples: list[list[str]] = []
    phase_samples: list[list[str]] = []
    for label, captured in (
        ("Order parameter data", samples),
        ("Phase trajectory data", phase_samples),
    ):
        region = page.get_by_role("region", name=label + " samples", exact=True)
        table = region.get_by_role("table", name=label, exact=True)
        while True:
            rows = cast(
                list[list[str]],
                table.locator("tbody tr").evaluate_all(
                    "rows => rows.map(row => [...row.cells].map(cell => cell.textContent))"
                ),
            )
            assert 0 < len(rows) <= 20, (label, "mounted rows", len(rows))
            captured.extend(rows)
            next_page = region.get_by_role("button", name="Next samples")
            if next_page.is_disabled():
                break
            next_page.focus()
            page.keyboard.press("Enter")
            expect(table.locator("tbody tr").first.locator("th")).to_have_text(str(len(captured)))
    assert [int(row[0]) for row in samples] == list(range(len(expected))), {
        "steps": [row[0] for row in samples],
        "expected_count": len(expected),
    }
    assert [row[:2] for row in phase_samples] == samples, "Original Play and Lab samples differ"
    values = [float(row[1]) for row in samples]
    deviation = max(
        abs(actual - declared) for actual, declared in zip(values, expected, strict=True)
    )
    assert deviation < 1e-10, (
        "Original live-request Python reference deviation",
        deviation,
        "first",
        values[:2],
        "last",
        values[-2:],
    )
    assert len(phase_samples[0]) == fixture["scenario"]["n"] + 2, (
        "Phase oscillator identities differ"
    )
    points = (
        page.get_by_role("img", name="order parameter over time")
        .locator("polyline")
        .get_attribute("points")
    )
    assert points is not None
    coordinates = [tuple(map(float, point.split(","))) for point in points.split()]
    assert len(coordinates) == len(samples), (
        "Plot points",
        len(coordinates),
        "Samples",
        len(samples),
    )
    for step, ((x, y), sample) in enumerate(zip(coordinates, values, strict=True)):
        assert abs(x - step) <= 0.0050001, ("x", step, x)
        assert abs(y - (80 - sample * 80)) <= 0.0050001, ("y", step, y, sample)
    expect(
        page.get_by_role("img", name="phase-space cylinder: oscillator phases over time")
    ).to_be_visible()
    expect(
        page.get_by_role("img", name="Bloch sphere equator: final phases as spin-coherent points")
    ).to_be_visible()
    return {
        "samples": len(samples),
        "phase_columns": len(phase_samples[0]) - 2,
        "play_lab_raw_values_exact": True,
        "independent_reference_tolerance": "1e-10",
        "original_live_dt": 0.05,
        "committed_replay_dt": fixture["scenario"]["dt"],
        "mounted_rows_per_page_max": 20,
    }


def graph_table_identity(page: Page, source: str) -> dict[str, object]:
    """Read the original signed matrix through its public native archive preview.

    Parameters
    ----------
    page
        Existing isolated page; no user data or public screenshot is recorded.
    source
        Owned original source origin with the pre-existing explicit conformance host.

    Returns
    -------
    dict[str, object]
        Independent directed-edge oracle and unchanged original archive identity.

    """
    from playwright.sync_api import expect

    page.goto(source + "browser-tests/parameterEditor.html", wait_until="networkidle")
    corpus = (REPOSITORY / "tests/data/studio_workspace/documents.json").read_text()
    native = cast(
        dict[str, object],
        page.evaluate(
            "async corpus => (await import('/browser-tests/parameterEditor.native.tsx')).createParameterConformanceArchive(corpus)",
            corpus,
        ),
    )
    original = cast(str, native["json"])
    workspace = page.get_by_role("region", name="Local workspace")
    archive_input = workspace.get_by_label("Workspace archive JSON")
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
    archive_input.fill(original)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    editor = workspace.get_by_role("region", name="Linked parameter editor")
    table = editor.get_by_role("table", name="K_nm coupling edges")
    expect(table).to_be_visible()
    expected = [["1", "0", "-2"], ["0", "1", "5"], ["2", "1", "3"], ["0", "2", "-4"]]
    actual = table.locator("tbody tr").evaluate_all(
        "rows => rows.map(row => [...row.cells].slice(0,3).map(cell => cell.textContent))"
    )
    assert actual == expected
    titles = (
        editor.get_by_role("img", name="K_nm coupling graph diagram")
        .locator(":scope > path > title")
        .all_text_contents()
    )
    assert titles == [f"{row[0]} → {row[1]}: {row[2]} rad/s" for row in expected]
    selection = table.get_by_role("button", name="Edge 1 → 0: -2 rad/s", exact=True)
    selection.focus()
    page.keyboard.press("Enter")
    expect(selection).to_have_attribute("aria-pressed", "true")
    expect(editor.get_by_role("button", name="K_nm[0,1] = -2", exact=True)).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(editor.get_by_label("Selected value", exact=True)).to_have_value("-2")
    expect(archive_input).to_have_value(original)
    return {
        "edges": expected,
        "original_archive_unchanged": True,
        "workspace_writes": 0,
        "boundary": "Original explicit synthetic metadata conformance host; no numerical or provider claim",
    }


def populated_states(
    page: Page, auditor: str, theme: str, reports: list[dict[str, object]]
) -> dict[str, object]:
    """Audit native compiler tables, located refusals and an empty result filter.

    Parameters
    ----------
    page
        Owned original built workbench, with no user workspace imported.
    auditor
        Exact admitted axe-core source.
    theme
        Current native browser colour scheme for the retained state identity.
    reports
        Caller-owned complete audit reports, including a failed assertion.

    Returns
    -------
    dict[str, object]
        Native compiled-control traversal and explicit refused/empty observations.

    """
    from playwright.sync_api import expect

    page.evaluate("location.hash = '#/build'")
    editor = page.get_by_role("region", name="Program authoring", exact=True)
    compile_control = editor.get_by_role("button", name="Compile source", exact=True)
    compile_control.focus()
    page.keyboard.press("Enter")
    expect(editor.get_by_role("table", name="Program IR", exact=True)).to_be_visible()
    trace = page.get_by_role("region", name="Compiler trace inspector", exact=True)
    trace.get_by_role("button", name="Open native example", exact=True).focus()
    page.keyboard.press("Enter")
    expect(trace.get_by_role("table", name="Qubit mapping", exact=True)).to_be_visible()
    editor.get_by_text("Exact emitted record", exact=True).focus()
    page.keyboard.press("Enter")
    for name in (
        "Program IR table scrolling",
        "Exact emitted record scrolling",
        "Qubit mapping table scrolling",
        "Gate changes table scrolling",
    ):
        region = page.get_by_role("region", name=name, exact=True)
        region.focus()
        expect(region).to_be_focused()
    keyboard = keyboard_route(page)
    page.evaluate("document.documentElement.style.zoom = '200%'")
    audit_page(page, auditor, f"{theme}:native-compiled-tables:css-zoom-200", reports)
    source = editor.get_by_label("Program source", exact=True)
    source.fill(source.input_value() + "mystery q[0];\n")
    compile_control.focus()
    page.keyboard.press("Enter")
    expect(editor.get_by_role("alert")).to_contain_text("unsupported_operation")
    located = editor.get_by_role("region", name="Located source diagnostic scrolling")
    located.focus()
    expect(located).to_be_focused()
    audit_page(page, auditor, f"{theme}:native-located-refusal:css-zoom-200", reports)
    page.evaluate("document.documentElement.style.zoom = ''")
    page.evaluate("location.hash = '#/results'")
    operation = page.get_by_label("Operation", exact=True)
    operation.fill("no-declared-operation-matches-this-filter")
    expect(page.get_by_text("No committed support row matches these filters.")).to_be_visible()
    audit_page(page, auditor, f"{theme}:empty-result-filter", reports)
    operation.fill("")
    return {"compiled_keyboard": keyboard, "located_refusal": True, "empty_filter": True}
