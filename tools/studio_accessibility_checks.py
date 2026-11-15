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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

AXE_VERSION = "4.14.0"
AXE_SHA256 = "20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878"
REPOSITORY = Path(__file__).resolve().parents[1]
_MARK_CONTROLS = """() => {
      let index = 0;
      for (const element of document.querySelectorAll('a[href],button,input,select,textarea,[tabindex]')) {
        if (element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length) {
          element.dataset.accessibilityControl = String(index++);
        }
      }
      return index;
    }"""


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

    # CSS reveals command results only while their navigation has focus.
    page.get_by_label("Search commands", exact=True).focus()
    count = cast(int, page.evaluate(_MARK_CONTROLS))
    origin = page.get_by_role("button", name="Skip to current view", exact=True)
    origin.focus()
    seen = _tab_through(page, count)
    _check_help_dialog(page)
    origin.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("region", name="Workbench view", exact=True)).to_be_focused()
    return {
        "enabled_controls": count,
        "tab_reached": len(seen),
        "native_modal_trap_and_origin": True,
    }


def _tab_through(page: Page, count: int) -> set[str]:
    """Press Tab from the skip link until every marked control has had focus.

    Parameters
    ----------
    page
        Loaded route with its enabled controls marked.
    count
        Number of marked controls.

    Returns
    -------
    set[str]
        Marked control identities that received focus.

    """
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
    return seen


def _check_help_dialog(page: Page) -> None:
    """Trap focus in the keyboard help dialog, close it by Escape and reopen-close it.

    Parameters
    ----------
    page
        Loaded route with its keyboard help control.

    """
    from playwright.sync_api import expect

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


def _live_reference(fixture: dict[str, Any]) -> list[float]:
    """Compute the independent Python order parameter of the live Play request.

    Parameters
    ----------
    fixture
        Committed mean-field scenario.

    Returns
    -------
    list[float]
        Order parameter at every step.

    """
    from scpn_quantum_control.studio.kuramoto_reference import simulate

    # The live instruments already use dt=.05; the committed replay fixture uses dt=.02.
    # Exercise the unchanged live request with its independent original reference.
    return cast(
        list[float],
        simulate(
            "mean-field",
            fixture["scenario"]["omega"],
            fixture["scenario"]["theta0"],
            steps=300,
            dt=0.05,
            coupling=2.5,
        ).order_parameter.tolist(),
    )


def _read_samples(page: Page, label: str) -> list[list[str]]:
    """Read every row of a paged sample table, advancing by keyboard.

    Parameters
    ----------
    page
        Workspace route with the sample tables.
    label
        Table name.

    Returns
    -------
    list[list[str]]
        Cell text of every row in order.

    """
    from playwright.sync_api import expect

    captured: list[list[str]] = []
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
            return captured
        next_page.focus()
        page.keyboard.press("Enter")
        expect(table.locator("tbody tr").first.locator("th")).to_have_text(str(len(captured)))


def _check_samples(
    samples: list[list[str]],
    phase_samples: list[list[str]],
    expected: list[float],
    oscillators: int,
) -> list[float]:
    """Match both tables to each other and the order parameter to the Python reference.

    Parameters
    ----------
    samples
        Order parameter table rows.
    phase_samples
        Phase trajectory table rows.
    expected
        Independent Python order parameter.
    oscillators
        Number of oscillators in the scenario.

    Returns
    -------
    list[float]
        Order parameter values read from the table.

    """
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
    assert len(phase_samples[0]) == oscillators + 2, "Phase oscillator identities differ"
    return values


def _check_plot(page: Page, values: list[float]) -> None:
    """Place every plotted point on its sample and find the other two views visible.

    Parameters
    ----------
    page
        Workspace route with the plots.
    values
        Order parameter values read from the table.

    """
    from playwright.sync_api import expect

    points = (
        page.get_by_role("img", name="order parameter over time")
        .locator("polyline")
        .get_attribute("points")
    )
    assert points is not None
    coordinates = [tuple(map(float, point.split(","))) for point in points.split()]
    assert len(coordinates) == len(values), (
        "Plot points",
        len(coordinates),
        "Samples",
        len(values),
    )
    for step, ((x, y), sample) in enumerate(zip(coordinates, values, strict=True)):
        assert abs(x - step) <= 0.0050001, ("x", step, x)
        assert abs(y - (80 - sample * 80)) <= 0.0050001, ("y", step, y, sample)
    cylinder = "phase-space cylinder: oscillator phases over time"
    expect(page.get_by_role("img", name=cylinder)).to_be_visible()
    bloch = "Bloch sphere equator: final phases as spin-coherent points"
    expect(page.get_by_role("img", name=bloch)).to_be_visible()


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
    fixture = json.loads(
        (REPOSITORY / "data/studio/kuramoto_scenario_meanfield_20260708.json").read_text()
    )
    expected = _live_reference(fixture)
    samples = _read_samples(page, "Order parameter data")
    phase_samples = _read_samples(page, "Phase trajectory data")
    values = _check_samples(samples, phase_samples, expected, fixture["scenario"]["n"])
    _check_plot(page, values)
    return {
        "samples": len(samples),
        "phase_columns": len(phase_samples[0]) - 2,
        "play_lab_raw_values_exact": True,
        "independent_reference_tolerance": "1e-10",
        "original_live_dt": 0.05,
        "committed_replay_dt": fixture["scenario"]["dt"],
        "mounted_rows_per_page_max": 20,
    }


_EDGES = [["1", "0", "-2"], ["0", "1", "5"], ["2", "1", "3"], ["0", "2", "-4"]]
_COMPILED_REGIONS = (
    "Program IR table scrolling",
    "Exact emitted record scrolling",
    "Qubit mapping table scrolling",
    "Gate changes table scrolling",
)


def _admit_conformance_archive(page: Page, source: str) -> tuple[Locator, str]:
    """Build the native conformance archive and preview it in the workspace.

    Parameters
    ----------
    page
        Existing isolated page.
    source
        Owned original source origin.

    Returns
    -------
    tuple[Locator, str]
        The workspace archive editor and the admitted archive text.

    """
    from playwright.sync_api import expect

    page.goto(source + "browser-tests/parameterEditor.html", wait_until="networkidle")
    corpus = (REPOSITORY / "tests/data/studio_workspace/documents.json").read_text()
    native = cast(
        dict[str, object],
        page.evaluate(
            "async corpus => (await import('/browser-tests/parameterEditor.native.tsx'))"
            ".createParameterConformanceArchive(corpus)",
            corpus,
        ),
    )
    original = cast(str, native["json"])
    workspace = page.get_by_role("region", name="Local workspace")
    archive_input = workspace.get_by_label("Workspace archive JSON")
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
    archive_input.fill(original)
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    return archive_input, original


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

    archive_input, original = _admit_conformance_archive(page, source)
    editor = page.get_by_role("region", name="Linked parameter editor")
    table = editor.get_by_role("table", name="K_nm coupling edges")
    expect(table).to_be_visible()
    actual = table.locator("tbody tr").evaluate_all(
        "rows => rows.map(row => [...row.cells].slice(0,3).map(cell => cell.textContent))"
    )
    assert actual == _EDGES
    diagram = editor.get_by_role("img", name="K_nm coupling graph diagram")
    titles = diagram.locator(":scope > path > title").all_text_contents()
    assert titles == [f"{row[0]} → {row[1]}: {row[2]} rad/s" for row in _EDGES]
    selection = table.get_by_role("button", name="Edge 1 → 0: -2 rad/s", exact=True)
    selection.focus()
    page.keyboard.press("Enter")
    expect(selection).to_have_attribute("aria-pressed", "true")
    element = editor.get_by_role("button", name="K_nm[0,1] = -2", exact=True)
    expect(element).to_have_attribute("aria-pressed", "true")
    expect(editor.get_by_label("Selected value", exact=True)).to_have_value("-2")
    expect(archive_input).to_have_value(original)
    return {
        "edges": _EDGES,
        "original_archive_unchanged": True,
        "workspace_writes": 0,
        "boundary": "Original explicit synthetic metadata conformance host; no numerical or provider claim",
    }


def _open_compiled_tables(page: Page, editor: Locator) -> dict[str, object]:
    """Compile by keyboard, open the native trace and focus every scrolling region.

    Parameters
    ----------
    page
        Owned built workbench on the Build route.
    editor
        Program authoring region.

    Returns
    -------
    dict[str, object]
        Keyboard traversal of the compiled route.

    """
    from playwright.sync_api import expect

    editor.get_by_role("button", name="Compile source", exact=True).focus()
    page.keyboard.press("Enter")
    expect(editor.get_by_role("table", name="Program IR", exact=True)).to_be_visible()
    trace = page.get_by_role("region", name="Compiler trace inspector", exact=True)
    trace.get_by_role("button", name="Open native example", exact=True).focus()
    page.keyboard.press("Enter")
    expect(trace.get_by_role("table", name="Qubit mapping", exact=True)).to_be_visible()
    editor.get_by_text("Exact emitted record", exact=True).focus()
    page.keyboard.press("Enter")
    for name in _COMPILED_REGIONS:
        region = page.get_by_role("region", name=name, exact=True)
        region.focus()
        expect(region).to_be_focused()
    return keyboard_route(page)


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
    keyboard = _open_compiled_tables(page, editor)
    page.evaluate("document.documentElement.style.zoom = '200%'")
    audit_page(page, auditor, f"{theme}:native-compiled-tables:css-zoom-200", reports)
    source = editor.get_by_label("Program source", exact=True)
    source.fill(source.input_value() + "mystery q[0];\n")
    editor.get_by_role("button", name="Compile source", exact=True).focus()
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
