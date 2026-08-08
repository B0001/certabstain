"""Consistency check: pushing_conservatism_report.json vs. its three sources.

TECHNICAL_NOTE.md section 5 describes pushing_conservatism_report.py as a
pure consolidation of pushing_stick_report.json, pushing_slide_left_report.json
and pushing_slide_right_report.json -- it "only reads and consolidates those
three files," with no independent computation of eps/floor/abstention. If any
of the three per-mode reports is regenerated (or reverted) without re-running
the consolidator, the consolidated file goes stale relative to its own stated
inputs while still parsing as valid JSON, which is exactly the failure mode
certabstain-n7z.2 found (the committed pushing_conservatism_report.json once
carried the *pre-09984b8* stick/slide_left numbers after those two artifacts
had already been reverted back to matching bytes -- stale-vs-stale, not
stale-vs-current, but stale either way).

This test does not regenerate anything (no CERTABSTAIN_WRITE_ARTIFACTS
dependency): it recomputes build_report() from the three currently-committed
per-mode JSONs in-process and asserts the result equals the currently
committed pushing_conservatism_report.json field-by-field. It is therefore a
pure read-four-files-and-diff check, safe to run in any environment.
"""

from __future__ import annotations

import json
from pathlib import Path

from certabstain.pushing_conservatism_report import build_report

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def test_conservatism_report_matches_its_three_source_artifacts() -> None:
    committed = json.loads((ARTIFACTS / "pushing_conservatism_report.json").read_text())
    recomputed = build_report()

    assert recomputed == committed, (
        "artifacts/pushing_conservatism_report.json does not match a fresh "
        "consolidation of the three per-mode reports it claims to summarize "
        "(pushing_stick_report.json, pushing_slide_left_report.json, "
        "pushing_slide_right_report.json). Re-run `python "
        "pushing_conservatism_report.py` with CERTABSTAIN_WRITE_ARTIFACTS=1 "
        "to regenerate it from the currently-committed per-mode reports."
    )


def test_conservatism_report_rows_match_field_by_field() -> None:
    committed = json.loads((ARTIFACTS / "pushing_conservatism_report.json").read_text())
    recomputed = build_report()

    committed_rows = {r["mode"]: r for r in committed["rows"]}
    recomputed_rows = {r["mode"]: r for r in recomputed["rows"]}

    assert set(committed_rows) == set(recomputed_rows) == {"stick", "slide_left", "slide_right"}

    for mode in ("stick", "slide_left", "slide_right"):
        want = recomputed_rows[mode]
        got = committed_rows[mode]
        for field in want:
            assert got.get(field) == want[field], (
                f"mode={mode!r} field={field!r}: committed "
                f"pushing_conservatism_report.json has {got.get(field)!r}, but "
                f"the currently-committed pushing_{mode}_report.json backs "
                f"{want[field]!r}"
            )
