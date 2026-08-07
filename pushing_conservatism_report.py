"""M6 acceptance: the conservatism report across PusherSlider's three modes.

Consolidates the three per-mode artifacts (pushing_stick_report.json,
pushing_slide_left_report.json, pushing_slide_right_report.json -- each
produced by certifying the mode-dependent one-step (py, vpx, vpy) -> py2
subsystem, per spec section 6's "per-mode certification unioned over the
three modes") into the single comparison spec section 6 asks for: certified
epsilon vs. empirical gap vs. abstention rate, side by side.

Spec's accept bar is explicit: the sticking mode must reach nominal
abstention <= 2*alpha; the other two modes get an honest write-up whether
they clear it or not. All three happened to clear it here (see the per-mode
notes fields for how each was arrived at) -- that is reported plainly, not
treated as a foregone conclusion.

Outputs: artifacts/pushing_conservatism_report.json and, if matplotlib is
present, artifacts/pushing_conservatism_report.png.
"""

from __future__ import annotations

import json
from pathlib import Path

from certabstain.provenance import write_provenance_sidecar

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODES = ("stick", "slide_left", "slide_right")


def _normalize(mode: str) -> dict:
    """The three per-mode reports (written independently) use two slightly
    different JSON shapes; flatten both to one comparison-ready row."""
    raw = json.loads((ARTIFACTS / f"pushing_{mode}_report.json").read_text())

    if "certifier" in raw:  # slide_left / slide_right shape
        cert = raw["certifier"]
        mon = raw["monitor"]
        eps = cert["eps"]
        floor = cert["empirical_floor"]
        ratio = cert["eps_to_floor_ratio"]
        cover = cert["cover_fraction"]
        alpha = mon["alpha"]
        abstention = mon["measured_abstention_rate"]
        bar = mon["two_alpha_bar"]
    else:  # stick shape
        eps = raw["eps"]
        floor = raw["empirical_floor"]
        ratio = raw["eps_over_floor"]
        cover = raw["cover_fraction"]
        alpha = raw["alpha"]
        abstention = raw["abstention_rate"]
        bar = raw["twice_alpha"]

    return {
        "mode": mode,
        "eps": eps,
        "empirical_floor": floor,
        "eps_to_floor_ratio": ratio,
        "cover_fraction": cover,
        "alpha": alpha,
        "measured_abstention_rate": abstention,
        "two_alpha_bar": bar,
        "meets_M6_bar": bool(raw["verdict"]["meets_M6_bar"]),
        "verdict_reason": raw["verdict"]["reason"],
    }


def build_report() -> dict:
    rows = [_normalize(m) for m in MODES]
    n_pass = sum(r["meets_M6_bar"] for r in rows)
    return {
        "milestone": "M6",
        "accept_criterion": (
            "certified two-sided monitor on at least the sticking mode with "
            "nominal abstention <= 2x the conformal alpha; honest write-up of "
            "any mode that fails to certify and why"
        ),
        "rows": rows,
        "modes_meeting_bar": n_pass,
        "modes_total": len(rows),
        "stick_meets_required_bar": next(r for r in rows if r["mode"] == "stick")["meets_M6_bar"],
    }


def print_table(data: dict) -> None:
    print(f"{'mode':<12}{'eps':>10}{'eps/floor':>11}{'cover':>8}{'abstain':>10}{'2*alpha':>9}{'verdict':>10}")
    for r in data["rows"]:
        tag = "PASS" if r["meets_M6_bar"] else "FAIL"
        required = " (required)" if r["mode"] == "stick" else ""
        print(
            f"{r['mode']:<12}{r['eps']:>10.4g}{r['eps_to_floor_ratio']:>10.2f}x"
            f"{r['cover_fraction']:>8.1%}{r['measured_abstention_rate']:>10.4f}"
            f"{r['two_alpha_bar']:>9.3f}{tag:>10}{required}"
        )
    print(
        f"\n{data['modes_meeting_bar']}/{data['modes_total']} modes meet the "
        f"2*alpha abstention bar. Required mode (stick): "
        f"{'MET' if data['stick_meets_required_bar'] else 'NOT MET'}."
    )


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    data = build_report()
    print_table(data)
    (ARTIFACTS / "pushing_conservatism_report.json").write_text(json.dumps(data, indent=2))
    write_provenance_sidecar(
        ARTIFACTS / "pushing_conservatism_report.json",
        writer="pushing_conservatism_report.py",
    )
    print(f"\nwrote {ARTIFACTS / 'pushing_conservatism_report.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = data["rows"]
        modes = [r["mode"] for r in rows]
        x = range(len(modes))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

        ax1.bar(x, [r["eps"] for r in rows], color="steelblue", label="certified eps")
        ax1.bar(
            x, [r["empirical_floor"] for r in rows], color="orange", alpha=0.7,
            width=0.5, label="empirical floor",
        )
        ax1.set_xticks(list(x), modes)
        ax1.set_ylabel("value (units of py)")
        ax1.set_title("Certified eps vs. empirical gap, per mode")
        ax1.legend(fontsize=8)

        ax2.bar(x, [r["measured_abstention_rate"] for r in rows], color="seagreen")
        bar = rows[0]["two_alpha_bar"]
        ax2.axhline(bar, color="red", ls="--", lw=1, label="2*alpha bar")
        ax2.set_xticks(list(x), modes)
        ax2.set_ylabel("nominal abstention rate")
        ax2.set_title("Abstention rate vs. the M6 bar, per mode")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(ARTIFACTS / "pushing_conservatism_report.png", dpi=140)
        print(f"wrote {ARTIFACTS / 'pushing_conservatism_report.png'}")
    except ImportError:
        print("matplotlib not available; JSON only")


if __name__ == "__main__":
    main()
