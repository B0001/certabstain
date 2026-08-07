"""The tube-vs-stiffness sweep (M4 acceptance: tube-width vs K, kill-criterion).

stiffness_sweep.py (M2) showed enclosure width blowing up with contact
stiffness for a *single* step. The natural follow-on question for the tube
(M4) is what that does to a *K-step* certified reachable set: does the tube
stay usable, or does conservatism compound step over step until it engulfs
the whole operating envelope? This script measures that directly, on the
(y, vy) subsystem of SpringDamper2D (the same slice stiffness_sweep.py
already singles out as "where contact acts"; x, vx never enter these
equations, so nothing is lost by dropping them here).

For each stiffness k: train a small net for the (y, vy, uy) -> (y', vy')
map near that k's contact equilibrium, certify its discrepancy epsilon
(M3), and propagate the tube (M4) out to K_MAX steps or until it leaves
its own certified cover -- whichever comes first, reported honestly rather
than truncated silently (spec section 5's rule, applied to a time horizon).

Kill-criterion checkpoint (spec M4): if by K = 5 the tube has already left
its cover, or its vy-width has grown past a declared operational clearance
band, that stiffness is flagged. The spec's own text names this the signal
to stop and mitigate (shrink the domain, use a smaller net, shrink dt)
before any further milestone -- so it is reported here as data, not hidden.

Outputs: artifacts/tube_sweep.json and, if matplotlib is present,
artifacts/tube_sweep.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from certabstain import Interval, propagate_tube
from certabstain.discrepancy import certify_epsilon
from certabstain.nnbound import MLP, fit_mlp
from certabstain.provenance import write_provenance_sidecar
from certabstain.reference import SpringDamper2D

OUT = Path(__file__).resolve().parent / "artifacts"

K_MAX = 10
CLEARANCE_BAND = 0.15  # declared operational vy-clearance margin, for the kill check
KS = np.logspace(1, 6, 12)


def _yv_subsystem(model: SpringDamper2D):
    """(y, vy, uy) -> (y', vy'), the slice contact acts on. No x/vx coupling."""

    def step(yv: np.ndarray, uy: np.ndarray) -> np.ndarray:
        n = yv.shape[0]
        s = np.zeros((n, 4))
        s[:, 1] = yv[:, 0]
        s[:, 3] = yv[:, 1]
        u = np.zeros((n, 2))
        u[:, 1] = uy[:, 0]
        return model.step(s, u)[:, [1, 3]]

    def step_interval(lo: np.ndarray, hi: np.ndarray):
        n = lo.shape[0]
        zero = np.zeros(n)
        S = Interval(
            np.stack([zero, lo[:, 0], zero, lo[:, 1]], axis=1),
            np.stack([zero, hi[:, 0], zero, hi[:, 1]], axis=1),
        )
        U = Interval(
            np.stack([zero, lo[:, 2]], axis=1), np.stack([zero, hi[:, 2]], axis=1)
        )
        enc = model.step_interval(S, U)
        return enc.lo[:, [1, 3]], enc.hi[:, [1, 3]]

    return step, step_interval


def _run_one(k: float, seed: int = 7) -> dict:
    model = SpringDamper2D(k=k)
    y_eq = -model.m * model.g / k
    step, step_interval = _yv_subsystem(model)
    rng = np.random.default_rng(seed)

    span_y = max(0.3 * abs(y_eq), 0.01)  # certification domain around equilibrium
    domain = Interval(
        np.array([y_eq - span_y, -0.3, -0.2]), np.array([y_eq + span_y, 0.3, 0.2])
    )
    X = rng.uniform(domain.lo, domain.hi, size=(60_000, 3))
    Y = step(X[:, :2], X[:, 2:])
    net: MLP = fit_mlp((3, 8, 2), X, Y, steps=6_000, lr=2e-3, seed=1)

    cert = certify_epsilon(
        net,
        lambda lo, hi: step_interval(lo, hi),
        domain,
        reference_id=f"SpringDamper2D(k={k:g}) (y, vy) subsystem",
        ref_float=lambda p: step(p[:, :2], p[:, 2:]),
        target=None,
        max_leaf_evals=200_000,
        floor_samples=150_000,
    )

    X0 = Interval(
        np.array([y_eq - 0.2 * span_y, -0.02]), np.array([y_eq + 0.2 * span_y, 0.02])
    )
    U_box = Interval(np.array([-0.02]), np.array([0.02]))
    tube = propagate_tube(net, cert, X0, [U_box] * K_MAX, n_states=2)

    vy_widths = tube.widths[:, 1].tolist()
    step5 = min(5, tube.horizon)
    kill = tube.horizon < 5 or vy_widths[step5] > CLEARANCE_BAND

    return {
        "k": float(k),
        "y_eq": float(y_eq),
        "eps": cert.eps.tolist(),
        "empirical_floor": cert.empirical_floor.tolist(),
        "cover_fraction": float(cert.cover_fraction),
        "requested_horizon": K_MAX,
        "achieved_horizon": tube.horizon,
        "cover_exit_reason": tube.cover_exit_reason,
        "y_widths": tube.widths[:, 0].tolist(),
        "vy_widths": vy_widths,
        "kill_by_5": bool(kill),
    }


def run_sweep() -> dict:
    rows = []
    for k in KS:
        row = _run_one(float(k))
        rows.append(row)
        flag = " KILL" if row["kill_by_5"] else ""
        print(
            f"k={row['k']:10.3g}  y_eq={row['y_eq']:10.3e}  "
            f"eps=[{row['eps'][0]:.3g},{row['eps'][1]:.3g}]  "
            f"cover={row['cover_fraction']:.1%}  "
            f"horizon={row['achieved_horizon']:2d}/{K_MAX}  "
            f"vy_width@5={row['vy_widths'][min(5, row['achieved_horizon'])]:.3g}"
            f"{flag}"
        )
    return {
        "model": "SpringDamper2D (y, vy) subsystem",
        "K_MAX": K_MAX,
        "clearance_band": CLEARANCE_BAND,
        "rows": rows,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = run_sweep()
    (OUT / "tube_sweep.json").write_text(json.dumps(data, indent=2))
    write_provenance_sidecar(OUT / "tube_sweep.json", writer="tube_sweep.py")
    print(f"\nwrote {OUT / 'tube_sweep.json'}")

    n_killed = sum(r["kill_by_5"] for r in data["rows"])
    if n_killed:
        print(
            f"\nKILL-CRITERION: {n_killed}/{len(data['rows'])} stiffness values "
            f"failed to keep the tube inside its clearance band by K=5. Per spec "
            f"section 9/M4: mitigate (shrink D, smaller net, smaller dt) before "
            f"proceeding to M5, rather than proceeding on a tube that has already "
            f"gone vacuous at the low-stiffness end."
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = data["rows"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        cmap = plt.get_cmap("viridis")
        for i, r in enumerate(rows):
            color = cmap(i / max(1, len(rows) - 1))
            ks_axis = list(range(len(r["vy_widths"])))
            ax1.semilogy(
                ks_axis, r["vy_widths"], "o-", color=color, ms=3,
                label=f"k={r['k']:.2g}" if i % 2 == 0 else None,
            )
        ax1.axhline(CLEARANCE_BAND, color="red", ls="--", lw=1, label="clearance band")
        ax1.set_xlabel("tube step K")
        ax1.set_ylabel("vy tube width")
        ax1.set_title("Tube width vs K, by stiffness")
        ax1.legend(fontsize=7, ncol=2)

        ax2.semilogx(
            [r["k"] for r in rows], [r["achieved_horizon"] for r in rows], "o-"
        )
        ax2.axhline(5, color="red", ls="--", lw=1, label="K=5 kill checkpoint")
        ax2.set_xlabel("contact stiffness k")
        ax2.set_ylabel("certified horizon reached")
        ax2.set_title("Certified horizon vs stiffness")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT / "tube_sweep.png", dpi=140)
        print(f"wrote {OUT / 'tube_sweep.png'}")
    except ImportError:
        print("matplotlib not available; JSON only")


if __name__ == "__main__":
    main()
