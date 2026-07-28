"""The stiffness sweep (M2 acceptance): where does this route die?

Parmar, Halm & Posa showed stiffness alone degrades *learning* of contact.
The same mechanism must bite *verification*: as k grows, the one-step map
steepens near the guard, so enclosures over any box touching the contact
boundary widen. This script measures that, on the spring-damper model, and
publishes the curve as-is -- the boundary of the method is a deliverable,
not a secret.

Three quantities per stiffness k, on the vy' component (where contact acts):

  enc_width     width of the interval twin's enclosure over the box
  mc_width      width of the true reachable range, estimated by 4x10^5
                Monte Carlo samples (corners + interior)
  sharpness     enc_width / mc_width  --  1.0 would be a perfect enclosure;
                growth in this ratio is *our* conservatism, growth in
                mc_width is the physics itself getting steeper

measured on two boxes:

  straddle      a box across the contact guard (y in [-5, +5] mm) -- the
                regime a certifier near the boundary must survive
  contact       a box strictly inside contact (y in [-10, -2] mm) -- the
                per-mode regime A2 buys us

plus point_width: the enclosure width at a single in-contact point, which
isolates pure rounding cost (should stay at ulp scale for all k).

Outputs: artifacts/stiffness_sweep.json and, if matplotlib is present,
artifacts/stiffness_sweep.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from certabstain import Interval
from certabstain.reference import SpringDamper2D

OUT = Path(__file__).resolve().parent / "artifacts"

STRADDLE = dict(
    lo=np.array([-0.01, -0.005, -0.5, -0.5]),
    hi=np.array([0.01, 0.005, 0.5, 0.5]),
)
CONTACT = dict(
    lo=np.array([-0.01, -0.010, -0.5, -0.5]),
    hi=np.array([0.01, -0.002, 0.5, 0.5]),
)
U_BOX = dict(lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]))
POINT = np.array([0.0, -0.005, 0.1, -0.2])
U_POINT = np.array([0.3, -0.4])


def _mc_width_vy(model: SpringDamper2D, box, ubox, n=400_000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    s = rng.uniform(box["lo"], box["hi"], size=(n, 4))
    u = rng.uniform(ubox["lo"], ubox["hi"], size=(n, 2))
    # corners matter for the extremes: mix in vertex sampling
    corners_s = np.array(
        np.meshgrid(*zip(box["lo"], box["hi"]), indexing="ij")
    ).reshape(4, -1).T
    corners_u = np.array(
        np.meshgrid(*zip(ubox["lo"], ubox["hi"]), indexing="ij")
    ).reshape(2, -1).T
    cs = np.repeat(corners_s, corners_u.shape[0], axis=0)
    cu = np.tile(corners_u, (corners_s.shape[0], 1))
    vy = np.concatenate(
        [model.step(s, u)[:, 3], model.step(cs, cu)[:, 3]]
    )
    return float(vy.max() - vy.min())


def run_sweep() -> dict:
    ks = np.logspace(1, 7, 13)
    rows = []
    for k in ks:
        model = SpringDamper2D(k=float(k))
        row = {"k": float(k)}
        for name, box in (("straddle", STRADDLE), ("contact", CONTACT)):
            S = Interval(box["lo"], box["hi"])
            U = Interval(U_BOX["lo"], U_BOX["hi"])
            enc = model.step_interval(S, U)
            enc_w = float(enc.width()[3])
            mc_w = _mc_width_vy(model, box, U_BOX)
            row[f"{name}_enc_width"] = enc_w
            row[f"{name}_mc_width"] = mc_w
            row[f"{name}_sharpness"] = enc_w / mc_w
        pt = model.step_interval(Interval(POINT, POINT), Interval(U_POINT, U_POINT))
        row["point_width"] = float(pt.width()[3])
        rows.append(row)
        print(
            f"k={k:9.3g}  straddle enc={row['straddle_enc_width']:9.4g} "
            f"sharp={row['straddle_sharpness']:6.3f}   "
            f"contact enc={row['contact_enc_width']:9.4g} "
            f"sharp={row['contact_sharpness']:6.3f}   "
            f"point={row['point_width']:8.2e}"
        )
    return {"model": "SpringDamper2D", "vy_component": 3, "rows": rows}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = run_sweep()
    (OUT / "stiffness_sweep.json").write_text(json.dumps(data, indent=2))
    print(f"\nwrote {OUT / 'stiffness_sweep.json'}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = data["rows"]
        ks = [r["k"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, style in (("straddle", "o-"), ("contact", "s-")):
            ax1.loglog(ks, [r[f"{name}_enc_width"] for r in rows], style, label=f"{name}: enclosure")
            ax1.loglog(ks, [r[f"{name}_mc_width"] for r in rows], style.replace("-", "--"), alpha=0.6, label=f"{name}: true (MC)")
            ax2.semilogx(ks, [r[f"{name}_sharpness"] for r in rows], style, label=name)
        ax1.set_xlabel("contact stiffness k")
        ax1.set_ylabel("width of vy' range")
        ax1.set_title("Enclosure vs. true range")
        ax1.legend(fontsize=8)
        ax2.set_xlabel("contact stiffness k")
        ax2.set_ylabel("enclosure / true width")
        ax2.set_title("Conservatism (1.0 = perfect)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "stiffness_sweep.png", dpi=140)
        print(f"wrote {OUT / 'stiffness_sweep.png'}")
    except ImportError:
        print("matplotlib not available; JSON only")


if __name__ == "__main__":
    main()
