"""M6 acceptance: input-dimension and horizon scaling studies.

Two separate questions, both explicitly requested by spec section 6 (M6):

**Input-dimension scaling.** Branch-and-bound splits one axis at a time; the
same leaf-evaluation budget buys an exponentially coarser per-leaf width as
input dimension grows, which loosens the certified epsilon even when the
function itself is unchanged. This uses the exact construction M4's unit
tests already validated (a linear contraction f(x) = lam*x + b composed with
a bias-shifted twin of itself, so the true sup-gap is exactly ``lam*width +
delta`` in closed form -- no training noise, purely a BnB-geometry question)
and scales the input dimension from 2 to 8 (spec's own v1 cap, section 9)
at a FIXED leaf-eval budget, reporting how far the certified epsilon drifts
from the true gap.

**Horizon scaling.** M4's SpringDamper2D acceptance test achieved K=10 with
zero tube escapes. This asks the natural follow-on: how much further does
that same tube survive before it leaves its own certified cover? Reuses the
exact net/cert/domain from test_tube.py's acceptance test and propagates
further, reporting the width curve and the actual horizon where cover_exit
occurs (spec section 5: report the shrinkage, don't hide it).

Outputs: artifacts/scaling_study.json and, if matplotlib is present,
artifacts/scaling_study.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from certabstain import Interval, MLP, propagate_tube
from certabstain.discrepancy import _batched_ibp, certify_epsilon
from certabstain.nnbound import fit_mlp
from certabstain.provenance import artifact_writes_enabled, write_provenance_sidecar
from certabstain.reference import SpringDamper2D

OUT = Path(__file__).resolve().parent / "artifacts"

DIMS = (2, 3, 4, 5, 6, 8)
FIXED_BUDGET = 50_000
DELTA = 0.02
LAM = 0.4


# ===================================================================== #
# Part A: input-dimension scaling (BnB curse of dimensionality)
# ===================================================================== #


def _dim_d_contraction(d: int):
    """f(x) = LAM*x + b, a d-dim contraction; twin = f + DELTA (per-dim,
    alternating sign so no dimension is accidentally symmetric/degenerate)."""
    W = LAM * np.eye(d)
    b = 0.01 * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(d)])
    net = MLP(((W, b),))
    deltas = DELTA * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(d)])
    twin = MLP(((W, b + deltas),))
    return net, twin


def dimension_scaling() -> list[dict]:
    rows = []
    for d in DIMS:
        net, twin = _dim_d_contraction(d)

        def ref(lo, hi, twin=twin):
            return _batched_ibp(twin, lo, hi)

        domain = Interval(-0.2 * np.ones(d), 0.2 * np.ones(d))
        t0 = time.time()
        cert = certify_epsilon(
            net,
            ref,
            domain,
            reference_id=f"linear contraction d={d}",
            ref_float=twin.forward,
            target=None,
            max_leaf_evals=FIXED_BUDGET,
            floor_samples=20_000,
        )
        elapsed = time.time() - t0
        eps = float(np.max(cert.eps))
        ratio = eps / DELTA
        rows.append(
            {
                "d": d,
                "eps": eps,
                "true_gap": DELTA,
                "ratio_to_true_gap": ratio,
                "n_leaves": cert.n_leaves,
                "n_leaf_evals": cert.n_leaf_evals,
                "cover_fraction": cert.cover_fraction,
                "seconds": elapsed,
            }
        )
        print(
            f"d={d}  eps={eps:.4g}  true_gap={DELTA}  ratio={ratio:.2f}x  "
            f"leaves={cert.n_leaves}  evals={cert.n_leaf_evals}  "
            f"time={elapsed:.2f}s"
        )
    return rows


# ===================================================================== #
# Part B: horizon scaling (how far past K=10 does the tube survive?)
# ===================================================================== #


def _yv_subsystem(model: SpringDamper2D):
    def step(yv, uy):
        n = yv.shape[0]
        s = np.zeros((n, 4))
        s[:, 1] = yv[:, 0] + 1.0
        s[:, 3] = yv[:, 1]
        u = np.zeros((n, 2))
        u[:, 1] = uy[:, 0]
        out = model.step(s, u)
        return out[:, [1, 3]] - np.array([1.0, 0.0])

    def step_interval(lo, hi):
        n = lo.shape[0]
        zero = np.zeros(n)
        S = Interval(
            np.stack([zero, lo[:, 0] + 1.0, zero, lo[:, 1]], axis=1),
            np.stack([zero, hi[:, 0] + 1.0, zero, hi[:, 1]], axis=1),
        )
        U = Interval(np.stack([zero, lo[:, 2]], axis=1), np.stack([zero, hi[:, 2]], axis=1))
        enc = model.step_interval(S, U)
        offset = np.array([1.0, 0.0])
        return enc.lo[:, [1, 3]] - offset, enc.hi[:, [1, 3]] - offset

    return step, step_interval


def horizon_scaling(k_max: int = 40) -> dict:
    model = SpringDamper2D()
    step, step_interval = _yv_subsystem(model)
    rng = np.random.default_rng(5)

    domain = Interval(np.array([-0.2, -0.5, -0.3]), np.array([0.2, 0.5, 0.3]))
    X = rng.uniform(domain.lo, domain.hi, size=(100_000, 3))
    Y = step(X[:, :2], X[:, 2:])
    net = fit_mlp((3, 8, 2), X, Y, steps=15_000, lr=2e-3, seed=1)

    cert = certify_epsilon(
        net,
        lambda lo, hi: step_interval(lo, hi),
        domain,
        reference_id="SpringDamper2D()/free-flight (y, vy) subsystem",
        ref_float=lambda p: step(p[:, :2], p[:, 2:]),
        target=None,
        max_leaf_evals=400_000,
        floor_samples=300_000,
    )

    X0 = Interval(np.array([-0.02, -0.02]), np.array([0.02, 0.02]))
    U_box = Interval(np.array([-0.02]), np.array([0.02]))
    tube = propagate_tube(net, cert, X0, [U_box] * k_max, n_states=2)

    print(
        f"horizon requested={k_max}  achieved={tube.horizon}  "
        f"exit_reason={tube.cover_exit_reason!r}"
    )
    widths = tube.widths.tolist()
    for t, w in enumerate(widths):
        print(f"  K={t:2d}  width_y={w[0]:.4g}  width_vy={w[1]:.4g}")

    return {
        "requested_horizon": k_max,
        "achieved_horizon": tube.horizon,
        "cover_exit_reason": tube.cover_exit_reason,
        "eps": cert.eps.tolist(),
        "domain": {"lo": domain.lo.tolist(), "hi": domain.hi.tolist()},
        "y_widths": [w[0] for w in widths],
        "vy_widths": [w[1] for w in widths],
    }


def main() -> None:
    print("=== Part A: input-dimension scaling ===")
    dim_rows = dimension_scaling()

    print("\n=== Part B: horizon scaling ===")
    horizon_row = horizon_scaling()

    data = {"dimension_scaling": dim_rows, "horizon_scaling": horizon_row}

    if not artifact_writes_enabled():
        print(
            "\n[artifacts] CERTABSTAIN_WRITE_ARTIFACTS not set; not writing "
            f"{OUT / 'scaling_study.json'} (leaving committed bytes as-is)"
        )
        return

    OUT.mkdir(exist_ok=True)
    (OUT / "scaling_study.json").write_text(json.dumps(data, indent=2))
    write_provenance_sidecar(OUT / "scaling_study.json", writer="scaling_study.py")
    print(f"\nwrote {OUT / 'scaling_study.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        ds = [r["d"] for r in dim_rows]
        ax1.semilogy(ds, [r["ratio_to_true_gap"] for r in dim_rows], "o-")
        ax1.axhline(3.0, color="red", ls="--", lw=1, label="M3's 3x acceptance ratio")
        ax1.set_xlabel("input dimension d")
        ax1.set_ylabel("certified eps / true gap")
        ax1.set_title(f"BnB conservatism vs dimension (fixed {FIXED_BUDGET:,} evals)")
        ax1.legend(fontsize=8)

        hy = horizon_row["y_widths"]
        hv = horizon_row["vy_widths"]
        ks_axis = list(range(len(hy)))
        ax2.plot(ks_axis, hy, "o-", label="y width")
        ax2.plot(ks_axis, hv, "s-", label="vy width")
        if horizon_row["achieved_horizon"] < horizon_row["requested_horizon"]:
            ax2.axvline(
                horizon_row["achieved_horizon"], color="red", ls="--", lw=1,
                label=f"cover exit @ K={horizon_row['achieved_horizon']}",
            )
        ax2.set_xlabel("tube step K")
        ax2.set_ylabel("tube width")
        ax2.set_title("Horizon scaling: (y, vy) tube past M4's K=10")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT / "scaling_study.png", dpi=140)
        print(f"wrote {OUT / 'scaling_study.png'}")
    except ImportError:
        print("matplotlib not available; JSON only")


if __name__ == "__main__":
    main()
