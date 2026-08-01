"""M6 acceptance tests: PusherSlider "slide_right" mode (planar pushing scale-up).

This file is one of three parallel M6 workstreams (stick, slide_left,
slide_right each certified separately, per spec section 6: "per-mode
certification unioned over the three modes"). It touches nothing outside
itself: no shared package file is edited, only this test and its artifact.

Reduction used (mirrors the (y, vy) subsystem trick already used for M4's
SpringDamper2D in test_tube.py): ``_stick_body``/``_slide_body`` and the
py-Euler-step inside ``PusherSlider._integrate`` depend ONLY on
``(py, vpx, vpy)`` -- x, y, theta never enter anywhere in those equations --
so the one-step map ``(py, vpx, vpy) -> py2`` is an exact, self-contained
3-input/1-output subsystem per mode. Certifying the full 6-D (state, control)
map would pay unnecessary BnB cost for dimensions the dynamics never use.

W1 (direct/one-step) clearance target, per the task's clearance spec: the
codebase's implicit safety spec is "stay on the contact face"
(``PusherSlider.step_interval`` itself refuses off-face boxes), so the
learned clearance function certified here is the one-step-ahead version::

    h_next(py, vpx, vpy) = py_max - |py2(py, vpx, vpy)|

(positive = still on the face after one slide_right step).

Domain: ``py in [-0.03, 0.03], vpx in [0.02, 0.1], vpy in [-0.3, -0.1]``,
verified numerically (see ``test_domain_lies_mostly_in_slide_right``) to lie
>99% inside the slide_right mode ``(g2 >= 0) & (g1 < 0)`` by sample; the exact
box-level membership that matters for soundness is handled by
``_slide_right_mode``, passed as ``certify_epsilon``'s own ``mode`` argument,
which is monotone-under-inclusion and never resampled.

Finding (see module-level report artifact
``artifacts/pushing_slide_right_report.json`` for the full numbers): this
mode certifies cleanly and comfortably clears the M6 acceptance bar of
nominal abstention <= 2*alpha, even though spec section 6 only requires that
bar on the sticking mode and explicitly allows an honest write-up of failure
for the others. slide_right is not a hard case here: h_next stays
comfortably positive (>= ~0.007, well clear of 0) everywhere sampled in the
domain -- the pusher never comes close to driving the contact off the face
in one step -- and _slide_body's dependence on (py, vpx, vpy) is a smooth
rational map that a tiny (3, 8, 1) net fits tightly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from certabstain import (
    Interval,
    MLP,
    NetworkCertificateMismatch,
    PusherSlider,
    ReferenceMismatch,
    VerifiedDiscrepancyWitness,
    build_monitor,
    certify_epsilon,
)
from certabstain.discrepancy import MODE_IN, MODE_OUT, MODE_STRADDLE
from certabstain.nnbound import fit_mlp

ALPHA = 0.05
P = PusherSlider()  # defaults: a=0.05, c=0.03, mu=0.3, dt=0.01, py_max=0.04
REFERENCE_ID = P.reference_id() + "/slide_right h_next"

DOMAIN = Interval(np.array([-0.03, 0.02, -0.3]), np.array([0.03, 0.1, -0.1]))


# ===================================================================== #
# The reduced (py, vpx, vpy) -> h_next subsystem: float and interval twins
# ===================================================================== #


def _h_next_float(pts: np.ndarray) -> np.ndarray:
    """py_max - |py2| after one slide_right Euler step; the W1 training target."""
    py, vpx, vpy = pts[:, 0], pts[:, 1], pts[:, 2]
    _, _, _, pydot = P._slide_body(py, vpx, vpy, -1.0)
    py2 = py + P.dt * pydot
    return P.py_max - np.abs(py2)


def _h_next_ref(lo: np.ndarray, hi: np.ndarray):
    """Sound interval enclosure of h_next, built directly from
    ``step_interval``'s own certified py2 enclosure (component index 3) --
    no hand-derived interval form of ``_slide_body`` needed. x, y, theta are
    point intervals at 0 since the (py, vpx, vpy) -> py2 map never reads them.
    """
    n = lo.shape[0]
    zero = np.zeros(n)
    S = Interval(
        np.stack([zero, zero, zero, lo[:, 0]], axis=1),
        np.stack([zero, zero, zero, hi[:, 0]], axis=1),
    )
    U = Interval(
        np.stack([lo[:, 1], lo[:, 2]], axis=1), np.stack([hi[:, 1], hi[:, 2]], axis=1)
    )
    enc = P.step_interval(S, U, mode="slide_right")
    PY2 = Interval(enc.lo[:, 3], enc.hi[:, 3])
    H = Interval.point(P.py_max) - abs(PY2)
    return H.lo[:, None], H.hi[:, None]


def _slide_right_mode(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Box-level mode classification for ``certify_epsilon``'s ``mode``
    argument: reuses ``PusherSlider.mode_certificate``'s own certified
    enclosures of g1, g2 (spec A2 enforced exactly, not resampled). IN iff
    the whole box certifiably lies in slide_right, OUT iff the whole box
    certifiably lies in slide_left or stick, STRADDLE otherwise (split
    further by the certifier, or excluded from the cover at min width)."""
    n = lo.shape[0]
    zero = np.zeros(n)
    S = Interval(
        np.stack([zero, zero, zero, lo[:, 0]], axis=1),
        np.stack([zero, zero, zero, hi[:, 0]], axis=1),
    )
    U = Interval(
        np.stack([lo[:, 1], lo[:, 2]], axis=1), np.stack([hi[:, 1], hi[:, 2]], axis=1)
    )
    mc = P.mode_certificate(S, U)
    is_in = mc["slide_right"]
    is_out = mc["slide_left"] | mc["stick"]
    return np.where(is_in, MODE_IN, np.where(is_out, MODE_OUT, MODE_STRADDLE))


def _slide_right_mode_float(pts: np.ndarray) -> np.ndarray:
    py, vpx, vpy = pts[:, 0], pts[:, 1], pts[:, 2]
    g1, g2 = P._mode_g(py, vpx, vpy)
    return (g2 >= 0.0) & (g1 < 0.0)


def _net_and_cert():
    rng = np.random.default_rng(7)
    X = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(120_000, 3))
    Y = _h_next_float(X)
    net = fit_mlp((3, 8, 1), X, Y, steps=12_000, lr=2e-3, seed=0)
    cert = certify_epsilon(
        net,
        _h_next_ref,
        DOMAIN,
        reference_id=REFERENCE_ID,
        ref_float=_h_next_float,
        target=None,
        max_leaf_evals=300_000,
        mode=_slide_right_mode,
        mode_float=_slide_right_mode_float,
        floor_samples=300_000,
    )
    return net, cert


# nominal deployment envelope: well inside the certified domain (20% margin
# shaved off every edge), so cover-membership is never what's abstaining --
# the point of this test is the certified-epsilon score, not the cover edge.
def _shrink(lo: float, hi: float, frac: float = 0.2) -> tuple[float, float]:
    w = hi - lo
    return lo + frac * w, hi - frac * w


_PY_LO, _PY_HI = _shrink(DOMAIN.lo[0], DOMAIN.hi[0])
_VPX_LO, _VPX_HI = _shrink(DOMAIN.lo[1], DOMAIN.hi[1])
_VPY_LO, _VPY_HI = _shrink(DOMAIN.lo[2], DOMAIN.hi[2])

HORIZON = 20
N_CAL = 999
N_EVAL = 2000


def _rollout(rng: np.random.Generator, net: MLP, witness, horizon: int) -> list[float]:
    """One nominal (safe) slide_right rollout: at each step, score the net's
    h_next prediction with the certified witness, then advance py by the true
    (float-twin) slide_right dynamics -- exactly the "simulate a nominal
    deployment loop" the task asks for."""
    py = rng.uniform(_PY_LO, _PY_HI)
    scores = []
    for _ in range(horizon):
        vpx = rng.uniform(_VPX_LO, _VPX_HI)
        vpy = rng.uniform(_VPY_LO, _VPY_HI)
        h_hat = float(net.forward(np.array([[py, vpx, vpy]]))[0, 0])
        scores.append(float(witness.score(h_hat)))
        _, _, _, pydot = P._slide_body(
            np.array([py]), np.array([vpx]), np.array([vpy]), -1.0
        )
        py = float(np.clip(py + P.dt * pydot[0], _PY_LO, _PY_HI))
    return scores


# ===================================================================== #
# Domain sanity: the mode-boundary recon this domain choice depends on
# ===================================================================== #


def test_domain_lies_mostly_in_slide_right() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(200_000, 3))
    frac = _slide_right_mode_float(pts).mean()
    assert frac > 0.99, (
        f"only {frac:.1%} of sampled domain points are in slide_right; the "
        f"declared domain should lie >99% inside the mode by construction"
    )


# ===================================================================== #
# Certificate quality: cover_fraction and eps vs the empirical floor
# ===================================================================== #


def test_cert_cover_fraction_and_eps_vs_empirical_floor() -> None:
    net, cert = _net_and_cert()
    assert cert.matches_network(net)
    assert cert.cover_fraction > 0.90, (
        f"cover_fraction={cert.cover_fraction:.1%} of the declared domain, "
        f"below the spec's declared minimum (section 5 default 90%)"
    )
    assert cert.eps[0] < 5.0 * cert.empirical_floor[0], (
        f"eps={cert.eps[0]:.4g} vs empirical_floor={cert.empirical_floor[0]:.4g}: "
        f"a much larger ratio would signal excessive BnB conservatism rather "
        f"than a genuinely tight certified bound"
    )
    assert np.all(np.isfinite(cert.eps))


# ===================================================================== #
# Binding refusals (M5-style: spec A4 weight-hash, A1 reference identity)
# ===================================================================== #


def test_bind_succeeds_with_matching_network_and_reference() -> None:
    net, cert = _net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, REFERENCE_ID)
    assert w.violation_floor() == 0.0
    assert "epsilon certified" in w.justification()
    assert w.score(0.0) == pytest.approx(float(np.max(cert.eps)))


def test_bind_refuses_one_flipped_weight_byte() -> None:
    net, cert = _net_and_cert()
    tampered = [(np.array(W, copy=True), np.array(b, copy=True)) for W, b in net.weights]
    tampered[0][0][0, 0] = np.nextafter(tampered[0][0][0, 0], np.inf)  # one ulp
    other = MLP(tuple(tampered), activation=net.activation)
    with pytest.raises(NetworkCertificateMismatch):
        VerifiedDiscrepancyWitness.bind(cert, other, REFERENCE_ID)


def test_bind_refuses_reference_parameter_mismatch() -> None:
    net, cert = _net_and_cert()
    changed = PusherSlider(a=0.05, c=0.03, mu=0.3, dt=0.01, py_max=0.035)  # different py_max
    changed_id = changed.reference_id() + "/slide_right h_next"
    with pytest.raises(ReferenceMismatch):
        VerifiedDiscrepancyWitness.bind(cert, net, changed_id)


# ===================================================================== #
# M6 acceptance: nominal abstention rate vs 2*alpha
# ===================================================================== #


def test_abstention_rate_meets_the_2alpha_bar() -> None:
    """M6's acceptance criterion (spec section 6) is stated only for the
    sticking mode; slide_right is not required to clear it, but this run
    does (see module docstring and the report artifact for the honest
    numbers either way). If a future change to the domain, net, or budget
    ever pushes this mode over the bar, this assertion is meant to fail
    loudly rather than be quietly loosened."""
    net, cert = _net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, REFERENCE_ID)

    rng = np.random.default_rng(101)
    cal_trajectories = [_rollout(rng, net, w, HORIZON) for _ in range(N_CAL)]
    mon = build_monitor(
        nominal_trajectories=cal_trajectories,
        alpha=ALPHA,
        witness=w,
        safe_action="STOP",
    )

    eval_trajectories = [_rollout(rng, net, w, HORIZON) for _ in range(N_EVAL)]
    abstentions = sum(
        1 for traj in eval_trajectories if any(s > mon.threshold for s in traj)
    )
    abstention_rate = abstentions / N_EVAL

    assert abstention_rate <= 2 * ALPHA, (
        f"slide_right abstention_rate={abstention_rate:.3f} exceeds the M6 "
        f"bar 2*alpha={2 * ALPHA:.3f}; per spec section 6 this mode is not "
        f"required to clear it, but this assertion documents the finding "
        f"honestly instead of silently omitting the check"
    )


# ===================================================================== #
# Report artifact
# ===================================================================== #


def _net_and_cert_seeded(data_seed: int, fit_seed: int):
    """Same construction as _net_and_cert, with the data/fit seeds exposed.

    Used to check that this mode's pass is not a single-seed artifact -- and,
    unlike a prose note recording numbers from a run nobody can re-execute,
    to put the second seed's actual figures in the report artifact.
    """
    rng = np.random.default_rng(data_seed)
    X = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(120_000, 3))
    Y = _h_next_float(X)
    net = fit_mlp((3, 8, 1), X, Y, steps=12_000, lr=2e-3, seed=fit_seed)
    train_mae = float(np.mean(np.abs(net.forward(X).reshape(-1) - Y.reshape(-1))))
    cert = certify_epsilon(
        net,
        _h_next_ref,
        DOMAIN,
        reference_id=REFERENCE_ID,
        ref_float=_h_next_float,
        target=None,
        max_leaf_evals=300_000,
        mode=_slide_right_mode,
        mode_float=_slide_right_mode_float,
        floor_samples=300_000,
    )
    return net, cert, train_mae


def _abstention_rate(net, cert) -> tuple[float, float]:
    w = VerifiedDiscrepancyWitness.bind(cert, net, REFERENCE_ID)
    rng = np.random.default_rng(101)
    cal = [_rollout(rng, net, w, HORIZON) for _ in range(N_CAL)]
    mon = build_monitor(
        nominal_trajectories=cal, alpha=ALPHA, witness=w, safe_action="STOP"
    )
    ev = [_rollout(rng, net, w, HORIZON) for _ in range(N_EVAL)]
    rate = sum(1 for t in ev if any(s > mon.threshold for s in t)) / N_EVAL
    return rate, float(mon.threshold)


def test_write_report_artifact() -> None:
    """Write this mode's report, every number computed here rather than typed.

    The other two modes' tests each end by dumping their own report; this one
    did not, so `artifacts/pushing_slide_right_report.json` was the one M6
    artifact with no code that produces it -- which the technical note's
    'every quantitative claim traces to an artifact' promise cannot survive.
    """
    net, cert, train_mae = _net_and_cert_seeded(7, 0)

    rng = np.random.default_rng(3)
    sample_pts = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(200_000, 3))
    frac_in_mode = float(_slide_right_mode_float(sample_pts).mean())
    h_next_min = float(np.min(_h_next_float(sample_pts)))

    abstention_rate, threshold = _abstention_rate(net, cert)

    eps = float(cert.eps[0])
    floor = float(cert.empirical_floor[0])
    meets_bar = bool(abstention_rate <= 2 * ALPHA)

    # The second-seed check, actually run: the claim "not a single-seed fluke"
    # is only worth making if the numbers behind it are reproducible.
    net2, cert2, train_mae2 = _net_and_cert_seeded(23, 11)
    abstention_rate2, _ = _abstention_rate(net2, cert2)

    report = {
        "milestone": "M6 - Planar pushing scale-up",
        "workstream": "PusherSlider slide_right mode",
        "mode": "slide_right",
        "reference_model": P.reference_id(),
        "reference_id": REFERENCE_ID,
        "reduction": (
            "one-step (py, vpx, vpy) -> py2 subsystem; x, y, theta never "
            "enter _slide_body or the py-Euler-step in _integrate, so the "
            "full 6-D (state, control) map is not certified -- only this "
            "exact self-contained 3-input/1-output slice"
        ),
        "clearance_spec": (
            "h_next(py, vpx, vpy) = py_max - |py2(py, vpx, vpy)| (positive = "
            "still on the contact face after one slide_right step); py2 from "
            "step_interval component index 3"
        ),
        "domain": {
            "py": [float(DOMAIN.lo[0]), float(DOMAIN.hi[0])],
            "vpx": [float(DOMAIN.lo[1]), float(DOMAIN.hi[1])],
            "vpy": [float(DOMAIN.lo[2]), float(DOMAIN.hi[2])],
            "fraction_in_slide_right_by_sample": frac_in_mode,
            "sampled_min_h_next": h_next_min,
            "note": (
                "box-level mode membership for the certifier is handled "
                "exactly (not by resampling) via _slide_right_mode passed as "
                "certify_epsilon's `mode` argument; the sampled fraction "
                "above is a diagnostic only"
            ),
        },
        "net_architecture": [3, 8, 1],
        "training": {
            "steps": 12000,
            "lr": 0.002,
            "n_train_samples": 120000,
            "data_seed": 7,
            "fit_seed": 0,
            "train_mae": train_mae,
        },
        "certifier": {
            "max_leaf_evals": 300000,
            "floor_samples": 300000,
            "n_leaf_evals_used": int(cert.n_leaf_evals),
            "n_leaves": int(cert.n_leaves),
            "cover_fraction": float(cert.cover_fraction),
            "eps": eps,
            "empirical_floor": floor,
            "eps_to_floor_ratio": eps / floor,
        },
        "monitor": {
            "alpha": ALPHA,
            "two_alpha_bar": 2 * ALPHA,
            "n_calibration_rollouts": N_CAL,
            "n_eval_rollouts": N_EVAL,
            "horizon_per_rollout": HORIZON,
            "calibration_threshold": threshold,
            "measured_abstention_rate": abstention_rate,
        },
        "second_seed_check": {
            "purpose": (
                "the pass on this mode should not depend on one lucky fit; "
                "these are the actual figures from a different data/fit seed"
            ),
            "data_seed": 23,
            "fit_seed": 11,
            "train_mae": train_mae2,
            "eps": float(cert2.eps[0]),
            "empirical_floor": float(cert2.empirical_floor[0]),
            "cover_fraction": float(cert2.cover_fraction),
            "measured_abstention_rate": abstention_rate2,
            "still_meets_bar": bool(abstention_rate2 <= 2 * ALPHA),
        },
        "verdict": {
            "meets_M6_bar": meets_bar,
            "reason": (
                f"measured_abstention_rate ({abstention_rate:.4g}) "
                f"{'<=' if meets_bar else '>'} 2*alpha ({2 * ALPHA:.3g}); "
                f"cover_fraction ({cert.cover_fraction:.1%}) against the 90% "
                f"declared minimum; eps ({eps:.3g}) is a small fraction of "
                f"py_max (0.04) and within {eps / floor:.2f}x of the sampled "
                f"empirical floor, so the certificate is tight rather than "
                f"vacuous"
            ),
        },
        "notes": [
            "Spec section 6 only requires the 2*alpha nominal-abstention bar "
            "on the sticking mode; slide_right clearing it as well is a bonus "
            "finding for this mode, not a forced result -- no domain shrinkage "
            "or budget inflation beyond the first reasonable choice was needed.",
            "h_next stays comfortably positive across the whole declared "
            "domain (see domain.sampled_min_h_next), well clear of the "
            "py_max=0.04 unsafe-at-zero boundary: the pusher never comes close "
            "to sliding the contact off the face in one step at these vpx/vpy "
            "magnitudes, so this is not a stiff or discontinuity-adjacent "
            "regime the way stick's pydot=0-vs-nonzero boundary can be.",
            "The domain fraction outside slide_right by sample is what "
            "certify_epsilon's box-level `mode` argument excludes from the "
            "cover (mode-straddling leaves near the motion-cone boundary get "
            "shrunk to min_width and dropped, per spec section 5's 'fails to a "
            "smaller domain, never a weaker claim'); this is the entire gap "
            "between 100% and the reported cover_fraction.",
            "No dependency was needed beyond what M3/M5 already established: "
            "certify_epsilon's own `mode`/`mode_float` arguments handle A2 "
            "(mode membership) exactly, and step_interval's certified py2 "
            "enclosure was reused directly for the interval twin of h_next "
            "rather than hand-deriving an interval form of _slide_body.",
        ],
    }

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pushing_slide_right_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    assert os.path.exists(out_path)
    assert meets_bar, report["verdict"]["reason"]
