"""M6 acceptance tests: planar pushing scale-up, "stick" mode.

One of three parallel per-mode workstreams (the siblings cover slide_left and
slide_right in their own files); this file only ever touches PusherSlider's
stick mode.

Reduction used throughout (see module docstring rationale in the M6 task):
PusherSlider's one-step map (py, vpx, vpy) -> py2 is an exact, self-contained
3-input/1-output subsystem in every mode (x, y, theta never enter the body
equations or the py2 update). W1's learned clearance function is therefore
the one-step-ahead face clearance

    h_next(py, vpx, vpy) = py_max - |py2(py, vpx, vpy)|

which in the stick mode collapses to ``py_max - |py|`` exactly, since
``_stick_body`` returns ``pydot = 0`` identically -- the interesting part is
that certification still has to *prove* that collapse sound-arithmetically
rather than assume it, and that a learned net trained on 3 inputs discovers
it.

Spec criteria exercised here (section 6, M6, restricted to this mode):
  * certified two-sided monitor on the sticking mode with nominal abstention
    <= 2 * alpha;
  * (mirroring M5's binding tests, section 6 M5 / section 7 refusal surface)
    one flipped weight byte => refusal at load; reference-parameter
    mismatch => refusal.
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
from certabstain.provenance import artifact_writes_enabled, write_provenance_sidecar

# Mode-boundary recon (verified numerically, see task notes): this box lies
# ~99.8% inside the stick mode for PusherSlider() defaults.
DOM_LO = np.array([-0.008, 0.05, -0.05])
DOM_HI = np.array([0.008, 0.15, 0.05])
DOMAIN = Interval(DOM_LO, DOM_HI)

# Nominal deployment region: comfortably inside the domain and away from its
# edges (and therefore certainly away from the mode boundary).
INNER_LO = np.array([-0.003, 0.07, -0.03])
INNER_HI = np.array([0.003, 0.13, 0.03])

ALPHA = 0.05


def _h_next_float(p: PusherSlider, X: np.ndarray) -> np.ndarray:
    """Float twin of the learned target: py_max - |py2| under the stick body."""
    py, vpx, vpy = X[:, 0], X[:, 1], X[:, 2]
    _vx, _vy, _w, pydot = p._stick_body(py, vpx, vpy)
    py2 = py + p.dt * pydot
    return p.py_max - np.abs(py2)


def _mode_fn(p: PusherSlider):
    """Certified stick-mode membership for branch-and-bound's mode filter.

    IN iff both motion-cone cross products are certifiably non-negative
    everywhere in the box; OUT iff either is certifiably negative somewhere
    (i.e. the box certifiably touches slide_left or slide_right); otherwise
    STRADDLE, to be split further or excluded per spec section 5.
    """

    def mode_fn(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        PY = Interval(lo[:, 0], hi[:, 0])
        VPX = Interval(lo[:, 1], hi[:, 1])
        VPY = Interval(lo[:, 2], hi[:, 2])
        G1, G2 = p._mode_g(PY, VPX, VPY)
        stick_in = (G1.lo >= 0.0) & (G2.lo >= 0.0)
        stick_out = (G1.hi < 0.0) | (G2.hi < 0.0)
        out = np.full(lo.shape[0], MODE_STRADDLE)
        out[stick_in] = MODE_IN
        out[stick_out] = MODE_OUT
        return out

    return mode_fn


def _mode_float(p: PusherSlider):
    def mode_float(X: np.ndarray) -> np.ndarray:
        g1, g2 = p._mode_g(X[:, 0], X[:, 1], X[:, 2])
        return (g1 >= 0.0) & (g2 >= 0.0)

    return mode_float


def _ref_interval(p: PusherSlider):
    """Certified h_next enclosure, wired directly through step_interval's
    already-certified py2 component (index 3) -- no hand-derived interval
    form of _stick_body needed. Only called on leaves already certified to
    lie in the stick mode by the mode filter above, so step_interval's own
    in_mode / face-domain checks never fire here."""

    def ref(lo: np.ndarray, hi: np.ndarray):
        B = lo.shape[0]
        zero = np.zeros(B)
        S = Interval(
            np.stack([zero, zero, zero, lo[:, 0]], axis=1),
            np.stack([zero, zero, zero, hi[:, 0]], axis=1),
        )
        U = Interval(
            np.stack([lo[:, 1], lo[:, 2]], axis=1),
            np.stack([hi[:, 1], hi[:, 2]], axis=1),
        )
        enc = p.step_interval(S, U, mode="stick")
        PY2 = Interval(enc.lo[:, 3], enc.hi[:, 3])
        H = Interval.point(p.py_max) - abs(PY2)
        return H.lo[:, None], H.hi[:, None]

    return ref


def _net_and_cert(p: PusherSlider | None = None, seed: int = 0):
    p = p or PusherSlider()
    rng = np.random.default_rng(seed)
    X = rng.uniform(DOM_LO, DOM_HI, size=(200_000, 3))
    Y = _h_next_float(p, X)
    net = fit_mlp((3, 8, 1), X, Y, steps=10_000, lr=2e-3, seed=seed)

    cert = certify_epsilon(
        net,
        _ref_interval(p),
        DOMAIN,
        reference_id=p.reference_id(),
        ref_float=lambda pts: _h_next_float(p, pts),
        target=None,
        max_leaf_evals=200_000,
        mode=_mode_fn(p),
        mode_float=_mode_float(p),
        floor_samples=300_000,
        seed=seed,
    )
    return p, net, cert


@pytest.fixture(scope="module")
def net_and_cert():
    return _net_and_cert()


# ===================================================================== #
# Certificate sanity: cover_fraction and eps vs empirical_floor
# ===================================================================== #


def test_certificate_covers_the_declared_domain(net_and_cert) -> None:
    p, net, cert = net_and_cert
    # spec section 5 default: refuse below 90% of the declared domain; this
    # domain was chosen (recon) to sit almost entirely inside stick, so the
    # achieved cover should clear that bar comfortably.
    assert cert.cover_fraction >= 0.90, cert.cover_fraction
    assert cert.matches_network(net)
    assert cert.reference_id == p.reference_id()


def test_certificate_eps_brackets_empirical_floor(net_and_cert) -> None:
    _p, _net, cert = net_and_cert
    eps = float(cert.eps[0])
    floor = float(cert.empirical_floor[0])
    # sanity, not a tight ratio requirement (per task): the certified bound
    # can never sit below an observed gap, and should not be wildly loose.
    assert eps >= floor, "certified bound cannot sit below an observed gap"
    ratio = eps / floor if floor > 0 else float("inf")
    print(
        f"\n[M6 stick] eps={eps:.6g} empirical_floor={floor:.6g} ratio={ratio:.3g} "
        f"cover_fraction={cert.cover_fraction:.4g} leaves={cert.n_leaves} "
        f"leaf_evals={cert.n_leaf_evals}"
    )
    assert np.isfinite(ratio)


# ===================================================================== #
# Binding refusals (mirrors M5's test_witness2.py pattern)
# ===================================================================== #


def test_bind_succeeds_with_matching_network_and_reference(net_and_cert) -> None:
    p, net, cert = net_and_cert
    w = VerifiedDiscrepancyWitness.bind(cert, net, p.reference_id())
    assert w.violation_floor() == 0.0
    assert "epsilon certified" in w.justification()


def test_bind_refuses_one_flipped_weight_byte(net_and_cert) -> None:
    p, net, cert = net_and_cert
    tampered = [(np.array(W, copy=True), np.array(b, copy=True)) for W, b in net.weights]
    tampered[0][0][0, 0] = np.nextafter(tampered[0][0][0, 0], np.inf)  # one ulp
    other = MLP(tuple(tampered), activation=net.activation)
    with pytest.raises(NetworkCertificateMismatch):
        VerifiedDiscrepancyWitness.bind(cert, other, p.reference_id())


def test_bind_refuses_reference_parameter_mismatch(net_and_cert) -> None:
    _p, net, cert = net_and_cert
    changed = PusherSlider(py_max=0.035)  # different declared face domain
    with pytest.raises(ReferenceMismatch):
        VerifiedDiscrepancyWitness.bind(cert, net, changed.reference_id())


# ===================================================================== #
# The M6 acceptance bar: nominal abstention <= 2 * alpha
# ===================================================================== #


def test_nominal_abstention_rate_within_twice_alpha(net_and_cert) -> None:
    p, net, cert = net_and_cert
    witness = VerifiedDiscrepancyWitness.bind(cert, net, p.reference_id())

    rng = np.random.default_rng(1)
    n_cal, horizon = 500, 20
    trajs = []
    for _ in range(n_cal):
        pts = rng.uniform(INNER_LO, INNER_HI, size=(horizon, 3))
        h_hat = net.forward(pts)[:, 0]
        trajs.append(list(witness.score(h_hat)))

    monitor = build_monitor(
        nominal_trajectories=trajs, alpha=ALPHA, witness=witness, safe_action="STOP"
    )
    assert monitor.claim is not None and monitor.claim.miss_bound == 0.0

    n_roll = 5000
    fired = 0
    for _ in range(n_roll):
        pt = rng.uniform(INNER_LO, INNER_HI, size=(1, 3))
        h_hat = float(net.forward(pt)[0, 0])
        score = float(witness.score(h_hat))
        d = monitor.step(observation=pt[0], proposed_action=np.array([0.0]), score=score)
        fired += bool(d.abstained)

    rate = fired / n_roll
    print(
        f"\n[M6 stick acceptance] abstention_rate={rate:.4g} "
        f"2*alpha={2 * ALPHA:.4g} eps={float(cert.eps[0]):.6g}"
    )
    assert rate <= 2 * ALPHA, (
        f"nominal abstention rate {rate:.4g} exceeds the M6 bar of "
        f"2*alpha={2 * ALPHA:.4g}"
    )


# ===================================================================== #
# Write the M6 artifact report
# ===================================================================== #


def test_write_report_artifact(net_and_cert) -> None:
    p, net, cert = net_and_cert
    witness = VerifiedDiscrepancyWitness.bind(cert, net, p.reference_id())

    rng = np.random.default_rng(2)
    n_cal, horizon = 500, 20
    trajs = []
    for _ in range(n_cal):
        pts = rng.uniform(INNER_LO, INNER_HI, size=(horizon, 3))
        h_hat = net.forward(pts)[:, 0]
        trajs.append(list(witness.score(h_hat)))

    monitor = build_monitor(
        nominal_trajectories=trajs, alpha=ALPHA, witness=witness, safe_action="STOP"
    )

    n_roll = 5000
    fired = 0
    for _ in range(n_roll):
        pt = rng.uniform(INNER_LO, INNER_HI, size=(1, 3))
        h_hat = float(net.forward(pt)[0, 0])
        score = float(witness.score(h_hat))
        d = monitor.step(observation=pt[0], proposed_action=np.array([0.0]), score=score)
        fired += bool(d.abstained)
    rate = fired / n_roll

    eps = float(cert.eps[0])
    floor = float(cert.empirical_floor[0])
    meets_bar = bool(rate <= 2 * ALPHA)

    report = {
        "milestone": "M6",
        "mode": "stick",
        "reference_id": p.reference_id(),
        "domain": {
            "variables": ["py", "vpx", "vpy"],
            "lo": DOM_LO.tolist(),
            "hi": DOM_HI.tolist(),
        },
        "nominal_region": {
            "lo": INNER_LO.tolist(),
            "hi": INNER_HI.tolist(),
        },
        "eps": eps,
        "empirical_floor": floor,
        "eps_over_floor": eps / floor if floor > 0 else None,
        "cover_fraction": cert.cover_fraction,
        "n_leaves": cert.n_leaves,
        "n_leaf_evals": cert.n_leaf_evals,
        "alpha": ALPHA,
        "abstention_rate": rate,
        "twice_alpha": 2 * ALPHA,
        "verdict": {
            "meets_M6_bar": meets_bar,
            "reason": (
                f"nominal abstention rate {rate:.4g} <= 2*alpha={2 * ALPHA:.4g}, "
                f"with certified eps={eps:.4g} (cover {cert.cover_fraction:.1%})"
                if meets_bar
                else (
                    f"nominal abstention rate {rate:.4g} EXCEEDS 2*alpha="
                    f"{2 * ALPHA:.4g}; eps={eps:.4g} too loose relative to the "
                    f"nominal safety margin in the declared inner region"
                )
            ),
        },
    }

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    out_path = os.path.join(out_dir, "pushing_stick_report.json")
    if artifact_writes_enabled():
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        write_provenance_sidecar(out_path, writer="test_pushing_stick.py")
    else:
        print(
            f"\n[artifacts] CERTABSTAIN_WRITE_ARTIFACTS not set; leaving "
            f"{out_path} as committed"
        )

    assert os.path.exists(out_path), (
        f"{out_path} is missing and CERTABSTAIN_WRITE_ARTIFACTS=1 was not set "
        "to generate it"
    )
    assert meets_bar, report["verdict"]["reason"]
