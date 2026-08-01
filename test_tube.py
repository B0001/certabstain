"""M4 acceptance tests for the interval tube.

Spec criteria (section 6, M4):
  * K = 10 on spring-damper: 1e5 Monte Carlo true rollouts, zero tube escapes;
    tube-width vs K curve reported.
  * certified-horizon shrinkage on cover exit, reported rather than papered
    over.
Plus unit coverage this milestone's acceptance text doesn't itself exercise:
  * cover_contains_box's disjoint-leaf volume accounting;
  * the Grönwall/interval intersection is sound (contains the true state) and
    never looser than either bound alone, on a closed-form construction where
    the true sup-gap is exact (mirrors test_discrepancy.py's dim-4 exactness
    check, extended across a multi-step tube);
  * clearance_lower_bounds, the module's other explicit M4 responsibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from certabstain import (
    EpsilonCertificate,
    HorizonTooShort,
    Interval,
    MLP,
    NetworkCertificateMismatch,
    clearance_lower_bounds,
    cover_contains_box,
    propagate_tube,
)
from certabstain.discrepancy import _batched_ibp, certify_epsilon
from certabstain.nnbound import fit_mlp
from certabstain.reference import CircleClearance, SpringDamper2D


def _fake_cert(cover_lo: np.ndarray, cover_hi: np.ndarray) -> EpsilonCertificate:
    """A minimal certificate carrying only what cover_contains_box reads."""
    d = cover_lo.shape[1] if cover_lo.ndim == 2 else 0
    return EpsilonCertificate(
        eps=np.zeros(1),
        domain_lo=np.zeros(d),
        domain_hi=np.ones(d),
        cover_lo=cover_lo,
        cover_hi=cover_hi,
        cover_fraction=1.0,
        net_hash="unused",
        reference_id="unused",
        empirical_floor=np.zeros(1),
        n_leaf_evals=0,
        n_leaves=cover_lo.shape[0],
        target=None,
    )


# ===================================================================== #
# cover_contains_box: disjoint-leaf volume accounting
# ===================================================================== #


def test_cover_contains_box_single_leaf() -> None:
    cert = _fake_cert(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    assert cover_contains_box(cert, Interval(np.array([0.2, 0.2]), np.array([0.8, 0.8])))
    assert not cover_contains_box(cert, Interval(np.array([0.2, 0.2]), np.array([1.2, 0.8])))


def test_cover_contains_box_tiled_leaves_no_gap() -> None:
    # two leaves tiling [0,2] x [0,1] exactly; a box spanning both is covered
    cert = _fake_cert(
        np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 1.0], [2.0, 1.0]])
    )
    assert cover_contains_box(cert, Interval(np.array([0.3, 0.1]), np.array([1.7, 0.9])))


def test_cover_contains_box_gap_between_leaves_fails_closed() -> None:
    # a genuine gap in [0.9, 1.1]: a box straddling it must NOT be covered
    cert = _fake_cert(
        np.array([[0.0, 0.0], [1.1, 0.0]]), np.array([[0.9, 1.0], [2.0, 1.0]])
    )
    assert cover_contains_box(cert, Interval(np.array([0.1, 0.1]), np.array([0.8, 0.9])))
    assert not cover_contains_box(cert, Interval(np.array([0.5, 0.1]), np.array([1.5, 0.9])))


def test_cover_contains_box_empty_cover() -> None:
    cert = _fake_cert(np.zeros((0, 2)), np.zeros((0, 2)))
    assert not cover_contains_box(cert, Interval(np.array([0.0, 0.0]), np.array([0.1, 0.1])))
    # a zero-volume (degenerate) box is trivially covered regardless
    assert cover_contains_box(cert, Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))


# ===================================================================== #
# Closed-form exactness: a bias-shifted twin, sup-gap known in closed form
# ===================================================================== #


def _dim4_dynamics_cert(
    delta: float = 0.02,
    k_evals: int = 20_000,
    x0_half: float = 0.01,
    lam: float = 0.4,
):
    """A single-affine-layer "net" -- f(x) = lam*x + b, a genuine contraction
    (Lipschitz = lam < 1) -- composed with a bias-shifted twin of itself, so
    the true sup-gap is exactly ``lam*width(box) + delta`` in closed form
    (IBP is exact for a diagonal affine map, zero dependency-problem slack).
    Contraction keeps both the point trajectory and the interval tube's width
    converging to a bounded steady state, so a small fixed domain comfortably
    contains every step without the empirical-padding guesswork a general
    (non-contractive) net would need."""
    W = lam * np.eye(4)
    b = np.array([0.01, -0.01, 0.005, -0.005])
    net = MLP(((W, b),))
    deltas = np.array([delta, -delta, 0.5 * delta, -0.5 * delta])
    twin = MLP(((W, b + deltas),))

    def ref(lo, hi):
        return _batched_ibp(twin, lo, hi)

    domain = Interval(-0.2 * np.ones(4), 0.2 * np.ones(4))
    cert = certify_epsilon(
        net,
        ref,
        domain,
        reference_id="linear contraction + bias-shifted twin (tube test)",
        ref_float=twin.forward,
        target=4 * delta,  # loose: only a handful of splits needed to clear it
        max_leaf_evals=k_evals,
        floor_samples=20_000,
    )
    X0 = Interval(-x0_half * np.ones(4), x0_half * np.ones(4))
    return net, twin, cert, X0


def test_tube_intersection_sound_and_tighter_than_either_bound() -> None:
    K = 6
    net, twin, cert, X0 = _dim4_dynamics_cert()
    controls = [Interval(np.zeros(0), np.zeros(0)) for _ in range(K)]

    tube = propagate_tube(net, cert, X0, controls, n_states=4)
    assert tube.horizon == K
    assert tube.cover_exit_reason is None

    # the reported box is never wider than either sound alternative
    for t in range(1, K + 1):
        combined_w = tube.widths[t]
        assert np.all(combined_w <= tube.gronwall_widths[t] + 1e-12)

    # Monte Carlo soundness: the TRUE reference trajectory (the twin, exactly
    # -- this synthetic construction's "trusted reference") must never
    # escape the reported boxes.
    rng = np.random.default_rng(4)
    x = rng.uniform(X0.lo, X0.hi, size=(20_000, 4))
    escapes = 0
    for t in range(K):
        box = tube.boxes[t]
        ok = (box.lo[None, :] <= x) & (x <= box.hi[None, :])
        escapes += int(ok.size - np.count_nonzero(ok))
        x = twin.forward(x)
    box = tube.boxes[K]
    ok = (box.lo[None, :] <= x) & (x <= box.hi[None, :])
    escapes += int(ok.size - np.count_nonzero(ok))
    assert escapes == 0, f"{escapes} true-trajectory escapes out of the certified tube"


def test_tube_refuses_mismatched_network() -> None:
    net, twin, cert, X0 = _dim4_dynamics_cert()
    other = MLP.random((4, 12, 4), rng=np.random.default_rng(999))
    with pytest.raises(NetworkCertificateMismatch):
        propagate_tube(other, cert, X0, [], n_states=4)


def _dim4_control_cert(seed: int = 31, k_evals: int = 80_000):
    """Same closed-form construction as _dim4_dynamics_cert, but with a real
    (state, control) input split, for tests that need a control dimension."""
    rng = np.random.default_rng(seed)
    net = MLP.random((5, 12, 4), rng=rng, scale=0.5)  # 4 states + 1 control -> 4 states
    delta = 0.02
    W2 = [(np.array(W, copy=True), np.array(b, copy=True)) for W, b in net.weights]
    W2[-1] = (W2[-1][0], W2[-1][1] + delta)
    twin = MLP(tuple(W2), activation=net.activation)

    def ref(lo, hi):
        return _batched_ibp(twin, lo, hi)

    domain = Interval(-0.1 * np.ones(5), 0.1 * np.ones(5))
    cert = certify_epsilon(
        net,
        ref,
        domain,
        reference_id="bias-shifted twin w/ control (tube test)",
        ref_float=twin.forward,
        target=None,
        max_leaf_evals=k_evals,
        floor_samples=50_000,
    )
    return net, twin, cert


def test_tube_horizon_shrinks_and_reports_reason_on_cover_exit() -> None:
    """A control box outside the certified domain forces an immediate cover
    exit: horizon 0, reported by name -- not a silently truncated success."""
    net, twin, cert = _dim4_control_cert()
    X0 = Interval(-0.01 * np.ones(4), 0.01 * np.ones(4))
    # certified control domain is [-0.1, 0.1]; this box is far outside it
    wide_control = Interval(np.array([-5.0]), np.array([5.0]))
    tube = propagate_tube(net, cert, X0, [wide_control] * 3, n_states=4)
    assert tube.horizon == 0
    assert tube.requested_horizon == 3
    assert tube.cover_exit_reason is not None
    assert "before step 0" in tube.cover_exit_reason
    assert tube.boxes == (X0,)


def test_tube_refuses_when_certified_horizon_is_below_the_declared_one() -> None:
    """Spec 7.8, second clause. Shrinking the horizon is only half the refusal:
    a deployment that declared it needs K steps and got fewer must be refused,
    not handed a short TubeResult whose `horizon` field it might not read."""
    net, twin, cert = _dim4_control_cert()
    X0 = Interval(-0.01 * np.ones(4), 0.01 * np.ones(4))
    wide_control = Interval(np.array([-5.0]), np.array([5.0]))

    # Same cover exit as the test above -- but this caller declared it needs 3.
    with pytest.raises(HorizonTooShort) as exc:
        propagate_tube(
            net, cert, X0, [wide_control] * 3, n_states=4, required_horizon=3
        )
    msg = str(exc.value)
    assert "certified horizon is 0" in msg      # what was achieved
    assert "requires 3" in msg                  # what was declared
    assert "before step 0" in msg               # why it fell short, carried through

    # ...and the shortfall is the only reason it refuses. This construction's
    # tube widens out of the cover after one step even under an in-cover
    # control, so a caller declaring 1 gets its result -- note horizon (1) is
    # still short of requested_horizon (3), proving the refusal keys off what
    # the deployment *declared it needs*, not off the requested length.
    ok_control = Interval(np.array([-0.01]), np.array([0.01]))
    tube = propagate_tube(
        net, cert, X0, [ok_control] * 3, n_states=4, required_horizon=1
    )
    assert tube.horizon == 1
    assert tube.requested_horizon == 3
    assert tube.cover_exit_reason is not None


def test_tube_refuses_a_required_horizon_it_was_given_no_controls_for() -> None:
    """Declaring a horizon longer than the supplied control sequence is
    unreachable by construction -- refused up front, not after the walk."""
    net, twin, cert = _dim4_control_cert()
    X0 = Interval(-0.01 * np.ones(4), 0.01 * np.ones(4))
    ok_control = Interval(np.array([-0.01]), np.array([0.01]))
    with pytest.raises(HorizonTooShort, match="unreachable by construction"):
        propagate_tube(net, cert, X0, [ok_control] * 2, n_states=4, required_horizon=5)


def test_tube_default_still_shrinks_rather_than_refusing() -> None:
    """The refusal is opt-in. Sweeps that *measure* where the horizon collapses
    (tube_sweep.py) must keep getting a truncated result, not an exception."""
    net, twin, cert = _dim4_control_cert()
    X0 = Interval(-0.01 * np.ones(4), 0.01 * np.ones(4))
    wide_control = Interval(np.array([-5.0]), np.array([5.0]))
    tube = propagate_tube(net, cert, X0, [wide_control] * 3, n_states=4)
    assert tube.horizon == 0 and tube.cover_exit_reason is not None


# ===================================================================== #
# The headline acceptance run: SpringDamper2D, K = 10, 1e5 MC rollouts
# ===================================================================== #


def _yv_subsystem(model: SpringDamper2D):
    """The (y, vy, uy) -> (y', vy') slice of SpringDamper2D.

    x and vx never appear in the equations for y, vy (the contact force
    depends only on y, vy; there's no cross-coupling), so this slice is the
    model's full dynamics restricted to the two components the guard and the
    contact force actually act on -- exactly the "vy' component (where
    contact acts)" stiffness_sweep.py (M2) already singles out as the
    interesting one. x0=1.0 keeps the free-flight branch (y+1 > 0) selected
    regardless of the y-perturbation tested here.
    """

    def step(yv: np.ndarray, uy: np.ndarray) -> np.ndarray:
        n = yv.shape[0]
        s = np.zeros((n, 4))
        s[:, 1] = yv[:, 0] + 1.0
        s[:, 3] = yv[:, 1]
        u = np.zeros((n, 2))
        u[:, 1] = uy[:, 0]
        out = model.step(s, u)
        return out[:, [1, 3]] - np.array([1.0, 0.0])

    def step_interval(lo: np.ndarray, hi: np.ndarray):
        n = lo.shape[0]
        zero = np.zeros(n)
        S = Interval(
            np.stack([zero, lo[:, 0] + 1.0, zero, lo[:, 1]], axis=1),
            np.stack([zero, hi[:, 0] + 1.0, zero, hi[:, 1]], axis=1),
        )
        U = Interval(
            np.stack([zero, lo[:, 2]], axis=1), np.stack([zero, hi[:, 2]], axis=1)
        )
        enc = model.step_interval(S, U)
        offset = np.array([1.0, 0.0])
        return enc.lo[:, [1, 3]] - offset, enc.hi[:, [1, 3]] - offset

    return step, step_interval


def test_tube_acceptance_spring_damper() -> None:
    # The (y, vy) subsystem, free-flight branch: exactly affine, so a small
    # (single-hidden-layer) net fits it tightly. A wider/deeper net was tried
    # first and produced a much looser CROWN enclosure per layer -- compounded
    # over K=10 steps that "wrapping effect" alone made the tube blow up long
    # before any real escape would occur, independent of how good the
    # discrepancy certificate was. The genuinely nonlinear in-contact/
    # stiffness regime, where that conservatism is the whole point, is swept
    # separately in tube_sweep.py (mirroring stiffness_sweep.py, M2's own
    # precedent: that study is a published artifact script, not part of the
    # fast pytest acceptance suite).
    model = SpringDamper2D()
    step, step_interval = _yv_subsystem(model)
    rng = np.random.default_rng(5)

    domain = Interval(np.array([-0.2, -0.5, -0.3]), np.array([0.2, 0.5, 0.3]))
    n_train = 100_000
    X = rng.uniform(domain.lo, domain.hi, size=(n_train, 3))
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
    assert cert.matches_network(net)
    assert cert.cover_fraction > 0.90

    X0 = Interval(np.array([-0.02, -0.02]), np.array([0.02, 0.02]))
    U_box = Interval(np.array([-0.02]), np.array([0.02]))
    K = 10
    tube = propagate_tube(net, cert, X0, [U_box] * K, n_states=2)

    assert tube.cover_exit_reason is None, tube.cover_exit_reason
    assert tube.horizon == K

    # 1e5 Monte Carlo TRUE rollouts (the float reference model, not the net):
    # zero tube escapes, spec's headline M4 acceptance number.
    n_rollouts = 100_000
    x = rng.uniform(X0.lo, X0.hi, size=(n_rollouts, 2))
    escapes_per_step = np.zeros(K + 1, dtype=int)
    box0 = tube.boxes[0]
    ok = (box0.lo[None, :] <= x) & (x <= box0.hi[None, :])
    escapes_per_step[0] = ok.size - np.count_nonzero(ok)
    for t in range(K):
        u = rng.uniform(U_box.lo, U_box.hi, size=(n_rollouts, 1))
        x = step(x, u)
        box = tube.boxes[t + 1]
        ok = (box.lo[None, :] <= x) & (x <= box.hi[None, :])
        escapes_per_step[t + 1] = ok.size - np.count_nonzero(ok)

    total_escapes = int(escapes_per_step.sum())
    assert total_escapes == 0, (
        f"{total_escapes} tube escapes across {n_rollouts} Monte Carlo rollouts "
        f"(per-step counts: {escapes_per_step.tolist()})"
    )

    # tube-width vs K, reported (not merely asserted): spec wants this curve
    # published, mirroring stiffness_sweep.py's "publish the curve as-is".
    widths = tube.widths  # (K+1, 2): [y, vy]
    print("\n[M4 tube widths, (y, vy)] " + " ".join(
        f"K={t}:{np.array2string(widths[t], precision=3)}" for t in range(K + 1)
    ))
    assert np.all(np.isfinite(widths))

    # clearance evaluation (this module's other explicit M4 responsibility):
    # reuse CircleClearance over (y, vy) as a stand-in 2-D clearance function.
    clear = CircleClearance(ox=10.0, oy=10.0, r=0.15)  # far away: plumbing only
    lo_bounds = clearance_lower_bounds(tube, clear.interval_batch)
    assert lo_bounds.shape == (K + 1,)
    assert np.all(np.isfinite(lo_bounds))


def test_tube_refuses_a_nonsensical_required_horizon() -> None:
    """0 and negatives silently behaved as None: `horizon < required_horizon`
    is false for both, so a caller computing `k - 1` and landing on 0 got no
    refusal at all. Validated up front, before any other check."""
    net, twin, cert = _dim4_control_cert()
    X0 = Interval(-0.01 * np.ones(4), 0.01 * np.ones(4))
    ok_control = Interval(np.array([-0.01]), np.array([0.01]))
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError, match="positive whole number"):
            propagate_tube(
                net, cert, X0, [ok_control] * 3, n_states=4, required_horizon=bad
            )
