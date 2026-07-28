"""M5 acceptance tests: witnesses, bindings, and gate cover-membership.

Spec criteria (section 6, M5):
  * one flipped weight byte => refusal at load;
  * reference-parameter mismatch => refusal;
  * state driven out of the cover => abstention with reason
    "left certified domain";
  * all Phase 1 tests still green (verified by the rest of the suite, not
    this file).
"""

from __future__ import annotations

import numpy as np
import pytest

from certabstain import (
    ActionGate,
    CircleClearance,
    Interval,
    MLP,
    NetworkCertificateMismatch,
    PredictiveTubeWitness,
    ReferenceMismatch,
    SpringDamper2D,
    TwoSidedClaim,
    VerifiedDiscrepancyWitness,
    certify_epsilon,
    propagate_tube,
)
from certabstain.discrepancy import _batched_ibp

DOMAIN = Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
CLEAR = CircleClearance(ox=0.0, oy=0.0, r=0.15)


def _clearance_net_and_cert():
    rng = np.random.default_rng(0)
    X = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(20_000, 2))
    Y = CLEAR.value(X)
    from certabstain.nnbound import fit_mlp

    net = fit_mlp((2, 16, 16, 1), X, Y, steps=2000, seed=0)
    cert = certify_epsilon(
        net,
        CLEAR.interval_batch,
        DOMAIN,
        reference_id=CLEAR.reference_id(),
        ref_float=CLEAR.value,
        target=None,
        max_leaf_evals=80_000,
        floor_samples=200_000,
    )
    return net, cert


# ===================================================================== #
# VerifiedDiscrepancyWitness (W1): binding refusals
# ===================================================================== #


def test_bind_succeeds_with_matching_network_and_reference() -> None:
    net, cert = _clearance_net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, CLEAR.reference_id())
    assert w.violation_floor() == 0.0
    assert w.miss_probability(-0.01) == 0.0
    assert "epsilon certified" in w.justification()
    # score() delegates to Phase 1's CertifiedModelErrorWitness unchanged
    assert w.score(0.0) == pytest.approx(float(np.max(cert.eps)))


def test_bind_refuses_one_flipped_weight_byte() -> None:
    net, cert = _clearance_net_and_cert()
    tampered = [(np.array(W, copy=True), np.array(b, copy=True)) for W, b in net.weights]
    tampered[0][0][0, 0] = np.nextafter(tampered[0][0][0, 0], np.inf)  # one ulp
    other = MLP(tuple(tampered), activation=net.activation)
    with pytest.raises(NetworkCertificateMismatch):
        VerifiedDiscrepancyWitness.bind(cert, other, CLEAR.reference_id())


def test_bind_refuses_reference_parameter_mismatch() -> None:
    net, cert = _clearance_net_and_cert()
    changed = CircleClearance(ox=0.0, oy=0.0, r=0.20)  # different radius
    with pytest.raises(ReferenceMismatch):
        VerifiedDiscrepancyWitness.bind(cert, net, changed.reference_id())


def test_covers_matches_certificate_contains() -> None:
    net, cert = _clearance_net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, CLEAR.reference_id())
    inside = np.array([0.1, -0.1])
    outside = np.array([10.0, 10.0])
    assert w.covers(inside) == bool(cert.contains(inside))
    assert w.covers(outside) is False


# ===================================================================== #
# PredictiveTubeWitness (W2): L4 score and cover membership
# ===================================================================== #


def _yv_tube():
    """A small, fast (y, vy) free-flight tube -- same construction as
    test_tube.py's acceptance test, reused here for the witness layer."""
    model = SpringDamper2D()

    def step(yv, uy):
        n = yv.shape[0]
        s = np.zeros((n, 4))
        s[:, 1] = yv[:, 0] + 1.0
        s[:, 3] = yv[:, 1]
        u = np.zeros((n, 2))
        u[:, 1] = uy[:, 0]
        return model.step(s, u)[:, [1, 3]] - np.array([1.0, 0.0])

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

    rng = np.random.default_rng(5)
    domain = Interval(np.array([-0.2, -0.5, -0.3]), np.array([0.2, 0.5, 0.3]))
    from certabstain.nnbound import fit_mlp

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
    tube = propagate_tube(net, cert, X0, [U_box] * 10, n_states=2)
    return tube


def test_predictive_tube_witness_score_and_composition() -> None:
    tube = _yv_tube()
    assert tube.cover_exit_reason is None and tube.horizon == 10

    # a clearance function on (y, vy): certifiably far below any y visited,
    # so c_required is trivially cleared and the composition succeeds.
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    c_required = -5.0  # comfortably below every certified lower bound
    w = PredictiveTubeWitness.build(tube, clear.interval_batch, c_required)

    assert w.violation_floor() == 0.0
    assert "K=10" in w.justification()
    s = w.score()
    assert np.isfinite(s)
    assert s <= 0.0, "c_required was chosen to be trivially satisfied"

    claim = TwoSidedClaim.compose(threshold=-1e-9, false_alarm_bound=0.05, witness=w)
    assert claim.miss_bound == 0.0


def test_predictive_tube_witness_covers_the_certified_boxes_only() -> None:
    tube = _yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    w = PredictiveTubeWitness.build(tube, clear.interval_batch, c_required=-5.0)

    assert w.covers(np.array([0.0, 0.0]))  # inside X0
    assert not w.covers(np.array([50.0, 50.0]))  # nowhere near any tube box


# ===================================================================== #
# Gate cover-membership (the one Phase 1 change): abstains by name
# ===================================================================== #


def test_gate_abstains_with_left_certified_domain_reason() -> None:
    net, cert = _clearance_net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, CLEAR.reference_id())
    gate = ActionGate(
        threshold=0.0, false_alarm_bound=0.05, safe_action="HALT", cover=w.covers
    )

    inside = np.array([0.1, -0.1])
    d_in = gate.step(observation=inside, proposed_action=np.array([1.0]), score=-1.0)
    assert not d_in.abstained

    outside = np.array([10.0, 10.0])  # "model breakdown": driven off the domain
    d_out = gate.step(observation=outside, proposed_action=np.array([1.0]), score=-1.0)
    assert d_out.abstained
    assert d_out.reason == "left certified domain"
    assert d_out.action == "HALT"


def test_gate_without_cover_is_unaffected() -> None:
    """A gate with no cover predicate behaves exactly like Phase 1's."""
    gate = ActionGate(threshold=0.0, false_alarm_bound=0.05, safe_action="STOP")
    d = gate.step(observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0)
    assert not d.abstained


def test_gate_public_surface_unchanged_by_the_cover_addition() -> None:
    """M5 adds one constructor parameter, no new public attribute or method:
    the freeze test from Phase 1 (test_gate_exposes_no_override_path) needs
    no change, which is itself the thing worth checking here."""
    import inspect

    public = [n for n in dir(ActionGate) if not n.startswith("_")]
    assert set(public) == {"authority", "log", "miss_bound", "step", "abstention_rate"}
    sig = inspect.signature(ActionGate.__init__)
    assert "cover" in sig.parameters
    banned = {"force", "override", "strict", "unsafe", "bypass", "skip_check"}
    assert not banned & set(sig.parameters)
