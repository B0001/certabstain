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

from functools import lru_cache

import numpy as np
import pytest

from certabstain import (
    ActionGate,
    CircleClearance,
    HorizonTooShort,
    Interval,
    MLP,
    NetworkCertificateMismatch,
    PredictiveTubeWitness,
    ReferenceMismatch,
    SpringDamper2D,
    TwoSidedClaim,
    VerifiedDiscrepancyWitness,
    build_monitor,
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


def test_direct_construction_bypassing_bind_is_rejected() -> None:
    """bind() is documented as the only way to build a VerifiedDiscrepancyWitness,
    but its A4/A1 checks need `net`/`reference_id`, which are not stored
    fields -- confirmed empirically that the raw constructor previously built
    a witness bound to a fake certificate whose matches_network() would have
    returned False, skipping the checks entirely. The constructor now
    requires an unexported bind() token."""
    net, cert = _clearance_net_and_cert()
    from certabstain.soundness import CertifiedModelErrorWitness

    with pytest.raises(TypeError, match="bind"):
        VerifiedDiscrepancyWitness(
            certificate=cert, inner=CertifiedModelErrorWitness(epsilon=0.05)
        )
    # bind() itself is unaffected.
    w = VerifiedDiscrepancyWitness.bind(cert, net, CLEAR.reference_id())
    assert w.certificate is cert


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


@lru_cache(maxsize=1)
def _yv_net_and_cert():
    """The (y, vy) net and its certificate -- same construction as
    test_tube.py's acceptance test, reused here for the witness layer.

    Cached: certifying this costs 400k leaf evaluations and several tests in
    this file need it, none of which mutate what they get back.
    """
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
    return net, cert


def _yv_tube():
    """The full-horizon tube: 10 steps requested, 10 certified."""
    net, cert = _yv_net_and_cert()
    X0 = Interval(np.array([-0.02, -0.02]), np.array([0.02, 0.02]))
    U_box = Interval(np.array([-0.02]), np.array([0.02]))
    return propagate_tube(net, cert, X0, [U_box] * 10, n_states=2)


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


def test_w2_justification_names_both_references_spec_a1() -> None:
    """W1 names the reference it was proven against; W2 named neither.

    Two witnesses built on the same tube against completely different obstacle
    geometries produced byte-identical justification strings -- the audit log
    could not tell which geometry the clearance claim was proved for, and the
    dynamics reference was dropped by TubeResult before the witness could see
    it. propagate_tube enforces the network binding (A4); the reference binding
    (A1) had nowhere to survive on this path.
    """
    tube = _yv_tube()

    near = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    far = CircleClearance(ox=3.0, oy=-42.0, r=1.25)   # a different obstacle
    w_near = PredictiveTubeWitness.build(tube, near.interval_batch, -5.0)
    w_far = PredictiveTubeWitness.build(tube, far.interval_batch, -5.0)

    assert w_near.justification() != w_far.justification(), (
        "two different obstacle geometries must not produce the same audit string"
    )
    assert near.reference_id() in w_near.justification()
    assert far.reference_id() in w_far.justification()

    # the dynamics reference the tube was certified against is named too
    assert tube.reference_id in w_near.justification()
    assert "SpringDamper2D" in w_near.justification()


def test_w2_records_an_anonymous_clearance_as_undeclared() -> None:
    """A bare function has no identity; say so rather than imply one.

    Same idiom the horizon already uses -- ``required_horizon=None`` reads as
    "best effort (no horizon requirement declared)" rather than passing for a
    met requirement. An unnamed geometry must not read as a named one.
    """
    tube = _yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)

    anon = PredictiveTubeWitness.build(
        tube, lambda lo, hi: clear.interval_batch(lo, hi), -5.0
    )
    assert anon.clearance_id is None
    assert "undeclared clearance geometry" in anon.justification()

    # ...and a caller passing a plain function can still declare one
    named = PredictiveTubeWitness.build(
        tube, lambda lo, hi: clear.interval_batch(lo, hi), -5.0,
        clearance_id="hand-rolled clearance v3",
    )
    assert "hand-rolled clearance v3" in named.justification()
    assert "undeclared" not in named.justification()


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


def test_build_monitor_wires_the_cover_check_spec_7_7() -> None:
    """Spec 7.7 must hold on the *documented* path, not just a hand-built gate.

    The test above builds its ActionGate directly with ``cover=w.covers``, and
    so did demo.py and the driver skill -- every caller that actually got the
    check bypassed build_monitor to get it. build_monitor itself dropped the
    predicate, so anyone following the one-call route in the package docstring
    got a monitor that scored happily outside the domain its claim was proved
    over: a guarantee silently evaluated where it does not hold.
    """
    net, cert = _clearance_net_and_cert()
    w = VerifiedDiscrepancyWitness.bind(cert, net, CLEAR.reference_id())

    rng = np.random.default_rng(0)
    trajs = [list(w.score(rng.uniform(0.5, 1.5, 20))) for _ in range(400)]
    monitor = build_monitor(
        nominal_trajectories=trajs, alpha=0.05, witness=w, safe_action="HALT"
    )

    inside = np.array([0.1, -0.1])
    assert w.covers(inside)
    d_in = monitor.step(
        observation=inside, proposed_action=np.array([1.0]), score=-1.0
    )
    assert not d_in.abstained

    outside = np.array([10.0, 10.0])
    assert not w.covers(outside)
    d_out = monitor.step(
        observation=outside, proposed_action=np.array([1.0]), score=-1.0
    )
    assert d_out.abstained
    assert d_out.reason == "left certified domain"
    assert d_out.action == "HALT"


def test_build_monitor_leaves_cover_unset_for_a_witness_without_one() -> None:
    """Wiring 7.7 must not invent a cover for witnesses that have no domain.

    CertifiedModelErrorWitness carries a global epsilon, not a certified
    region; there is nothing to be outside of. The gate must stay cover-free
    rather than abstaining on everything.
    """
    from certabstain import CertifiedModelErrorWitness

    w = CertifiedModelErrorWitness(epsilon=0.1)
    rng = np.random.default_rng(0)
    trajs = [list(w.score(rng.uniform(0.5, 1.5, 20))) for _ in range(400)]
    monitor = build_monitor(
        nominal_trajectories=trajs, alpha=0.05, witness=w, safe_action="HALT"
    )
    d = monitor.step(
        observation=np.array([1e6, 1e6]), proposed_action=np.array([1.0]), score=-1.0
    )
    assert not d.abstained


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
    assert set(public) == {"verifier", "log", "miss_bound", "step", "abstention_rate"}
    sig = inspect.signature(ActionGate.__init__)
    assert "cover" in sig.parameters
    # Allowlist, not denylist -- see test_gate_exposes_no_override_path for why.
    assert set(sig.parameters) == {
        "self", "threshold", "false_alarm_bound", "safe_action",
        "claim", "cover",
    }


# ===================================================================== #
# W2 horizon requirement: spec 7.8 declared where a deployment declares it
# ===================================================================== #


def _short_yv_tube():
    """The same (y, vy) construction, but with a control box far outside the
    certified cover so the tube exits immediately: horizon 0 of 3 requested."""
    net, cert = _yv_net_and_cert()
    X0 = Interval(np.array([-0.02, -0.02]), np.array([0.02, 0.02]))
    wide = Interval(np.array([-5.0]), np.array([5.0]))
    tube = propagate_tube(net, cert, X0, [wide] * 3, n_states=2)
    assert tube.horizon == 0 and tube.requested_horizon == 3
    return tube


def test_witness_refuses_a_tube_shorter_than_the_requested_horizon() -> None:
    """The default is strict: propagate K control boxes and you are taken to
    need K steps, so a collapsed tube refuses without the caller saying
    anything. This is the case that previously built a one-step witness
    wearing the K-step interface."""
    tube = _short_yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    with pytest.raises(HorizonTooShort) as exc:
        PredictiveTubeWitness.build(tube, clear.interval_batch, c_required=-5.0)
    msg = str(exc.value)
    assert "need 3" in msg and "certifies 0" in msg
    assert "before step 0" in msg          # the cover-exit reason, carried through


def test_witness_accepts_a_shorter_horizon_when_the_deployment_declares_one() -> None:
    """A deployment that genuinely needs fewer steps than it propagated says
    so explicitly, and gets a witness scored over the horizon it declared."""
    tube = _yv_tube()                       # horizon 10 of 10
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    w = PredictiveTubeWitness.build(
        tube, clear.interval_batch, c_required=-5.0, required_horizon=4
    )
    assert w.required_horizon == 4
    assert "declared requirement 4, met" in w.justification()


def test_witness_best_effort_is_opt_in_and_shows_up_in_the_audit_string() -> None:
    """Explicit None is the old behaviour. It is still available -- studies and
    exploratory work need it -- but it must be asked for, and the weaker claim
    is stated in the justification rather than implied by silence."""
    tube = _short_yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    w = PredictiveTubeWitness.build(
        tube, clear.interval_batch, c_required=-5.0, required_horizon=None
    )
    assert w.required_horizon is None
    assert "best effort" in w.justification()
    assert "K=0" in w.justification()


def test_witness_refuses_a_requirement_longer_than_the_tube_was_propagated() -> None:
    tube = _yv_tube()                       # 10 requested
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    with pytest.raises(HorizonTooShort, match="unreachable by construction"):
        PredictiveTubeWitness.build(
            tube, clear.interval_batch, c_required=-5.0, required_horizon=11
        )


def test_witness_refuses_a_nonsensical_requirement() -> None:
    tube = _yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)
    for bad in (0, -2, 3.5, True, False):
        with pytest.raises(ValueError, match="positive whole number"):
            PredictiveTubeWitness.build(
                tube, clear.interval_batch, c_required=-5.0, required_horizon=bad
            )


def test_direct_construction_bypassing_build_still_enforces_horizon() -> None:
    """build() is documented as the only way to construct a
    PredictiveTubeWitness, but the horizon check only needs stored fields
    (tube, required_horizon) -- confirmed empirically that the raw
    constructor previously built a witness with tube.horizon=0 <
    required_horizon=3 and a justification() that falsely claimed the
    requirement was met. __post_init__ now re-runs the same check build()
    used to run."""
    tube = _short_yv_tube()
    with pytest.raises(HorizonTooShort, match="certifies 0"):
        PredictiveTubeWitness(
            tube=tube,
            clearance_lo=np.zeros(1),
            c_required=-5.0,
            required_horizon=3,
            clearance_id=None,
        )
    # A requirement the tube actually meets still constructs directly.
    w = PredictiveTubeWitness(
        tube=tube, clearance_lo=np.zeros(1), c_required=-5.0,
        required_horizon=None, clearance_id=None,
    )
    assert w.required_horizon is None


def test_horizon_refusal_fails_closed_inside_the_gate() -> None:
    """HorizonTooShort is a CertAbstainError, so if one ever surfaces on the
    actuation path -- a cover predicate that re-derives a tube, say -- the gate
    abstains rather than letting the exception escape past it."""
    tube = _short_yv_tube()
    clear = CircleClearance(ox=0.0, oy=-10.0, r=0.05)

    def exploding_cover(_obs):
        PredictiveTubeWitness.build(tube, clear.interval_batch, c_required=-5.0)
        return True

    gate = ActionGate(
        threshold=0.0, false_alarm_bound=0.05, safe_action="STOP",
        cover=exploding_cover,
    )
    d = gate.step(
        observation=np.zeros(2), proposed_action=np.ones(1), score=-1.0
    )
    assert d.abstained and d.action == "STOP"
    assert "certificate refused" in d.reason and "lookahead" in d.reason
