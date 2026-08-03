"""Tests.

The coverage tests are Monte Carlo *validations of the theorems*, not smoke
tests. If the finite-sample bounds in conformal.py are wrong, these fail.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from certabstain import (
    ActionGate,
    CertificateAuthority,
    CertificateVerifier,
    CertifiedModelErrorWitness,
    EmpiricalPowerWitness,
    ForgedCertificate,
    GroupTooSmall,
    InsufficientCalibrationData,
    MarginWitness,
    MondrianCalibrator,
    ShiftBudgetExceeded,
    SoundnessNotEstablished,
    SplitConformalCalibrator,
    TwoSidedClaim,
    build_monitor,
    kl_worst_case,
    min_calibration_size,
    robust_level,
    rollout_scores,
)

RNG = np.random.default_rng(20260725)


# ===================================================================== #
# 1. The conformal coverage theorem, validated by simulation
# ===================================================================== #


@pytest.mark.parametrize("alpha,n", [(0.05, 199), (0.10, 99), (0.02, 249)])
def test_split_conformal_coverage_is_sandwiched(alpha: int, n: int) -> None:
    """Empirical coverage must land in [1-alpha, 1-alpha+1/(n+1)]."""
    trials = 20_000
    hits = 0
    for _ in range(trials):
        sample = RNG.standard_exponential(n + 1)
        cal = SplitConformalCalibrator.fit(sample[:n], alpha)
        hits += sample[n] <= cal.threshold

    empirical = hits / trials
    se = math.sqrt(empirical * (1 - empirical) / trials)
    lower = 1 - alpha
    upper = 1 - alpha + 1 / (n + 1)

    assert empirical >= lower - 4 * se, f"coverage {empirical} below guarantee {lower}"
    assert empirical <= upper + 4 * se, f"coverage {empirical} above ceiling {upper}"


def test_coverage_holds_for_heavy_tails_and_discrete_scores() -> None:
    """Distribution-free means distribution-free. No moments assumed."""
    alpha, n, trials = 0.05, 199, 8_000
    for draw in (
        lambda k: RNG.standard_cauchy(k),
        lambda k: RNG.integers(0, 5, k).astype(float),
        lambda k: np.exp(RNG.standard_normal(k) * 8),
    ):
        hits = 0
        for _ in range(trials):
            sample = draw(n + 1)
            cal = SplitConformalCalibrator.fit(sample[:n], alpha)
            hits += sample[n] <= cal.threshold
        empirical = hits / trials
        se = math.sqrt(max(empirical * (1 - empirical), 1e-9) / trials)
        assert empirical >= (1 - alpha) - 4 * se


def test_rollout_level_guarantee_is_not_per_timestep() -> None:
    """Calibrating on the rollout max controls whole-rollout false alarms."""
    alpha, n, horizon, trials = 0.05, 199, 50, 4_000
    fired = 0
    for _ in range(trials):
        cal_trajs = [RNG.standard_normal(horizon) for _ in range(n)]
        test_traj = RNG.standard_normal(horizon)
        cal = SplitConformalCalibrator.fit(rollout_scores(cal_trajs), alpha)
        fired += bool(np.any(test_traj > cal.threshold))

    rate = fired / trials
    se = math.sqrt(rate * (1 - rate) / trials)
    assert rate <= alpha + 4 * se, (
        f"rollout-level false-alarm rate {rate:.4f} exceeded alpha={alpha}; "
        "a per-timestep calibration would have produced roughly "
        f"{1 - (1 - alpha) ** horizon:.2f} here"
    )


# ===================================================================== #
# 2. Refusals: the library must decline rather than degrade
# ===================================================================== #


def test_refuses_when_calibration_sample_too_small() -> None:
    with pytest.raises(InsufficientCalibrationData, match="at least"):
        SplitConformalCalibrator.fit(RNG.standard_normal(50), alpha=0.01)


def test_min_calibration_size_is_exact() -> None:
    for alpha in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001):
        n = min_calibration_size(alpha)
        SplitConformalCalibrator.fit(RNG.standard_normal(n), alpha)  # must succeed
        with pytest.raises(InsufficientCalibrationData):
            SplitConformalCalibrator.fit(RNG.standard_normal(n - 1), alpha)


def test_refuses_shift_budget_at_or_above_alpha() -> None:
    with pytest.raises(ShiftBudgetExceeded):
        robust_level(alpha=0.05, rho=0.05)
    with pytest.raises(ShiftBudgetExceeded):
        robust_level(alpha=0.05, rho=0.20)
    assert robust_level(alpha=0.05, rho=0.02) == pytest.approx(0.03)


def test_refuses_a_non_finite_shift_budget() -> None:
    """A nan budget is not a loose budget, it is no budget.

    Both guards in robust_level are comparisons -- `rho < 0.0` and
    `rho >= alpha` -- and both are False for nan, so a nan sailed through and
    returned `alpha - nan` = nan. Via build_monitor that nan did eventually
    raise, but from _check_alpha inside the calibrator, blaming `alpha` for a
    bad shift_budget. Called directly, robust_level just returned nan.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            robust_level(alpha=0.05, rho=bad)

    # the real budgets still behave
    assert robust_level(alpha=0.05, rho=0.02) == pytest.approx(0.03)
    with pytest.raises(ShiftBudgetExceeded):
        robust_level(alpha=0.05, rho=0.05)


def test_refuses_two_sided_without_witness() -> None:
    with pytest.raises(SoundnessNotEstablished, match="no soundness witness"):
        build_monitor(
            nominal_trajectories=[[-1.0] * 10 for _ in range(200)],
            alpha=0.05,
            safe_action="STOP",
        )


def test_refuses_two_sided_when_threshold_does_not_clear_floor() -> None:
    """Nominal rollouts that skim the constraint cannot support a miss bound."""
    trajs = [list(RNG.uniform(-0.2, 0.4, 20)) for _ in range(400)]  # max > 0 a.s.
    with pytest.raises(SoundnessNotEstablished, match="does not clear"):
        build_monitor(
            nominal_trajectories=trajs,
            alpha=0.05,
            witness=MarginWitness(),
            safe_action="STOP",
        )


def test_mondrian_refuses_understocked_strata() -> None:
    scores = RNG.standard_normal(300)
    groups = ["a"] * 250 + ["b"] * 50
    # alpha=0.01 needs 99 per stratum; "b" has 50.
    with pytest.raises(GroupTooSmall, match="understocked"):
        MondrianCalibrator.fit(scores, groups, alpha=0.01)
    # alpha=0.02 needs only 49, so the same data is sufficient.
    MondrianCalibrator.fit(scores, groups, alpha=0.02)


def test_mondrian_refuses_unseen_stratum_at_deployment() -> None:
    cal = MondrianCalibrator.fit(
        RNG.standard_normal(400), ["a"] * 200 + ["b"] * 200, alpha=0.05
    )
    with pytest.raises(GroupTooSmall, match="not present at calibration"):
        cal.fires(0.0, "c")


# ===================================================================== #
# 3. Group-conditional coverage really is conditional
# ===================================================================== #


def test_mondrian_gives_per_group_coverage_where_marginal_fails() -> None:
    """A pooled threshold under-covers the heavy-tailed stratum; Mondrian does not."""
    alpha, per_group, trials = 0.10, 99, 3_000
    pooled_miss = mondrian_miss = 0

    for _ in range(trials):
        easy = RNG.standard_normal(per_group) * 0.1
        hard = RNG.standard_normal(per_group) * 3.0
        scores = np.concatenate([easy, hard])
        groups = ["easy"] * per_group + ["hard"] * per_group

        pooled = SplitConformalCalibrator.fit(scores, alpha)
        mond = MondrianCalibrator.fit(scores, groups, alpha)

        probe = RNG.standard_normal() * 3.0  # a "hard" test point
        pooled_miss += probe > pooled.threshold
        mondrian_miss += probe > mond.calibrator_for("hard").threshold

    pooled_rate = pooled_miss / trials
    mondrian_rate = mondrian_miss / trials
    se = math.sqrt(alpha * (1 - alpha) / trials)

    assert mondrian_rate <= alpha + 4 * se, "conditional guarantee violated"
    assert pooled_rate > alpha + 4 * se, (
        "expected the pooled threshold to under-cover the hard stratum; "
        f"got {pooled_rate:.4f}"
    )


# ===================================================================== #
# 4. Shift robustness under an adversarial, budget-saturating shift
# ===================================================================== #


def test_claim_survives_worst_case_tv_shift() -> None:
    """Deployment mixes in a budget-sized adversarial component that always fires."""
    alpha, rho, n, trials = 0.10, 0.03, 999, 20_000
    inner = robust_level(alpha, rho, divergence="tv")
    assert inner == pytest.approx(alpha - rho)

    fired = 0
    for _ in range(trials):
        cal = SplitConformalCalibrator.fit(RNG.standard_normal(n), inner)
        if RNG.random() < rho:
            probe = np.inf  # adversarial mass, guaranteed to trip the monitor
        else:
            probe = RNG.standard_normal()
        fired += probe > cal.threshold

    rate = fired / trials
    se = math.sqrt(rate * (1 - rate) / trials)
    assert rate <= alpha + 4 * se, f"shift-robust claim violated: {rate:.4f} > {alpha}"


def test_kl_worst_case_is_monotone_and_tight_at_zero() -> None:
    assert kl_worst_case(0.05, 0.0) == pytest.approx(0.05)
    prev = 0.05
    for rho in (0.001, 0.01, 0.05, 0.2, 1.0):
        cur = kl_worst_case(0.05, rho)
        assert cur >= prev
        prev = cur
    assert kl_worst_case(0.05, 50.0) == pytest.approx(1.0, abs=1e-6)


def test_kl_robust_level_carries_the_claim() -> None:
    alpha, rho = 0.10, 0.005
    inner = robust_level(alpha, rho, divergence="kl")
    assert 0 < inner < alpha
    assert kl_worst_case(inner, rho) <= alpha + 1e-9


# ===================================================================== #
# 5. Two-sided composition: zero misses, by construction
# ===================================================================== #


def test_certified_model_error_witness_catches_every_violation() -> None:
    """No violating state slips past a threshold that clears the floor."""
    eps = 0.05
    witness = CertifiedModelErrorWitness(epsilon=eps)

    # Nominal rollouts keep a healthy margin: g_hat well above epsilon.
    trajs = [
        list(witness.score(RNG.uniform(0.5, 1.5, 20))) for _ in range(400)
    ]
    monitor = build_monitor(
        nominal_trajectories=trajs,
        alpha=0.05,
        witness=witness,
        safe_action="STOP",
    )
    assert monitor.threshold < 0.0
    assert monitor.claim is not None and monitor.claim.miss_bound == 0.0

    # Now feed genuinely unsafe states (true g < 0). Model error can be as
    # adverse as epsilon in either direction; the monitor must still fire.
    for _ in range(5000):
        true_g = RNG.uniform(-1.0, -1e-9)
        g_hat = true_g + RNG.uniform(-eps, eps)
        score = witness.score(g_hat)
        assert score > monitor.threshold, (
            f"missed a violation: true_g={true_g:.4g} g_hat={g_hat:.4g} "
            f"score={score:.4g} threshold={monitor.threshold:.4g}"
        )


def test_margin_witness_floor_and_composition() -> None:
    w = MarginWitness()
    assert w.violation_floor() == 0.0
    claim = TwoSidedClaim.compose(threshold=-0.1, false_alarm_bound=0.05, witness=w)
    assert claim.miss_bound == 0.0
    with pytest.raises(SoundnessNotEstablished):
        TwoSidedClaim.compose(threshold=0.1, false_alarm_bound=0.05, witness=w)


def test_empirical_power_witness_is_labelled_statistical() -> None:
    w = EmpiricalPowerWitness(n_failures=100, n_missed=2, confidence=0.95)
    bound = w.miss_probability(threshold=0.0)
    assert 0.02 < bound < 0.10
    assert "STATISTICAL" in w.justification()
    assert EmpiricalPowerWitness(n_failures=50, n_missed=50).miss_probability(0.0) == 1.0


# ===================================================================== #
# 6. The gate invariant: no actuation without a fresh, bound certificate
# ===================================================================== #


def _gate(**kw) -> ActionGate:
    return ActionGate(threshold=0.0, false_alarm_bound=0.05, safe_action="STOP", **kw)


def test_gate_emits_when_certified_and_abstains_when_not() -> None:
    g = _gate()
    ok = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0)
    assert not ok.abstained and ok.certificate is not None
    assert np.array_equal(ok.action, np.ones(2))

    bad = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=+1.0)
    assert bad.abstained and bad.action == "STOP" and "exceeds" in bad.reason


def test_certificate_cannot_be_forged_by_a_foreign_authority() -> None:
    g = _gate()
    attacker = CertificateAuthority()
    attacker._epoch = g.verifier.epoch  # noqa: SLF001 -- simulating an attacker
    forged = attacker.mint(
        observation=np.zeros(3),
        action=np.ones(2),
        score=-99.0,
        threshold=0.0,
        false_alarm_bound=0.0,
        miss_bound=0.0,
    )
    assert not g.verifier.verify(forged)
    with pytest.raises(ForgedCertificate):
        g._admit(forged, np.zeros(3), np.ones(2))  # noqa: SLF001


def test_certificate_is_single_use() -> None:
    from certabstain.errors import ReplayedCertificate

    g = _gate()
    d = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0)
    cert = d.certificate
    assert cert is not None
    with pytest.raises(ReplayedCertificate):
        g._admit(cert, np.zeros(3), np.ones(2))  # noqa: SLF001


def test_certificate_is_epoch_bound() -> None:
    from certabstain.errors import StaleCertificate

    g = _gate()
    cert = g._authority.mint(  # noqa: SLF001 -- minting is deliberately not public
        observation=np.zeros(3),
        action=np.ones(2),
        score=-1.0,
        threshold=0.0,
        false_alarm_bound=0.05,
        miss_bound=0.0,
    )
    g._authority.advance()  # noqa: SLF001
    with pytest.raises(StaleCertificate):
        g._admit(cert, np.zeros(3), np.ones(2))  # noqa: SLF001


def test_certificate_is_action_bound() -> None:
    from certabstain.errors import BindingMismatch

    g = _gate()
    cert = g._authority.mint(  # noqa: SLF001 -- minting is deliberately not public
        observation=np.zeros(3),
        action=np.ones(2),
        score=-1.0,
        threshold=0.0,
        false_alarm_bound=0.05,
        miss_bound=0.0,
    )
    with pytest.raises(BindingMismatch):
        g._admit(cert, np.zeros(3), np.full(2, 7.0))  # noqa: SLF001
    with pytest.raises(BindingMismatch):
        g._admit(cert, np.ones(3), np.ones(2))  # noqa: SLF001


def test_gate_fails_closed_on_nonfinite_score() -> None:
    g = _gate()
    for bad in (np.nan, np.inf, -np.inf):
        d = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=bad)
        assert d.abstained
        assert d.action == "STOP"
        # Pin the *reason*, as the cover-membership test does. Asserting only
        # `abstained` would still pass if the non-finite branch were deleted
        # and the score fell through to the threshold comparison or the broad
        # fail-closed handler -- three different refusals, one assertion.
        assert "non-finite monitor score" in d.reason


def test_gate_fails_closed_on_internal_exception() -> None:
    """An un-encodable observation must abstain, never propagate a live action."""

    class Unencodable:
        pass

    g = _gate()
    d = g.step(observation=Unencodable(), proposed_action=np.ones(2), score=-1.0)
    assert d.abstained
    assert d.action == "STOP"
    assert "failed closed" in d.reason


def test_gate_exposes_no_override_path() -> None:
    """There is no public affordance for emitting an uncertified action.

    An allowlist, not a denylist. A denylist of six names ("force", "override",
    ...) passes green on ``step(..., allow_uncertified=True)``, which is the
    same bypass under a name nobody thought to ban -- so every public parameter
    and attribute must be named here explicitly, and adding one is a
    deliberate edit to this test rather than a silent extension of the surface.
    """
    import inspect

    public = [n for n in dir(ActionGate) if not n.startswith("_")]
    assert set(public) == {"verifier", "log", "miss_bound", "step", "abstention_rate"}

    # `authority` itself must NOT be public: it carries mint(), an oracle that
    # applies no score/threshold/cover check.
    assert not hasattr(ActionGate, "authority")

    for method, allowed in (
        (ActionGate.step, {"self", "observation", "proposed_action", "score"}),
        (
            ActionGate.__init__,
            {
                "self", "threshold", "false_alarm_bound", "safe_action",
                "claim", "cover",
            },
        ),
    ):
        sig = inspect.signature(method)
        assert set(sig.parameters) == allowed, (
            f"{method.__qualname__} grew a parameter not blessed by this test: "
            f"{set(sig.parameters) - allowed}"
        )
        assert all(
            p.kind is inspect.Parameter.KEYWORD_ONLY
            for n, p in sig.parameters.items()
            if n != "self"
        ), f"{method.__qualname__} accepts positional or **kwargs arguments"


def test_no_public_mint_or_override_anywhere_in_the_gate_module() -> None:
    """Every public callable the gate module exports is frozen by exact set.

    A denylist over one class was not enough: a mutation study showed three
    real bypasses shipping green against the earlier version of this test --
    a new public `CertificateAuthority.mint_unchecked`, a new `allow_any=`
    parameter on `mint`, and a public *instance* attribute honoured by `step`
    (invisible to `dir(ActionGate)`, which only sees the class). So the
    signature of every public member is pinned by name, and __slots__ is
    asserted so instance attributes cannot appear at all.
    """
    import inspect

    from certabstain import gate as gate_mod

    assert set(gate_mod.__all__) == {
        "SafetyCertificate", "CertificateAuthority", "CertificateVerifier",
        "ActionGate", "GateDecision",
    }
    public_module_names = {
        n for n in dir(gate_mod)
        if not n.startswith("_") and not inspect.ismodule(getattr(gate_mod, n))
    }
    assert public_module_names <= set(gate_mod.__all__) | {
        "Any", "Callable", "Final", "dataclass", "field", "np",
        "TwoSidedClaim", "BindingMismatch", "CertAbstainError",
        "ForgedCertificate", "MissingCertificate", "ReplayedCertificate",
        "StaleCertificate", "annotations",
    }, f"gate.py grew an unexported public name: {public_module_names}"

    # Exact public surface of every exported class. Adding anything -- method,
    # property, or parameter -- must be a deliberate edit here.
    expected_members = {
        "ActionGate": {"verifier", "log", "miss_bound", "step", "abstention_rate"},
        "CertificateAuthority": {"epoch", "advance", "mint", "verify"},
        "CertificateVerifier": {"epoch", "verify"},
        "SafetyCertificate": {
            "binding", "epoch", "nonce", "tag", "false_alarm_bound",
            "miss_bound", "score", "threshold", "is_two_sided", "summary",
        },
        "GateDecision": {"action", "abstained", "reason", "certificate"},
    }
    expected_params = {
        ("ActionGate", "step"):
            {"self", "observation", "proposed_action", "score"},
        ("ActionGate", "__init__"):
            {"self", "threshold", "false_alarm_bound", "safe_action", "claim",
             "cover"},
        ("CertificateAuthority", "mint"):
            {"self", "observation", "action", "score", "threshold",
             "false_alarm_bound", "miss_bound"},
        ("CertificateAuthority", "verify"): {"self", "cert"},
        ("CertificateAuthority", "advance"): {"self"},
        ("CertificateVerifier", "verify"): {"self", "cert"},
        ("SafetyCertificate", "summary"): {"self"},
    }

    for name, expected in expected_members.items():
        cls = getattr(gate_mod, name)
        public = {n for n in dir(cls) if not n.startswith("_")}
        assert public == expected, (
            f"{name}'s public surface changed: added {public - expected}, "
            f"removed {expected - public}"
        )
        for mname in public:
            member = inspect.getattr_static(cls, mname, None)
            if isinstance(member, property) or not callable(member):
                continue
            try:
                params = set(inspect.signature(member).parameters)
            except (TypeError, ValueError):
                continue
            key = (name, mname)
            assert key in expected_params, (
                f"{name}.{mname} is callable but not pinned by this test"
            )
            assert params == expected_params[key], (
                f"{name}.{mname} parameters changed: "
                f"added {params - expected_params[key]}, "
                f"removed {expected_params[key] - params}"
            )

    # __init__ is not in dir()'s public set but is the other entry point.
    assert set(inspect.signature(gate_mod.ActionGate.__init__).parameters) == \
        expected_params[("ActionGate", "__init__")]
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for n, p in inspect.signature(gate_mod.ActionGate.__init__).parameters.items()
        if n != "self"
    )

    # Instance attributes are invisible to dir(cls), so pin __slots__ too: a
    # bypass flag set on the instance was the mutation that slipped through.
    for name in ("ActionGate", "CertificateAuthority", "CertificateVerifier"):
        cls = getattr(gate_mod, name)
        assert hasattr(cls, "__slots__"), f"{name} lost __slots__"
        assert not any(s.startswith("_") is False for s in cls.__slots__), (
            f"{name}.__slots__ declares a public attribute: {cls.__slots__}"
        )
    g = ActionGate(threshold=0.0, false_alarm_bound=0.05, safe_action="STOP")
    with pytest.raises(AttributeError):
        g.emit_anyway = True  # type: ignore[attr-defined]


def test_the_gate_hands_out_no_mint_oracle() -> None:
    """The policy cannot mint its own permission through the gate's surface.

    A public `authority` property was exactly that: mint() applies no score,
    threshold or cover check, so a gate configured to always abstain could
    still be handed a verifying certificate for any observation/action pair.
    """
    g = ActionGate(
        threshold=0.0, false_alarm_bound=0.05, safe_action="STOP",
        cover=lambda _obs: False,          # this gate must abstain, always
    )
    obs, act = np.zeros(3), np.ones(2)
    assert g.step(observation=obs, proposed_action=act, score=-1e9).abstained

    # The audit view is public, and it can check a tag but not create one.
    assert isinstance(g.verifier, CertificateVerifier)
    assert not hasattr(g.verifier, "mint")
    assert not hasattr(g, "authority")


def test_gate_accepts_no_authority_from_outside() -> None:
    """There is no injection point for the authority the gate verifies against.

    A type-checked `authority=` was not enough. Even a *genuine*
    CertificateAuthority left the caller holding mint() -- an oracle applying
    no score, threshold or cover check -- for the very instance the gate
    verifies against, so a gate rigged to always abstain could still be handed
    a certificate that its own public verifier accepts. And an isinstance
    check is subclass-permissive: a two-line subclass overriding verify() to
    return True defeats it. The gate mints its own authority instead.
    """
    assert "authority" not in inspect.signature(ActionGate.__init__).parameters

    class Lax(CertificateAuthority):
        def verify(self, cert) -> bool:  # would accept anything
            return True

    for substitute in (Lax(), CertificateAuthority(), object()):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ActionGate(
                threshold=0.0, false_alarm_bound=0.05, safe_action="STOP",
                authority=substitute,
            )

    # A foreign authority's certificates do not verify here, because the gate's
    # key is its own and was never shared.
    g = ActionGate(threshold=0.0, false_alarm_bound=0.05, safe_action="STOP")
    foreign = CertificateAuthority()
    obs, act = np.zeros(3), np.ones(2)
    forged = foreign.mint(
        observation=obs, action=act, score=-1e9, threshold=0.0,
        false_alarm_bound=0.0, miss_bound=0.0,
    )
    assert not g.verifier.verify(forged)


def test_gate_refuses_a_meaningless_threshold() -> None:
    """`score > threshold` is False for NaN and +inf, so a non-finite
    threshold certifies every score presented -- the strict-off switch the
    module docstring denies exists, reachable by direct construction."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="threshold must be finite"):
            ActionGate(threshold=bad, false_alarm_bound=0.05, safe_action="STOP")
    for bad in (float("nan"), -0.1, 1.5):
        with pytest.raises(ValueError, match="false_alarm_bound"):
            ActionGate(threshold=0.0, false_alarm_bound=bad, safe_action="STOP")


def test_certificate_score_and_threshold_are_authenticated() -> None:
    """The MAC covers every field, as SafetyCertificate's docstring claims.

    With score/threshold outside the tag, a certificate could be re-labelled
    with any score and still verify -- so an auditor reading cert.summary()
    would be reading unauthenticated data.
    """
    from dataclasses import replace

    g = _gate()
    d = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0)
    cert = d.certificate
    assert cert is not None and g.verifier.verify(cert)

    for field_name, bogus in (("score", -999.0), ("threshold", 999.0)):
        relabelled = replace(cert, **{field_name: bogus})
        assert not g.verifier.verify(relabelled), (
            f"{field_name} is outside the MAC: a re-labelled certificate verified"
        )


def test_gate_fails_closed_when_the_authority_itself_raises() -> None:
    """`Fails closed on *any* internal error` includes the epoch advance.

    With advance() outside the try, the exception escaped step() entirely and
    returned no GateDecision -- a caller with its own except/default could turn
    that into an emission.
    """
    g = _gate()

    class Exploding(CertificateAuthority):
        def advance(self) -> int:
            raise RuntimeError("authority unavailable")

    g._authority = Exploding()  # noqa: SLF001 -- injecting an internal failure
    d = g.step(observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0)
    assert d.abstained and d.action == "STOP"
    assert "failed closed on RuntimeError" in d.reason


def test_certificate_is_immutable() -> None:
    g = _gate()
    cert = g.step(
        observation=np.zeros(3), proposed_action=np.ones(2), score=-1.0
    ).certificate
    assert cert is not None
    for field in ("threshold", "score", "epoch", "miss_bound"):
        with pytest.raises((AttributeError, TypeError)):
            setattr(cert, field, 0.0)
    with pytest.raises((AttributeError, TypeError)):
        cert.injected = True  # type: ignore[attr-defined]


# ===================================================================== #
# 7. End to end
# ===================================================================== #


def test_end_to_end_abstention_rate_matches_the_bound() -> None:
    eps, alpha = 0.02, 0.05
    witness = CertifiedModelErrorWitness(epsilon=eps)
    trajs = [list(witness.score(RNG.uniform(0.3, 1.0, 25))) for _ in range(999)]

    monitor = build_monitor(
        nominal_trajectories=trajs,
        alpha=alpha,
        witness=witness,
        shift_budget=0.01,
        safe_action="STOP",
    )
    assert monitor.calibration_level == pytest.approx(alpha - 0.01)
    assert monitor.claim is not None and monitor.claim.miss_bound == 0.0

    # Nominal deployment: abstention should be rare and within budget.
    n_roll, fired = 2000, 0
    for _ in range(n_roll):
        traj = witness.score(RNG.uniform(0.3, 1.0, 25))
        fired += bool(np.any(traj > monitor.threshold))
    rate = fired / n_roll
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n_roll)
    assert rate <= alpha + 4 * se

    # Unsafe deployment: every violating rollout is caught.
    for _ in range(2000):
        true_g = RNG.uniform(-0.8, -1e-9)
        traj = witness.score(true_g + RNG.uniform(-eps, eps, 25))
        assert bool(np.any(traj > monitor.threshold))

    assert "UNBOUNDED" not in monitor.describe()
