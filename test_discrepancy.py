"""M3 acceptance tests for the discrepancy certifier.

Spec criteria:
  * a learned clearance net (input dim <= 4) certified with
    eps <= 3x the empirical sup-gap (10^7 samples) within <= 10^6 leaf
    evaluations;
  * a target below the empirical floor forces TargetNotCertified reporting
    the achieved bound and the worst uncertified leaf;
  * mode-aware partial certification: fails to a smaller cover, never a
    weaker claim; CoverTooSmall enforces the declared minimum;
  * certificate bindings: weight-bytes hash and reference identity.
Plus an exactness check the acceptance task can't give: certifying a net
against a bias-shifted copy of itself, where the true sup-gap is known in
closed form (= |delta| everywhere), in input dimension 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from certabstain import (
    CoverTooSmall,
    Interval,
    MLP,
    TargetNotCertified,
    certify_epsilon,
)
from certabstain.discrepancy import (
    MODE_IN,
    MODE_OUT,
    MODE_STRADDLE,
    _batched_ibp,
    weights_hash,
)
from certabstain.nnbound import fit_mlp, ibp_bounds
from certabstain.reference import CircleClearance

DOMAIN = Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
CLEAR = CircleClearance(ox=0.0, oy=0.0, r=0.15)


def _trained_clearance_net(seed: int = 0) -> MLP:
    rng = np.random.default_rng(seed)
    X = rng.uniform(DOMAIN.lo, DOMAIN.hi, size=(40_000, 2))
    Y = CLEAR.value(X)
    return fit_mlp((2, 32, 32, 1), X, Y, steps=4000, seed=seed)


NET = _trained_clearance_net()


# ===================================================================== #
# Batched IBP agrees with the single-box path
# ===================================================================== #


def test_batched_ibp_matches_single_box() -> None:
    rng = np.random.default_rng(3)
    net = MLP.random((3, 16, 8, 2), rng=rng)
    los = rng.uniform(-0.5, 0.0, size=(50, 3))
    his = los + rng.uniform(0.01, 0.5, size=(50, 3))
    blo, bhi = _batched_ibp(net, los, his)
    for i in range(50):
        single = ibp_bounds(net, Interval(los[i], his[i]))
        assert np.max(np.abs(blo[i] - single.lo)) < 1e-12
        assert np.max(np.abs(bhi[i] - single.hi)) < 1e-12


# ===================================================================== #
# The headline acceptance run
# ===================================================================== #


def test_clearance_certification_meets_spec_targets() -> None:
    cert = certify_epsilon(
        NET,
        CLEAR.interval_batch,
        DOMAIN,
        reference_id=CLEAR.reference_id(),
        ref_float=CLEAR.value,
        target=None,                    # best effort, then check the ratio
        max_leaf_evals=400_000,
        floor_samples=10_000_000,
    )
    eps = float(cert.eps[0])
    floor = float(cert.empirical_floor[0])
    assert cert.n_leaf_evals <= 1_000_000
    assert eps >= floor, "certified bound cannot sit below an observed gap"
    ratio = eps / floor
    assert ratio <= 3.0, (
        f"eps={eps:.4g} is {ratio:.2f}x the empirical floor {floor:.4g}; "
        f"spec requires <= 3x within the budget "
        f"({cert.n_leaf_evals} leaf evals, {cert.n_leaves} leaves)"
    )
    assert cert.cover_fraction > 0.999  # no mode predicate: full cover
    assert cert.matches_network(NET)
    print(
        f"\n[M3 metrics] eps={eps:.4g} floor={floor:.4g} ratio={ratio:.2f} "
        f"leaves={cert.n_leaves} evals={cert.n_leaf_evals}"
    )


def test_target_below_floor_forces_reporting_refusal() -> None:
    quick_floor = 200_000  # the refusal path doesn't need the full 1e7
    with pytest.raises(TargetNotCertified) as ei:
        certify_epsilon(
            NET,
            CLEAR.interval_batch,
            DOMAIN,
            reference_id=CLEAR.reference_id(),
            ref_float=CLEAR.value,
            target=1e-6,               # far below any plausible floor
            max_leaf_evals=120_000,
            floor_samples=quick_floor,
        )
    msg = str(ei.value)
    assert "achieved" in msg and "worst leaf" in msg and "empirical floor" in msg


# ===================================================================== #
# Exactness: certified eps must bracket a KNOWN sup-gap (input dim 4)
# ===================================================================== #


def test_exact_constant_gap_in_dim4() -> None:
    rng = np.random.default_rng(9)
    net = MLP.random((4, 8, 1), rng=rng, scale=0.7)
    delta = 0.05
    W2 = [(np.array(W, copy=True), np.array(b, copy=True)) for W, b in net.weights]
    W2[-1] = (W2[-1][0], W2[-1][1] + delta)
    twin = MLP(tuple(W2), activation=net.activation)

    def ref(lo, hi):
        return _batched_ibp(twin, lo, hi)

    dom = Interval(-0.08 * np.ones(4), 0.08 * np.ones(4))
    cert = certify_epsilon(
        net,
        ref,
        dom,
        reference_id="bias-shifted twin (delta=0.05)",
        ref_float=lambda p: twin.forward(p),
        target=None,
        max_leaf_evals=300_000,
        floor_samples=200_000,
        crown_polish=True,
    )
    eps = float(cert.eps[0])
    assert eps >= delta - 1e-12, "certified bound fell below the true sup-gap"
    # The overage here is dominated by the REFERENCE side: this synthetic's
    # reference is itself a network enclosed by IBP, whose per-leaf width adds
    # directly to the bound. (The acceptance task's analytic reference has
    # near-zero slack, which is why it achieves ~1.2x.) 1.4x at this budget is
    # the honest expectation for this construction, not certifier looseness.
    assert eps <= delta * 1.40, (
        f"eps={eps:.4g} vs true gap {delta}: exceeds the documented budget "
        f"for reference-side IBP slack on this construction"
    )


# ===================================================================== #
# Mode-aware partial certification
# ===================================================================== #


def _lower_half_mode(lo, hi):
    """Certified mode: y < 0 (state component 1)."""
    out = np.full(lo.shape[0], MODE_STRADDLE)
    out[hi[:, 1] < 0.0] = MODE_IN
    out[lo[:, 1] >= 0.0] = MODE_OUT
    return out


def test_partial_cover_and_cover_too_small() -> None:
    kw = dict(
        ref=CLEAR.interval_batch,
        domain=DOMAIN,
        reference_id=CLEAR.reference_id(),
        ref_float=CLEAR.value,
        mode=_lower_half_mode,
        mode_float=lambda p: p[:, 1] < 0.0,
        max_leaf_evals=200_000,
        floor_samples=500_000,
    )
    with pytest.raises(CoverTooSmall, match="below the declared minimum"):
        certify_epsilon(NET, min_cover_fraction=0.90, **kw)

    cert = certify_epsilon(NET, min_cover_fraction=0.30, **kw)
    assert 0.45 <= cert.cover_fraction <= 0.501, cert.cover_fraction
    # membership: certified half in, other half out, near-boundary excluded
    inside = cert.contains(np.array([[0.1, -0.3], [-0.2, -0.25]]))
    outside = cert.contains(np.array([[0.1, 0.3], [-0.2, 0.25], [0.0, 1e-5]]))
    assert bool(np.all(inside))
    assert not bool(np.any(outside))


# ===================================================================== #
# Bindings and immutability
# ===================================================================== #


def test_certificate_bindings_and_freeze() -> None:
    cert = certify_epsilon(
        NET,
        CLEAR.interval_batch,
        Interval(np.array([-0.2, -0.2]), np.array([0.2, 0.2])),
        reference_id=CLEAR.reference_id(),
        ref_float=CLEAR.value,
        max_leaf_evals=50_000,
        floor_samples=200_000,
    )
    assert cert.matches_network(NET)
    tampered = [(np.array(W, copy=True), np.array(b, copy=True))
                for W, b in NET.weights]
    tampered[0][0][0, 0] += 1e-12
    assert not cert.matches_network(MLP(tuple(tampered)))
    assert weights_hash(NET) == cert.net_hash
    with pytest.raises((AttributeError, TypeError)):
        cert.eps = np.zeros(1)  # type: ignore[misc]
    with pytest.raises((ValueError, RuntimeError)):
        cert.eps[0] = 99.0
