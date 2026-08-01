"""M0 acceptance tests for the interval substrate.

Tiers mirror the spec:
  A. 10^6 randomized expression instances -- float trajectories must never
     escape their interval trajectories.
  B. 10^4 of those instances re-evaluated at 100 significant digits with
     mpmath -- the *exact* value must be contained (this is the actual
     soundness claim; tier A is the cheap wide net).
  C. Boundary battery: denormals, signed zero, overflow refusal, domain
     refusals, clamps.
  D. Exact-rational validation of matvec/affine via stdlib Fraction.
  E. Immutability and the self-test's own refusal path.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import certabstain.interval as ivmod
from certabstain import (
    MLP,
    EnclosureError,
    NonFiniteEnclosure,
    EnvironmentUnsound,
    EpsilonCertificate,
    Interval,
    affine,
    matvec,
    require_sound_environment,
    rounding_self_test,
)

RNG = np.random.default_rng(20260726)


# ===================================================================== #
# Expression-tree machinery shared by tiers A and B
# ===================================================================== #
# Each op is a triple (name, interval_fn, float_fn). Compositions are kept
# inside safe domains by construction (denominators bounded away from zero,
# exp arguments bounded via tanh) so that refusals don't truncate the fuzz.

UNARY = (
    ("neg", lambda x: -x, lambda p: -p),
    ("abs", lambda x: abs(x), lambda p: np.abs(p)),
    ("sqr", lambda x: x.sqr(), lambda p: p * p),
    ("sqrt_abs", lambda x: abs(x).sqrt(), lambda p: np.sqrt(np.abs(p))),
    ("tanh", lambda x: x.tanh(), lambda p: np.tanh(p)),
    ("cos", lambda x: x.cos(), lambda p: np.cos(p)),
    ("sin", lambda x: x.sin(), lambda p: np.sin(p)),
    (
        "safe_exp",  # exp of a value squashed into [-3, 3]
        lambda x: (x.tanh() * 3.0).exp(),
        lambda p: np.exp(np.tanh(p) * 3.0),
    ),
)

BINARY = (
    ("add", lambda a, b: a + b, lambda p, q: p + q),
    ("sub", lambda a, b: a - b, lambda p, q: p - q),
    ("mul", lambda a, b: a * b, lambda p, q: p * q),
    ("min", lambda a, b: a.minimum(b), lambda p, q: np.minimum(p, q)),
    ("max", lambda a, b: a.maximum(b), lambda p, q: np.maximum(p, q)),
    (
        "safe_div",  # denominator squashed into [0.6, 1.4]
        lambda a, b: a / (b.tanh() * 0.4 + 1.0),
        lambda p, q: p / (np.tanh(q) * 0.4 + 1.0),
    ),
)


def _random_inputs(batch: int, rng) -> tuple[list[Interval], list[np.ndarray]]:
    """Three interval batches plus one sampled point batch per interval."""
    intervals, points = [], []
    for _ in range(3):
        centre = rng.normal(size=batch) * 10.0 ** rng.integers(-3, 4)
        radius = np.abs(rng.normal(size=batch)) * 10.0 ** rng.integers(-4, 2)
        lo, hi = centre - radius, centre + radius
        iv = Interval(lo, hi)
        t = rng.uniform(size=batch)
        t[rng.uniform(size=batch) < 0.05] = 0.0  # stress the corners
        t[rng.uniform(size=batch) < 0.05] = 1.0
        pt = np.clip(lo + t * (hi - lo), lo, hi)
        intervals.append(iv)
        points.append(pt)
    return intervals, points


def _random_program(n_ops: int, n_leaves: int, rng) -> list[tuple]:
    """A straight-line program over node indices; grows the node list."""
    prog = []
    n = n_leaves
    for _ in range(n_ops):
        if rng.uniform() < 0.5:
            name, ifn, ffn = UNARY[rng.integers(len(UNARY))]
            prog.append(("u", ifn, ffn, int(rng.integers(n))))
        else:
            name, ifn, ffn = BINARY[rng.integers(len(BINARY))]
            prog.append(
                ("b", ifn, ffn, int(rng.integers(n)), int(rng.integers(n)))
            )
        n += 1
    return prog


def _run_program(prog, intervals, points):
    ivs = list(intervals)
    pts = list(points)
    for step in prog:
        if step[0] == "u":
            _, ifn, ffn, i = step
            ivs.append(ifn(ivs[i]))
            pts.append(ffn(pts[i]))
        else:
            _, ifn, ffn, i, j = step
            ivs.append(ifn(ivs[i], ivs[j]))
            pts.append(ffn(pts[i], pts[j]))
    return ivs[-1], pts[-1]


# ===================================================================== #
# Tier A: one million randomized instances, zero escapes
# ===================================================================== #


def test_million_instance_containment() -> None:
    structures, batch = 1000, 1000  # 10^6 instances total
    rng = np.random.default_rng(11)
    violations = 0
    worst = None
    for _ in range(structures):
        intervals, points = _random_inputs(batch, rng)
        prog = _random_program(n_ops=6, n_leaves=3, rng=rng)
        try:
            iv, pt = _run_program(prog, intervals, points)
        except EnclosureError:
            continue  # a refusal is sound behaviour, not a containment failure
        ok = iv.contains(pt)
        bad = int(np.size(ok) - np.count_nonzero(ok))
        if bad:
            violations += bad
            idx = int(np.argmin(ok))
            worst = (pt[idx], iv.lo[idx], iv.hi[idx])
    assert violations == 0, (
        f"{violations} containment violations out of 1e6; worst case "
        f"point={worst[0]!r} escaped [{worst[1]!r}, {worst[2]!r}]"
    )


# ===================================================================== #
# Tier B: 10^4 instances cross-checked against 100-digit mpmath
# ===================================================================== #


def test_exact_containment_against_mpmath() -> None:
    from mpmath import mp, mpf

    mp.dps = 100
    m_unary = {
        "neg": lambda v: -v,
        "abs": lambda v: abs(v),
        "sqr": lambda v: v * v,
        "sqrt_abs": lambda v: mp.sqrt(abs(v)),
        "tanh": lambda v: mp.tanh(v),
        "cos": lambda v: mp.cos(v),
        "sin": lambda v: mp.sin(v),
        "safe_exp": lambda v: mp.exp(mp.tanh(v) * 3),
    }
    m_binary = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "min": lambda a, b: min(a, b),
        "max": lambda a, b: max(a, b),
        "safe_div": lambda a, b: a / (mp.tanh(b) * mpf(2) / 5 + 1),
    }
    # Rebuild op tables carrying names so the mpmath twin can be looked up.
    unary = {name: (ifn, m_unary[name]) for name, ifn, _ in UNARY}
    binary = {name: (ifn, m_binary[name]) for name, ifn, _ in BINARY}

    rng = np.random.default_rng(13)
    checked = 0
    while checked < 10_000:
        # scalar instance: 3 leaf intervals, 5 ops
        leaves_iv, leaves_mp = [], []
        for _ in range(3):
            c = float(rng.normal() * 10.0 ** rng.integers(-3, 4))
            r = float(abs(rng.normal()) * 10.0 ** rng.integers(-4, 2))
            lo, hi = c - r, c + r
            t = float(rng.uniform())
            pt = min(max(lo + t * (hi - lo), lo), hi)
            leaves_iv.append(Interval(lo, hi))
            leaves_mp.append(mpf(pt))  # exact: mpf(float) is lossless
        ivs, mps = list(leaves_iv), list(leaves_mp)
        try:
            for _ in range(5):
                if rng.uniform() < 0.5:
                    name = list(unary)[rng.integers(len(unary))]
                    ifn, mfn = unary[name]
                    i = int(rng.integers(len(ivs)))
                    ivs.append(ifn(ivs[i]))
                    mps.append(mfn(mps[i]))
                else:
                    name = list(binary)[rng.integers(len(binary))]
                    ifn, mfn = binary[name]
                    i = int(rng.integers(len(ivs)))
                    j = int(rng.integers(len(ivs)))
                    ivs.append(ifn(ivs[i], ivs[j]))
                    mps.append(mfn(mps[i], mps[j]))
        except EnclosureError:
            continue
        enc, exact = ivs[-1], mps[-1]
        lo = mpf(float(np.asarray(enc.lo)))
        hi = mpf(float(np.asarray(enc.hi)))
        assert lo <= exact <= hi, (
            f"exact value {exact} escaped [{lo}, {hi}]"
        )
        checked += 1


# ===================================================================== #
# Tier C: boundary battery
# ===================================================================== #


def test_construction_refusals() -> None:
    # Inverted endpoints are malformed input, not a lost bound: plain
    # EnclosureError, and specifically NOT the non-finite subclass.
    with pytest.raises(EnclosureError) as exc:
        Interval(1.0, 0.0)
    assert not isinstance(exc.value, NonFiniteEnclosure)
    # Spec 7.9: a non-finite endpoint is its own refusal type.
    with pytest.raises(NonFiniteEnclosure):
        Interval(np.nan, 1.0)
    with pytest.raises(NonFiniteEnclosure):
        Interval(0.0, np.inf)


def test_operation_refusals() -> None:
    with pytest.raises(EnclosureError, match="containing zero"):
        Interval.point(1.0) / Interval(-1.0, 1.0)
    with pytest.raises(EnclosureError, match="below zero"):
        Interval(-1.0, 4.0).sqrt()
    # Overflow is a bound lost to non-finiteness, so it carries that type;
    # division-by-zero-containing and sqrt-below-zero above are domain
    # violations and deliberately stay the plain class.
    with pytest.raises(NonFiniteEnclosure, match="overflow"):
        Interval.point(1000.0).exp()
    with pytest.raises(NonFiniteEnclosure):  # add overflow -> non-finite endpoint
        Interval.point(1.7e308) + Interval.point(1.7e308)
    with pytest.raises(NonFiniteEnclosure):  # mul overflow
        Interval.point(1e200) * Interval.point(1e200)


def test_denormals_and_signed_zero() -> None:
    tiny = Interval(5e-324, 1e-320)
    s = tiny.sqrt()
    assert float(s.lo) >= 0.0 and float(s.hi) > 0.0
    z = Interval.point(0.0)
    assert bool((z * Interval(-1e300, 1e300)).contains(0.0))
    assert bool((z + z).contains(0.0))
    neg_zero = Interval.point(-0.0)
    assert bool(neg_zero.contains(0.0))


def test_transcendental_clamps() -> None:
    t = Interval.point(1000.0).tanh()
    assert float(t.hi) == 1.0 and float(t.lo) <= 1.0
    t2 = Interval.point(-1000.0).tanh()
    assert float(t2.lo) == -1.0
    e = Interval.point(-800.0).exp()
    assert float(e.lo) >= 0.0 and float(e.hi) > 0.0  # underflow stays sound


def test_mul_sign_grid_exact() -> None:
    """Every corner product of sign-mixed intervals is contained, exactly."""
    vals = [-2.5, -1.0, -0.0, 0.0, 0.5, 3.0]
    for a_lo in vals:
        for a_hi in vals:
            if a_lo > a_hi:
                continue
            for b_lo in vals:
                for b_hi in vals:
                    if b_lo > b_hi:
                        continue
                    prod = Interval(a_lo, a_hi) * Interval(b_lo, b_hi)
                    lo_f = Fraction(float(np.asarray(prod.lo)))
                    hi_f = Fraction(float(np.asarray(prod.hi)))
                    for x in (a_lo, a_hi):
                        for y in (b_lo, b_hi):
                            exact = Fraction(x) * Fraction(y)
                            assert lo_f <= exact <= hi_f


def test_sqr_never_dips_below_zero() -> None:
    s = Interval(-3.0, 2.0).sqr()
    assert float(s.lo) == 0.0
    assert bool(s.contains(0.0)) and bool(s.contains(9.0))


# ===================================================================== #
# Tier D: matvec/affine against exact rational arithmetic
# ===================================================================== #


def test_matvec_exact_rational_containment() -> None:
    rng = np.random.default_rng(17)
    for _ in range(200):
        m = int(rng.integers(1, 9))
        n = int(rng.integers(1, 9))
        W = rng.normal(size=(m, n)) * 10.0 ** rng.integers(-2, 3)
        lo = rng.normal(size=n)
        hi = lo + np.abs(rng.normal(size=n))
        x = Interval(lo, hi)
        y = matvec(W, x)
        # exact dot at corners and interior samples, via Fraction
        for _ in range(5):
            t = rng.uniform(size=n)
            t[rng.uniform(size=n) < 0.2] = 0.0
            t[rng.uniform(size=n) < 0.2] = 1.0
            pt = np.clip(lo + t * (hi - lo), lo, hi)
            for i in range(m):
                exact = sum(
                    Fraction(float(W[i, j])) * Fraction(float(pt[j]))
                    for j in range(n)
                )
                assert Fraction(float(y.lo[i])) <= exact <= Fraction(float(y.hi[i]))


def test_affine_matches_matvec_plus_bias() -> None:
    rng = np.random.default_rng(19)
    W = rng.normal(size=(4, 3))
    b = rng.normal(size=4)
    x = Interval(rng.normal(size=3), rng.normal(size=3) + 2.0)
    y = affine(W, x, b)
    pt = (x.lo + x.hi) / 2.0
    exact = [
        sum(Fraction(float(W[i, j])) * Fraction(float(pt[j])) for j in range(3))
        + Fraction(float(b[i]))
        for i in range(4)
    ]
    for i in range(4):
        assert Fraction(float(y.lo[i])) <= exact[i] <= Fraction(float(y.hi[i]))


def test_matvec_shape_and_finiteness_refusals() -> None:
    x = Interval(np.zeros(3), np.ones(3))
    # Shape errors are caller bugs and keep the plain class; only the
    # finiteness failure is spec 7.9's refusal.
    with pytest.raises(EnclosureError, match="2-D"):
        matvec(np.zeros(3), x)
    with pytest.raises(EnclosureError, match="mismatch"):
        matvec(np.zeros((2, 4)), x)
    with pytest.raises(NonFiniteEnclosure, match="non-finite"):
        matvec(np.array([[np.nan, 0.0, 0.0]]), x)


# ===================================================================== #
# Tier E: immutability and the self-test's refusal path
# ===================================================================== #


def test_interval_is_immutable() -> None:
    iv = Interval(np.zeros(3), np.ones(3))
    with pytest.raises((ValueError, RuntimeError)):
        iv.lo[0] = 5.0  # frozen array
    with pytest.raises(AttributeError):
        iv.lo = np.zeros(3)  # sealed attribute
    with pytest.raises(AttributeError):
        iv.injected = True


def test_self_test_passes_and_caches() -> None:
    rounding_self_test()
    require_sound_environment()
    require_sound_environment()  # cached second call


def test_self_test_refuses_on_corrupted_reference(monkeypatch) -> None:
    """A bracket the enclosure cannot contain must trip EnvironmentUnsound."""
    bad = dict(ivmod._REFERENCE_BRACKETS)
    bad["exp"] = ((0.0, 2.0, 2.0),)  # claims exp(0) is exactly 2 -- impossible
    monkeypatch.setattr(ivmod, "_REFERENCE_BRACKETS", bad)
    monkeypatch.setattr(ivmod, "_SELF_TEST_PASSED", None)
    with pytest.raises(EnvironmentUnsound, match="exp"):
        rounding_self_test()
    with pytest.raises(EnvironmentUnsound):
        require_sound_environment()  # failure is never cached as success


def test_nothing_certifies_on_an_unsound_environment(monkeypatch) -> None:
    """Spec 7.6 says "refuse to certify *anything*", so pin it at the two
    entry points that mint or propagate a claim, not only at the self-test.

    Both call require_sound_environment() first; a refactor that dropped
    either call would leave the self-test passing and the refusal gone.
    """
    import certabstain.discrepancy as dmod
    import certabstain.tube as tmod

    def boom() -> None:
        raise EnvironmentUnsound("simulated unsound substrate")

    for mod in (dmod, tmod):
        monkeypatch.setattr(mod, "require_sound_environment", boom)

    net = MLP.random((2, 4, 1), rng=np.random.default_rng(0), scale=0.5)
    domain = Interval(-0.1 * np.ones(2), 0.1 * np.ones(2))

    with pytest.raises(EnvironmentUnsound):
        dmod.certify_epsilon(
            net,
            lambda lo, hi: (np.zeros((lo.shape[0], 1)), np.zeros((lo.shape[0], 1))),
            domain,
            reference_id="unsound-environment guard test",
            target=None,
            max_leaf_evals=100,
        )

    with pytest.raises(EnvironmentUnsound):
        tmod.propagate_tube(
            net, _unused_cert_for_guard(), Interval(np.zeros(1), np.zeros(1)), [],
            n_states=1,
        )


def _unused_cert_for_guard():
    """A certificate the guard test never gets far enough to read."""
    return EpsilonCertificate(
        eps=np.zeros(1),
        domain_lo=np.zeros(1),
        domain_hi=np.zeros(1),
        cover_lo=np.zeros((1, 1)),
        cover_hi=np.zeros((1, 1)),
        cover_fraction=1.0,
        net_hash="unused",
        reference_id="unused",
        empirical_floor=np.zeros(1),
        n_leaf_evals=0,
        n_leaves=1,
        target=None,
    )
