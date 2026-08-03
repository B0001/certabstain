"""M2 acceptance tests for the reference contact models.

Spec criteria:
  * 10^6 random states per model: float twin result contained in the
    interval twin result, always. (Point-interval flavor -- the direct
    twin-consistency claim of assumption A1.)
  * Box flavor: sampled interior points of random boxes stay inside the
    interval twin's enclosure of the box.
  * Mode certificates align with the float twin's branch choice wherever
    they are definite, and the pushing twin refuses straddling boxes.
The stiffness sweep itself lives in demo/stiffness_sweep.py; its output is
published as an artifact, not asserted here beyond a smoke check.
"""

from __future__ import annotations

import numpy as np
import pytest

from certabstain import EnclosureError, Interval, ModeIndeterminate
from certabstain.interval import istack
from certabstain.reference import PusherSlider, SpringDamper2D

MODE_NAMES = ("stick", "slide_left", "slide_right")


def _spring_samples(n, rng):
    s = np.stack(
        [
            rng.normal(size=n) * 0.5,
            rng.uniform(-0.05, 0.05, size=n),  # straddle the contact guard
            rng.normal(size=n),
            rng.normal(size=n),
        ],
        axis=-1,
    )
    u = rng.normal(size=(n, 2)) * 5.0
    return s, u


def _pusher_samples(n, rng, p: PusherSlider):
    s = np.stack(
        [
            rng.normal(size=n) * 0.5,
            rng.normal(size=n) * 0.5,
            rng.uniform(-np.pi, np.pi, size=n),
            rng.uniform(-p.py_max, p.py_max, size=n),
        ],
        axis=-1,
    )
    u = np.stack(
        [rng.uniform(1e-3, 0.1, size=n), rng.uniform(-0.1, 0.1, size=n)],
        axis=-1,
    )
    return s, u


def _point_interval(arr):
    return Interval(arr, arr)


# ===================================================================== #
# 10^6 float-in-interval, per model
# ===================================================================== #


def test_springdamper_million_point_containment() -> None:
    model = SpringDamper2D()
    rng = np.random.default_rng(61)
    s, u = _spring_samples(1_000_000, rng)
    f = model.step(s, u)
    enc = model.step_interval(_point_interval(s), _point_interval(u))
    ok = enc.contains(f)
    bad = int(ok.size - np.count_nonzero(ok))
    assert bad == 0, f"{bad} float-twin results escaped the interval twin"


def test_pusher_million_point_containment() -> None:
    model = PusherSlider()
    rng = np.random.default_rng(67)
    s, u = _pusher_samples(1_000_000, rng, model)
    f = model.step(s, u)
    S, U = _point_interval(s), _point_interval(u)
    cert = model.mode_certificate(S, U)
    fmode = model.float_mode(s, u)

    checked = refused = bad = 0
    for idx, name in enumerate(MODE_NAMES):
        mask = cert[name]
        if not np.any(mask):
            continue
        # certificate must agree with the float branch wherever definite
        assert np.all(fmode[mask] == idx), f"certificate/branch mismatch: {name}"
        enc = model.step_interval(
            Interval(s[mask], s[mask]), Interval(u[mask], u[mask]), name
        )
        ok = enc.contains(f[mask])
        bad += int(ok.size - np.count_nonzero(ok))
        checked += int(np.count_nonzero(mask))
    refused = s.shape[0] - checked

    assert bad == 0, f"{bad} float-twin results escaped their mode's twin"
    assert checked > 0
    # indeterminate certificates should be measure-zero flukes at the cone
    assert refused <= 5, f"{refused} boundary refusals out of 1e6 (expected ~0)"


# ===================================================================== #
# Box flavor: interior samples stay inside the box enclosure
# ===================================================================== #


def test_springdamper_box_enclosure() -> None:
    model = SpringDamper2D()
    rng = np.random.default_rng(71)
    for _ in range(200):
        c, uc = _spring_samples(1, rng)
        rs = 10.0 ** rng.uniform(-4, -1.5)
        ru = 10.0 ** rng.uniform(-4, -1.5)
        S = Interval(c[0] - rs, c[0] + rs)
        U = Interval(uc[0] - ru, uc[0] + ru)
        enc = model.step_interval(S, U)
        pts_s = rng.uniform(S.lo, S.hi, size=(500, 4))
        pts_u = rng.uniform(U.lo, U.hi, size=(500, 2))
        f = model.step(pts_s, pts_u)
        ok = (enc.lo[None, :] <= f) & (f <= enc.hi[None, :])
        assert int(ok.size - np.count_nonzero(ok)) == 0


def test_pusher_box_enclosure() -> None:
    model = PusherSlider()
    rng = np.random.default_rng(73)
    done = 0
    while done < 200:
        c, uc = _pusher_samples(1, rng, model)
        rs = 10.0 ** rng.uniform(-4, -2)
        ru = 10.0 ** rng.uniform(-4, -2.5)
        S = Interval(c[0] - rs, c[0] + rs)
        U = Interval(uc[0] - ru, uc[0] + ru)
        cert = model.mode_certificate(S, U)
        mode = next((m for m in MODE_NAMES if bool(np.all(cert[m]))), None)
        if mode is None:
            continue  # box straddles the cone; the twin would (rightly) refuse
        try:
            enc = model.step_interval(S, U, mode)
        except EnclosureError:
            continue  # e.g. py domain edge; refusal is sound behaviour
        pts_s = rng.uniform(S.lo, S.hi, size=(500, 4))
        pts_u = rng.uniform(U.lo, U.hi, size=(500, 2))
        f = model.step(pts_s, pts_u)
        ok = (enc.lo[None, :] <= f) & (f <= enc.hi[None, :])
        assert int(ok.size - np.count_nonzero(ok)) == 0
        done += 1


# ===================================================================== #
# Modes, refusals, construction guards
# ===================================================================== #


def test_pusher_refuses_cone_straddling_box() -> None:
    model = PusherSlider()
    s = np.array([0.0, 0.0, 0.0, 0.0])
    S = Interval(s, s)
    # a control box wide enough in vpy to straddle the motion cone
    U = Interval(np.array([0.05, -0.5]), np.array([0.05, 0.5]))
    cert = model.mode_certificate(S, U)
    assert not any(bool(np.all(v)) for v in cert.values())
    for mode in MODE_NAMES:
        # Spec 7.3 has its own type: a straddling box is not a numeric
        # failure, and the remedy (split at the cone / certify per mode)
        # differs from every other enclosure refusal.
        with pytest.raises(ModeIndeterminate):
            model.step_interval(S, U, mode)


def test_pusher_refuses_py_domain_exit() -> None:
    model = PusherSlider()
    s = np.array([0.0, 0.0, 0.0, model.py_max * 1.5])
    S = Interval(s, s)
    U = Interval(np.array([0.05, 0.0]), np.array([0.05, 0.0]))
    mode = next(m for m, v in model.mode_certificate(S, U).items() if np.all(v))
    # The face-domain exit is a declared-domain violation, not mode
    # indeterminacy: it keeps the plain class, and this pins that difference.
    with pytest.raises(EnclosureError) as exc:
        model.step_interval(S, U, mode)
    assert "face domain" in str(exc.value)
    assert not isinstance(exc.value, ModeIndeterminate)


def test_springdamper_mode_predicates() -> None:
    model = SpringDamper2D()
    free = Interval(np.array([0.0, 0.01, -1.0, -1.0]), np.array([0.0, 0.02, 1.0, 1.0]))
    contact = Interval(np.array([0.0, -0.02, -1.0, -1.0]), np.array([0.0, -0.01, 1.0, 1.0]))
    straddle = Interval(np.array([0.0, -0.01, -1.0, -1.0]), np.array([0.0, 0.01, 1.0, 1.0]))
    assert model.in_mode(free, "free") and not model.in_mode(free, "contact")
    assert model.in_mode(contact, "contact") and not model.in_mode(contact, "free")
    assert not model.in_mode(straddle, "free")
    assert not model.in_mode(straddle, "contact")
    model.step_interval(straddle, Interval(np.zeros(2), np.zeros(2)))  # hulls, no refusal


def test_parameter_construction_refusals() -> None:
    with pytest.raises(ValueError, match="c > a\\*mu/2"):
        PusherSlider(a=0.05, c=0.007, mu=0.3)
    with pytest.raises(ValueError, match="py_max"):
        PusherSlider(py_max=0.05, a=0.05)
    with pytest.raises(ValueError):
        SpringDamper2D(m=-1.0)
    with pytest.raises(EnclosureError):
        SpringDamper2D(k=np.nan)


def test_unknown_mode_names_raise() -> None:
    model = PusherSlider()
    S = Interval(np.zeros(4), np.zeros(4))
    U = Interval(np.array([0.05, 0.0]), np.array([0.05, 0.0]))
    with pytest.raises(ValueError, match="unknown mode"):
        model.in_mode(S, U, "slip")
    with pytest.raises(ValueError, match="unknown mode"):
        SpringDamper2D().in_mode(S, "slip")
