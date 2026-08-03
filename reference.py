"""Interval-twin reference contact models (M2).

Assumption A1 of the spec, made concrete: the guarantee this package will
issue is *relative to the closed-form equations in this module*. Each model
is therefore implemented twice from the same equations:

    step(s, u)                 float64, vectorized -- generates training and
                               evaluation data; NOT certified; the analysand's
                               world.
    step_interval(S, U, ...)   the certified interval extension on the M0
                               substrate; what the discrepancy certifier (M3)
                               diffs a learned model against.

plus interval-checkable mode predicates, because assumption A2 lives here:
per-mode smoothness is what makes certification tractable, and the mode
boundary is exactly where enclosures widen.

Two models, per spec:

SpringDamper2D -- a 2-D point mass over a spring-damper ground at y = 0.
    One guard (contact vs. free flight), handled branchlessly through an
    interval indicator, so its interval twin accepts any box; a box
    straddling the guard simply pays the hull. The stiffness parameter k is
    the knob for the Parmar-Halm-Posa obstruction: demo/stiffness_sweep.py
    measures enclosure width against k and publishes the curve as-is.

PusherSlider -- quasistatic planar pushing with an ellipsoidal limit surface
    (Lynch-Mason / Hogan-Rodriguez lineage). Three modes from the motion
    cone: stick, slide_left (+y along the face), slide_right (-y). The
    sliding branches are genuinely different dynamics, so the interval twin
    requires a certified mode and REFUSES boxes that straddle the cone --
    A2 enforced at the API, not assumed.

Conventions for PusherSlider: slider body frame; pusher contacts the left
face at body point (-a, py); push direction +x; control is the pusher
velocity (vpx, vpy) in the body frame with vpx >= 0; c = tau_max / f_max is
the limit-surface length scale. Explicit Euler in the world frame. The
parameter condition c > a*mu/2 makes the sliding-mode denominator
(c^2 + a*mu*s*py + py^2) strictly positive for ALL py, so sliding kinematics
can never divide by zero; construction refuses parameter sets that break it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import EnclosureError, ModeIndeterminate, NonFiniteEnclosure
from .interval import Interval, istack

__all__ = ["SpringDamper2D", "PusherSlider", "CircleClearance"]


def _comp(S: Interval, i: int) -> Interval:
    return Interval(S.lo[..., i], S.hi[..., i])


def _indicator_neg(Y: Interval) -> Interval:
    """Sound enclosure of the indicator 1{y < 0}: {0}, {1}, or [0, 1]."""
    lo = np.where(Y.hi < 0.0, 1.0, 0.0)
    hi = np.where(Y.lo >= 0.0, 0.0, 1.0)
    return Interval(lo, hi)


def _sq(x):
    """x*x with the dependency handled when x is an interval."""
    return x.sqr() if isinstance(x, Interval) else x * x


def _cos(x):
    return x.cos() if isinstance(x, Interval) else np.cos(x)


def _sin(x):
    return x.sin() if isinstance(x, Interval) else np.sin(x)


# --------------------------------------------------------------------------- #
# Model (a): spring-damper point contact
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpringDamper2D:
    """2-D point mass, gravity, control force, spring-damper ground at y = 0.

    State (x, y, vx, vy); control (ux, uy). Contact model (in contact iff
    y < 0): normal force max(0, k*(-y) - d*vy), tangential viscous friction
    -mu_t*vx, both gated by the contact indicator. Semi-implicit Euler.
    """

    m: float = 1.0
    g: float = 9.81
    k: float = 1e4
    d: float = 10.0
    mu_t: float = 1.0
    dt: float = 1e-3

    n_states = 4
    n_controls = 2
    modes = ("free", "contact")

    def __post_init__(self) -> None:
        vals = (self.m, self.g, self.k, self.d, self.mu_t, self.dt)
        if not all(np.isfinite(v) for v in vals):
            raise NonFiniteEnclosure("SpringDamper2D parameters must be finite")
        if self.m <= 0 or self.dt <= 0 or min(self.k, self.d, self.mu_t) < 0:
            raise ValueError("need m > 0, dt > 0, and k, d, mu_t >= 0")

    # -- float twin ---------------------------------------------------------- #

    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        y, vx, vy = s[..., 1], s[..., 2], s[..., 3]
        in_c = (y < 0.0).astype(np.float64)
        fn = in_c * np.maximum(0.0, self.k * np.maximum(0.0, -y) - self.d * vy)
        ft = -self.mu_t * vx * in_c
        ax = (u[..., 0] + ft) / self.m
        ay = (u[..., 1] - self.m * self.g + fn) / self.m
        vx2 = vx + self.dt * ax
        vy2 = vy + self.dt * ay
        x2 = s[..., 0] + self.dt * vx2
        y2 = y + self.dt * vy2
        return np.stack([x2, y2, vx2, vy2], axis=-1)

    # -- interval twin ------------------------------------------------------- #

    def step_interval(self, S: Interval, U: Interval) -> Interval:
        """Sound one-step enclosure. Accepts any box; a box straddling the
        contact guard pays the hull through the interval indicator rather
        than refusing -- this model's single guard is cheap to hull."""
        X, Y, VX, VY = (_comp(S, i) for i in range(4))
        UX, UY = _comp(U, 0), _comp(U, 1)
        M = Interval.point(self.m)

        in_c = _indicator_neg(Y)
        pen = (-Y).maximum(0.0)
        fn_core = (Interval.point(self.k) * pen - Interval.point(self.d) * VY)
        FN = in_c * fn_core.maximum(0.0)
        FT = -(Interval.point(self.mu_t) * VX * in_c)
        AX = (UX + FT) / M
        AY = (UY - M * Interval.point(self.g) + FN) / M
        VX2 = VX + Interval.point(self.dt) * AX
        VY2 = VY + Interval.point(self.dt) * AY
        X2 = X + Interval.point(self.dt) * VX2
        Y2 = Y + Interval.point(self.dt) * VY2
        return istack([X2, Y2, VX2, VY2])

    # -- modes --------------------------------------------------------------- #

    def mode_certificate(self, S: Interval) -> dict[str, np.ndarray]:
        Y = _comp(S, 1)
        return {"free": Y.lo >= 0.0, "contact": Y.hi < 0.0}

    def in_mode(self, S: Interval, mode: str) -> bool:
        if mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; modes are {self.modes}")
        return bool(np.all(self.mode_certificate(S)[mode]))

    def reference_id(self) -> str:
        """Deterministic identity string for certificate binding (spec A1).

        Derived from the exact parameters, not hand-maintained: a changed
        stiffness, timestep, or any other field changes this string, so a
        certificate bound to the old string is detected as stale rather than
        silently reused against a system it was never analyzed against.
        """
        return (
            f"SpringDamper2D(m={self.m!r}, g={self.g!r}, k={self.k!r}, "
            f"d={self.d!r}, mu_t={self.mu_t!r}, dt={self.dt!r})"
        )


# --------------------------------------------------------------------------- #
# Model (b): quasistatic pusher-slider
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PusherSlider:
    """Quasistatic planar pushing with an ellipsoidal limit surface.

    State (x, y, theta, py): slider pose in the world plus the pusher's
    contact coordinate along the left face. Control (vpx, vpy): pusher
    velocity in the slider body frame, vpx >= 0.

    Sticking maps pusher velocity to slider twist through the limit-surface
    normal; sliding pins the contact force to a friction-cone edge
    b_s = (1, s*mu), s = +1 (slide_left) or -1 (slide_right), and py evolves
    by the tangential slip. Mode selection follows the motion cone spanned by
    the edge pusher-velocities u_minus, u_plus, tested by cross products
    (g1, g2) in the exact order the float twin branches on:
    g2 < 0 -> slide_left; elif g1 < 0 -> slide_right; else stick.
    """

    a: float = 0.05      # half-length; contact face at body x = -a
    c: float = 0.03      # tau_max / f_max
    mu: float = 0.3      # pusher-slider friction
    dt: float = 0.01
    py_max: float = 0.04  # declared |py| domain (contact stays on the face)

    n_states = 4
    n_controls = 2
    modes = ("stick", "slide_left", "slide_right")

    def __post_init__(self) -> None:
        vals = (self.a, self.c, self.mu, self.dt, self.py_max)
        if not all(np.isfinite(v) for v in vals):
            raise NonFiniteEnclosure("PusherSlider parameters must be finite")
        if min(self.a, self.c, self.dt) <= 0 or self.mu < 0 or self.py_max <= 0:
            raise ValueError("need a, c, dt > 0, mu >= 0, py_max > 0")
        if self.py_max >= self.a:
            raise ValueError("py_max must keep contact on the face: py_max < a")
        if not (self.c > self.a * self.mu / 2.0):
            raise ValueError(
                f"parameter condition c > a*mu/2 violated "
                f"(c={self.c}, a*mu/2={self.a * self.mu / 2}); the sliding "
                f"denominator c^2 + a*mu*s*py + py^2 could reach zero and "
                f"the model would be ill-posed. Refusing at construction."
            )

    # -- shared closed forms (generic over float arrays and Intervals) ------- #

    def _edges(self, py):
        c2 = self.c * self.c
        a2 = self.a * self.a
        upx = c2 + _sq(py) + (self.a * self.mu) * py
        upy = self.a * py + (c2 + a2) * self.mu
        umx = c2 + _sq(py) - (self.a * self.mu) * py
        umy = self.a * py - (c2 + a2) * self.mu
        return upx, upy, umx, umy

    def _mode_g(self, py, vpx, vpy):
        upx, upy, umx, umy = self._edges(py)
        g2 = vpx * upy - vpy * upx  # >= 0: inside the +mu edge
        g1 = umx * vpy - umy * vpx  # >= 0: inside the -mu edge
        return g1, g2

    def _stick_body(self, py, vpx, vpy):
        c2 = self.c * self.c
        a2 = self.a * self.a
        D = c2 + a2 + _sq(py)
        vx = ((c2 + a2) * vpx - (self.a) * py * vpy) / D
        vy = (-(self.a) * py * vpx + (c2 + _sq(py)) * vpy) / D
        w = (-(self.a) * vy - py * vx) / c2
        pydot = vpx * 0.0  # exact zero of the right shape/type
        return vx, vy, w, pydot

    def _slide_body(self, py, vpx, vpy, sign: float):
        c2 = self.c * self.c
        wdir = (-(self.a * self.mu * sign) - py) / c2
        N = (c2 + (self.a * self.mu * sign) * py + _sq(py)) / c2
        gam = vpx / N
        vx = gam
        vy = gam * (self.mu * sign)
        w = gam * wdir
        pydot = vpy - (vy + (-(self.a)) * w)
        return vx, vy, w, pydot

    def _integrate(self, x, y, th, py, vx, vy, w, pydot):
        dt = self.dt if not isinstance(x, Interval) else Interval.point(self.dt)
        ct, st = _cos(th), _sin(th)
        x2 = x + dt * (ct * vx - st * vy)
        y2 = y + dt * (st * vx + ct * vy)
        th2 = th + dt * w
        py2 = py + dt * pydot
        return x2, y2, th2, py2

    # -- float twin ---------------------------------------------------------- #

    def float_mode(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """0 = stick, 1 = slide_left, 2 = slide_right; the branch order is
        the definition the mode certificates align with."""
        py = np.asarray(s, dtype=np.float64)[..., 3]
        u = np.asarray(u, dtype=np.float64)
        g1, g2 = self._mode_g(py, u[..., 0], u[..., 1])
        return np.where(g2 < 0.0, 1, np.where(g1 < 0.0, 2, 0))

    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        x, y, th, py = (s[..., i] for i in range(4))
        vpx, vpy = u[..., 0], u[..., 1]
        mode = self.float_mode(s, u)

        with np.errstate(all="ignore"):  # unselected branch lanes may misbehave
            bodies = [
                self._stick_body(py, vpx, vpy),
                self._slide_body(py, vpx, vpy, +1.0),
                self._slide_body(py, vpx, vpy, -1.0),
            ]
            vx = np.choose(mode, [b[0] for b in bodies])
            vy = np.choose(mode, [b[1] for b in bodies])
            w = np.choose(mode, [b[2] for b in bodies])
            pydot = np.choose(mode, [b[3] for b in bodies])
            x2, y2, th2, py2 = self._integrate(x, y, th, py, vx, vy, w, pydot)
        return np.stack([x2, y2, th2, py2], axis=-1)

    # -- interval twin ------------------------------------------------------- #

    def mode_certificate(self, S: Interval, U: Interval) -> dict[str, np.ndarray]:
        """Per-element certified mode membership, aligned with the float
        twin's branch order. Indeterminate elements certify nothing."""
        PY = _comp(S, 3)
        VPX, VPY = _comp(U, 0), _comp(U, 1)
        G1, G2 = self._mode_g(PY, VPX, VPY)
        return {
            "stick": (G1.lo >= 0.0) & (G2.lo >= 0.0),
            "slide_left": G2.hi < 0.0,
            "slide_right": (G2.lo >= 0.0) & (G1.hi < 0.0),
        }

    def in_mode(self, S: Interval, U: Interval, mode: str) -> bool:
        if mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; modes are {self.modes}")
        return bool(np.all(self.mode_certificate(S, U)[mode]))

    def step_interval(self, S: Interval, U: Interval, mode: str) -> Interval:
        """Sound one-step enclosure within a certified mode.

        Refuses unless every element of the box certifiably lies in ``mode``:
        the sliding branches are different dynamics, and hulling across the
        motion cone would silently charge the certificate for a discontinuity
        it never declared. A2 is enforced here, not assumed.
        """
        if not self.in_mode(S, U, mode):
            raise ModeIndeterminate(
                f"box does not certifiably lie in mode {mode!r}; the motion-"
                f"cone predicate is indeterminate or violated somewhere in "
                f"the box. Split the domain at the cone or certify per mode."
            )
        PY = _comp(S, 3)
        if not bool(
            np.all(PY.lo >= -self.py_max) and np.all(PY.hi <= self.py_max)
        ):
            raise EnclosureError(
                f"py leaves the declared face domain |py| <= {self.py_max}; "
                f"the contact-on-face assumption does not hold on this box."
            )
        X, Y, TH = _comp(S, 0), _comp(S, 1), _comp(S, 2)
        VPX, VPY = _comp(U, 0), _comp(U, 1)
        if mode == "stick":
            vx, vy, w, pydot = self._stick_body(PY, VPX, VPY)
        else:
            sign = +1.0 if mode == "slide_left" else -1.0
            vx, vy, w, pydot = self._slide_body(PY, VPX, VPY, sign)
        X2, Y2, TH2, PY2 = self._integrate(X, Y, TH, PY, vx, vy, w, pydot)
        return istack([X2, Y2, TH2, PY2])

    def reference_id(self) -> str:
        """Deterministic identity string for certificate binding (spec A1)."""
        return (
            f"PusherSlider(a={self.a!r}, c={self.c!r}, mu={self.mu!r}, "
            f"dt={self.dt!r}, py_max={self.py_max!r})"
        )


# --------------------------------------------------------------------------- #
# Clearance reference (for the M3 acceptance task)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CircleClearance:
    """h(s) = ||(x, y) - (ox, oy)|| - r: signed clearance to a disk obstacle.

    Uses the first two state components; smooth everywhere; interval twin is
    exact-ish through sqr/sqrt on the M0 substrate. reference_id() gives a
    deterministic identity string for certificate binding.
    """

    ox: float = 0.0
    oy: float = 0.0
    r: float = 0.15

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) for v in (self.ox, self.oy, self.r)):
            raise NonFiniteEnclosure("CircleClearance parameters must be finite")
        if self.r <= 0:
            raise ValueError("obstacle radius must be positive")

    def reference_id(self) -> str:
        return f"CircleClearance(ox={self.ox!r}, oy={self.oy!r}, r={self.r!r})"

    def value(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        dx = s[..., 0] - self.ox
        dy = s[..., 1] - self.oy
        return np.sqrt(dx * dx + dy * dy) - self.r

    def interval_batch(
        self, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched interval twin: (B, d>=2) box arrays -> (B, 1) bounds."""
        DX = Interval(lo[..., 0], hi[..., 0]) - Interval.point(self.ox)
        DY = Interval(lo[..., 1], hi[..., 1]) - Interval.point(self.oy)
        S = DX.sqr() + DY.sqr()
        # outward rounding can push an exact 0 lower bound to -5e-324 when the
        # box contains the obstacle centre; a sum of squares is >= 0, so
        # clamping is sound and keeps sqrt's domain refusal for real bugs
        S = Interval(np.maximum(S.lo, 0.0), S.hi)
        H = S.sqrt() - Interval.point(self.r)
        return H.lo[..., None], H.hi[..., None]
