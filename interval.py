"""Outward-rounded interval arithmetic -- the certification substrate (M0).

Soundness contract (spec Lemma L1): for every operation provided here and
every real input contained in the input intervals, the exact real result is
contained in the output interval. Floating point may widen an enclosure; it
can never break containment.

Rounding discipline
-------------------
IEEE-754 binary64 guarantees correct rounding for +, -, *, /, and sqrt: the
computed value is the representable number nearest the exact result, so the
exact result lies strictly between the computed value's two neighbours, and
one directed ``nextafter`` step per side restores containment.

``exp`` and ``tanh`` carry no such guarantee from libm. They are assumed
*faithful* (within 1 ulp of exact) and widened by ``_TRANS_STEPS`` nextafter
steps -- and that assumption is checked, not trusted: ``rounding_self_test``
validates every primitive against hard-coded brackets generated offline at
700 significant digits and against exact rational arithmetic, and
``require_sound_environment`` refuses to let certification proceed in an
environment that fails. An untested assumption would be a hole in the chain;
a tested one that refuses on failure is a documented link.

Design rules inherited from the package:

* every failure is a raise, never a silent widening to +/- infinity
* intervals are immutable after construction
* the certified path depends on numpy and the standard library only
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .errors import EnclosureError, EnvironmentUnsound, NonFiniteEnclosure

__all__ = [
    "Interval",
    "istack",
    "matvec",
    "affine",
    "rounding_self_test",
    "require_sound_environment",
]

_INF = np.inf
_TRANS_STEPS = 2  # widening steps for functions libm does not correctly round


def _down(x: np.ndarray, steps: int = 1) -> np.ndarray:
    """Move every element at least ``steps`` representable values toward -inf."""
    for _ in range(steps):
        x = np.nextafter(x, -_INF)
    return x


def _up(x: np.ndarray, steps: int = 1) -> np.ndarray:
    """Move every element at least ``steps`` representable values toward +inf."""
    for _ in range(steps):
        x = np.nextafter(x, _INF)
    return x


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Return an array that cannot be made writable again.

    ``arr.setflags(write=False)`` looks like it freezes ``arr``, but ``arr``
    still owns its data -- numpy lets any array that owns its data flip
    WRITEABLE back to True on request, so a caller holding a reference can
    silently undo the freeze (confirmed: ``a.setflags(write=True); a[0] = ...``
    succeeds on a "frozen" array). A read-only *view* whose base is itself
    read-only does not have this hole: numpy refuses to set WRITEABLE on a
    view unless the base allows it. Round-tripping through ``bytes`` (which
    has no WRITEABLE flag to flip) gives such a view and makes the freeze
    permanent -- verified empirically that ``setflags(write=True)`` then
    raises ValueError instead of succeeding.
    """
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


# --------------------------------------------------------------------------- #
# The interval type
# --------------------------------------------------------------------------- #


class Interval:
    """A closed box [lo, hi] of float64 values, immutable, always finite.

    Elementwise over arbitrary numpy shapes. Construction validates finiteness
    and ordering; the endpoint arrays are frozen. All arithmetic returns new
    intervals with outward-rounded endpoints.
    """

    __slots__ = ("lo", "hi", "_sealed")

    def __init__(self, lo, hi) -> None:
        lo_arr = np.array(lo, dtype=np.float64, copy=True)
        hi_arr = np.array(hi, dtype=np.float64, copy=True)
        lo_b, hi_b = np.broadcast_arrays(lo_arr, hi_arr)
        lo_arr = np.array(lo_b, dtype=np.float64, copy=True)
        hi_arr = np.array(hi_b, dtype=np.float64, copy=True)

        if not (np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr))):
            raise NonFiniteEnclosure(
                "interval endpoints must be finite; a non-finite endpoint means "
                "an operation lost its bound (overflow or invalid input) and "
                "the enclosure would be vacuous. Refusing to construct it."
            )
        if np.any(lo_arr > hi_arr):
            raise EnclosureError("interval lower endpoint exceeds upper endpoint")

        self.lo = _freeze(lo_arr)
        self.hi = _freeze(hi_arr)
        self._sealed = True

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Interval is immutable")
        super().__setattr__(name, value)

    # -- construction helpers ---------------------------------------------- #

    @classmethod
    def point(cls, x) -> "Interval":
        """The degenerate interval [x, x]. Exact: the input is taken as given."""
        return cls(x, x)

    # -- introspection ------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, ...]:
        return self.lo.shape

    def contains(self, x) -> np.ndarray:
        """Elementwise membership test (inclusive)."""
        x = np.asarray(x, dtype=np.float64)
        return (self.lo <= x) & (x <= self.hi)

    def width(self) -> np.ndarray:
        """Diagnostic only -- computed in nearest rounding, NOT certified."""
        return self.hi - self.lo

    def __repr__(self) -> str:
        return f"Interval(lo={self.lo!r}, hi={self.hi!r})"

    # -- internal ----------------------------------------------------------- #

    @staticmethod
    def _coerce(other) -> "Interval":
        if isinstance(other, Interval):
            return other
        return Interval.point(other)

    # -- exact operations (no rounding required) ---------------------------- #

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __pos__(self) -> "Interval":
        return self

    def __abs__(self) -> "Interval":
        straddles = (self.lo <= 0.0) & (self.hi >= 0.0)
        lo = np.where(straddles, 0.0, np.minimum(np.abs(self.lo), np.abs(self.hi)))
        hi = np.maximum(np.abs(self.lo), np.abs(self.hi))
        return Interval(lo, hi)

    def minimum(self, other) -> "Interval":
        """Elementwise min. Exact: min is monotone in both arguments."""
        o = self._coerce(other)
        return Interval(np.minimum(self.lo, o.lo), np.minimum(self.hi, o.hi))

    def maximum(self, other) -> "Interval":
        """Elementwise max. Exact: max is monotone in both arguments."""
        o = self._coerce(other)
        return Interval(np.maximum(self.lo, o.lo), np.maximum(self.hi, o.hi))

    # -- correctly rounded IEEE operations (1-step widening) ---------------- #

    def __add__(self, other) -> "Interval":
        o = self._coerce(other)
        with np.errstate(over="ignore"):  # inf is caught by the constructor
            return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __sub__(self, other) -> "Interval":
        o = self._coerce(other)
        with np.errstate(over="ignore"):
            return Interval(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, other) -> "Interval":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other) -> "Interval":
        o = self._coerce(other)
        a1, b1 = np.broadcast_arrays(self.lo, o.lo)
        a2, b2 = np.broadcast_arrays(self.hi, o.hi)
        with np.errstate(over="ignore"):
            p = np.stack([a1 * b1, a1 * b2, a2 * b1, a2 * b2])
            return Interval(_down(p.min(axis=0)), _up(p.max(axis=0)))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Interval":
        o = self._coerce(other)
        if np.any((o.lo <= 0.0) & (o.hi >= 0.0)):
            raise EnclosureError(
                "division by an interval containing zero yields an unbounded "
                "enclosure. Refusing; split the domain so the denominator has "
                "a definite sign, or reformulate."
            )
        a1, b1 = np.broadcast_arrays(self.lo, o.lo)
        a2, b2 = np.broadcast_arrays(self.hi, o.hi)
        with np.errstate(over="ignore"):
            q = np.stack([a1 / b1, a1 / b2, a2 / b1, a2 / b2])
            return Interval(_down(q.min(axis=0)), _up(q.max(axis=0)))

    def __rtruediv__(self, other) -> "Interval":
        return self._coerce(other).__truediv__(self)

    def sqr(self) -> "Interval":
        """x*x with the dependency handled: the true range never dips below 0."""
        straddles = (self.lo <= 0.0) & (self.hi >= 0.0)
        with np.errstate(over="ignore"):
            c = np.stack([self.lo * self.lo, self.hi * self.hi])
        lo = np.where(straddles, 0.0, _down(c.min(axis=0)))
        hi = _up(c.max(axis=0))
        return Interval(lo, hi)

    def sqrt(self) -> "Interval":
        if np.any(self.lo < 0.0):
            raise EnclosureError(
                "sqrt of an interval extending below zero. Refusing rather "
                "than clipping: a negative lower endpoint here usually means "
                "a domain bug upstream, and clipping would hide it."
            )
        lo = np.maximum(_down(np.sqrt(self.lo)), 0.0)
        hi = _up(np.sqrt(self.hi))
        return Interval(lo, hi)

    # -- transcendentals (libm faithful, checked by self-test) --------------- #

    def exp(self) -> "Interval":
        with np.errstate(over="ignore"):  # inf is checked two lines down
            raw_hi = np.exp(self.hi)
        if np.any(np.isinf(raw_hi)):
            raise NonFiniteEnclosure(
                "exp overflows float64 on this interval; the enclosure would "
                "be unbounded. Shrink the domain."
            )
        hi = _up(raw_hi, _TRANS_STEPS)
        if np.any(np.isinf(hi)):
            raise NonFiniteEnclosure(
                "exp upper endpoint crossed the float64 ceiling under outward "
                "rounding. Shrink the domain."
            )
        lo = np.maximum(_down(np.exp(self.lo), _TRANS_STEPS), 0.0)
        return Interval(lo, hi)

    def tanh(self) -> "Interval":
        lo = np.maximum(_down(np.tanh(self.lo), _TRANS_STEPS), -1.0)
        hi = np.minimum(_up(np.tanh(self.hi), _TRANS_STEPS), 1.0)
        return Interval(lo, hi)


    # -- trigonometry (libm faithful, checked by self-test) ------------------ #
    # Extremum detection uses an interval enclosure of pi: a maximiser of cos
    # lies in [lo, hi] iff x/(2*pi) hits an integer there, and the quotient is
    # computed in interval arithmetic, so the integer test can only
    # over-include (widening toward [-1, 1]), never miss an extremum.

    def cos(self) -> "Interval":
        PI = Interval(math.pi, np.nextafter(math.pi, _INF))  # math.pi < pi
        TWO_PI = PI + PI
        q_max = self / TWO_PI            # maxima of cos at integer quotients
        q_min = (self - PI) / TWO_PI     # minima at integer shifted quotients
        has_max = np.floor(q_max.hi) >= np.ceil(q_max.lo)
        has_min = np.floor(q_min.hi) >= np.ceil(q_min.lo)
        cl = np.cos(self.lo)
        ch = np.cos(self.hi)
        lo = np.where(
            has_min, -1.0,
            np.maximum(_down(np.minimum(cl, ch), _TRANS_STEPS), -1.0),
        )
        hi = np.where(
            has_max, 1.0,
            np.minimum(_up(np.maximum(cl, ch), _TRANS_STEPS), 1.0),
        )
        return Interval(lo, hi)

    def sin(self) -> "Interval":
        PI = Interval(math.pi, np.nextafter(math.pi, _INF))
        return (self - PI / 2.0).cos()   # sin(x) = cos(x - pi/2), soundly

    # -- structure ----------------------------------------------------------- #

    def __getitem__(self, idx) -> "Interval":
        return Interval(self.lo[idx], self.hi[idx])


def istack(intervals, axis: int = -1) -> Interval:
    """Stack intervals along an axis (exact; endpoints are just rearranged)."""
    return Interval(
        np.stack([iv.lo for iv in intervals], axis=axis),
        np.stack([iv.hi for iv in intervals], axis=axis),
    )


# --------------------------------------------------------------------------- #
# Sound linear algebra (exact weights, interval vectors)
# --------------------------------------------------------------------------- #


def matvec(W, x: Interval) -> Interval:
    """Sound enclosure of W @ x for an exact float matrix W and interval x.

    Per output row, each term's extreme over the box is attained at an
    endpoint of x (linearity), the computed endpoint product is widened one
    step, and the sum is accumulated sequentially with a directed step after
    every addition. Deliberately the simple, obviously-sound accumulation:
    the networks this substrate serves are small, and an accumulation whose
    soundness argument is one line beats a faster one whose argument is a
    page.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise EnclosureError(f"matvec expects a 2-D matrix; got ndim={W.ndim}")
    if not np.all(np.isfinite(W)):
        raise NonFiniteEnclosure("matvec weight matrix contains non-finite entries")
    if x.lo.ndim != 1 or x.lo.shape[0] != W.shape[1]:
        raise EnclosureError(
            f"shape mismatch: W is {W.shape}, x is {x.lo.shape}; expected "
            f"x of length {W.shape[1]}"
        )

    tl = np.where(W >= 0.0, W * x.lo, W * x.hi)  # per-term minimal corner
    th = np.where(W >= 0.0, W * x.hi, W * x.lo)  # per-term maximal corner
    TL = _down(tl)
    TH = _up(th)

    acc_lo = np.zeros(W.shape[0])
    acc_hi = np.zeros(W.shape[0])
    for j in range(W.shape[1]):
        acc_lo = _down(acc_lo + TL[:, j])
        acc_hi = _up(acc_hi + TH[:, j])
    return Interval(acc_lo, acc_hi)


def affine(W, x: Interval, b) -> Interval:
    """Sound enclosure of W @ x + b for exact W, b and interval x."""
    b = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(b)):
        raise NonFiniteEnclosure("affine bias vector contains non-finite entries")
    y = matvec(W, x)
    return Interval(_down(y.lo + b), _up(y.hi + b))


# --------------------------------------------------------------------------- #
# The rounding self-test
# --------------------------------------------------------------------------- #
# Brackets generated offline at 700 significant digits (mpmath); each triple
# (x, true_lo, true_hi) satisfies true_lo <= f(x) <= true_hi with true_lo,
# true_hi adjacent (or equal, when f(x) is exactly representable) floats.
# 700 digits matters: tanh(700) differs from 1 by ~1e-608, which a lower
# working precision silently rounds away, turning the bracket into a lie.

_REFERENCE_BRACKETS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "exp": (
        (-745.0, 0.0, 5e-324),
        (-700.0, 9.85967654375977e-305, 9.859676543759773e-305),
        (-50.0, 1.9287498479639176e-22, 1.9287498479639178e-22),
        (-10.0, 4.539992976248485e-05, 4.5399929762484854e-05),
        (-2.0, 0.13533528323661267, 0.1353352832366127),
        (-1.0, 0.3678794411714423, 0.36787944117144233),
        (-0.5, 0.6065306597126333, 0.6065306597126334),
        (-1e-08, 0.99999999, 0.9999999900000001),
        (-1e-300, 0.9999999999999999, 1.0),
        (0.0, 1.0, 1.0),
        (1e-300, 1.0, 1.0000000000000002),
        (1e-08, 1.00000001, 1.0000000100000002),
        (0.5, 1.648721270700128, 1.6487212707001282),
        (1.0, 2.718281828459045, 2.7182818284590455),
        (2.0, 7.3890560989306495, 7.38905609893065),
        (10.0, 22026.465794806714, 22026.465794806718),
        (50.0, 5.184705528587072e+21, 5.184705528587073e+21),
        (700.0, 1.0142320547350045e+304, 1.0142320547350046e+304),
        (709.0, 8.218407461554971e+307, 8.218407461554972e+307),
    ),
    "tanh": (
        (-700.0, -1.0, -0.9999999999999999),
        (-30.0, -1.0, -0.9999999999999999),
        (-1.0, -0.761594155955765, -0.7615941559557649),
        (-0.5, -0.4621171572600098, -0.46211715726000974),
        (-1e-08, -1e-08, -9.999999999999999e-09),
        (-1e-300, -1e-300, -9.999999999999999e-301),
        (0.0, 0.0, 0.0),
        (1e-300, 9.999999999999999e-301, 1e-300),
        (1e-08, 9.999999999999999e-09, 1e-08),
        (0.5, 0.46211715726000974, 0.4621171572600098),
        (1.0, 0.7615941559557649, 0.761594155955765),
        (30.0, 0.9999999999999999, 1.0),
        (700.0, 0.9999999999999999, 1.0),
    ),
    "sqrt": (
        (5e-324, 2.2227587494850775e-162, 2.2227587494850775e-162),
        (1e-320, 9.999944335758488e-161, 9.99994433575849e-161),
        (1e-300, 1e-150, 1.0000000000000001e-150),
        (0.25, 0.5, 0.5),
        (0.5, 0.7071067811865475, 0.7071067811865476),
        (1.0, 1.0, 1.0),
        (2.0, 1.414213562373095, 1.4142135623730951),
        (3.0, 1.7320508075688772, 1.7320508075688774),
        (10.0, 3.162277660168379, 3.1622776601683795),
        (1e+300, 1e+150, 1.0000000000000002e+150),
        (1.5e+308, 1.2247448713915889e+154, 1.224744871391589e+154),
    ),
    "cos": (
        (-100.0, 0.8623188722876839, 0.862318872287684),
        (-10.0, -0.8390715290764525, -0.8390715290764524),
        (-1.0, 0.5403023058681397, 0.5403023058681398),
        (-1e-08, 0.9999999999999999, 1.0),
        (0.0, 1.0, 1.0),
        (1e-300, 0.9999999999999999, 1.0),
        (1e-08, 0.9999999999999999, 1.0),
        (0.5, 0.8775825618903726, 0.8775825618903728),
        (1.0, 0.5403023058681397, 0.5403023058681398),
        (2.0, -0.4161468365471424, -0.41614683654714235),
        (3.0, -0.9899924966004455, -0.9899924966004454),
        (3.141592653589793, -1.0, -0.9999999999999999),
        (4.0, -0.6536436208636119, -0.6536436208636118),
        (10.0, -0.8390715290764525, -0.8390715290764524),
        (100.0, 0.8623188722876839, 0.862318872287684),
        (1000000.0, 0.9367521275331447, 0.9367521275331449),
    ),
    "sin": (
        (-1000000.0, 0.34999350217129294, 0.349993502171293),
        (-100.0, 0.5063656411097588, 0.5063656411097589),
        (-3.0, -0.14112000805986724, -0.1411200080598672),
        (-1.0, -0.8414709848078966, -0.8414709848078965),
        (-1e-08, -1e-08, -9.999999999999999e-09),
        (0.0, 0.0, 0.0),
        (1e-300, 9.999999999999999e-301, 1e-300),
        (1e-08, 9.999999999999999e-09, 1e-08),
        (0.5, 0.47942553860420295, 0.479425538604203),
        (1.0, 0.8414709848078965, 0.8414709848078966),
        (1.5707963267948966, 0.9999999999999999, 1.0),
        (2.0, 0.9092974268256816, 0.9092974268256817),
        (3.0, 0.1411200080598672, 0.14112000805986724),
        (3.141592653589793, 1.224646799147353e-16, 1.2246467991473532e-16),
        (10.0, -0.5440211108893699, -0.5440211108893698),
        (1000000.0, -0.349993502171293, -0.34999350217129294),
    ),
}

_SELF_TEST_PASSED: bool | None = None


def rounding_self_test(n_random: int = 200, seed: int = 0) -> None:
    """Validate the numeric substrate. Raises EnvironmentUnsound on failure.

    Three tiers:
      1. nextafter behaves as directed rounding requires, including at zero
         and in the denormal range;
      2. every primitive's enclosure contains the hard-coded high-precision
         bracket at every reference point;
      3. +, -, *, / enclosures contain the exact rational result on random
         inputs, checked with stdlib Fraction arithmetic (floats are exact
         rationals, so this tier has no numerical error at all).
    """
    problems: list[str] = []

    # -- tier 1: nextafter sanity ------------------------------------------- #
    checks = (
        (np.nextafter(1.0, _INF) > 1.0, "nextafter(1, +inf) must increase"),
        (np.nextafter(1.0, -_INF) < 1.0, "nextafter(1, -inf) must decrease"),
        (np.nextafter(0.0, -_INF) < 0.0, "nextafter(0, -inf) must go negative"),
        (np.nextafter(0.0, _INF) > 0.0, "nextafter(0, +inf) must go positive"),
        (np.nextafter(5e-324, -_INF) == 0.0, "denormal step down must reach 0"),
        (np.nextafter(-5e-324, _INF) == 0.0, "denormal step up must reach 0"),
    )
    with np.errstate(over="ignore"):  # the overflow here is the point
        checks += (
            (
                np.nextafter(1.7976931348623157e308, _INF) == _INF,
                "step above float max must be +inf (so overflow is detectable)",
            ),
        )
    for ok, msg in checks:
        if not bool(ok):
            problems.append(f"nextafter: {msg}")

    # -- tier 2: hard-coded high-precision brackets -------------------------- #
    fns = {
        "exp": lambda iv: iv.exp(),
        "tanh": lambda iv: iv.tanh(),
        "sqrt": lambda iv: iv.sqrt(),
        "cos": lambda iv: iv.cos(),
        "sin": lambda iv: iv.sin(),
    }
    for name, fn in fns.items():
        for x, true_lo, true_hi in _REFERENCE_BRACKETS[name]:
            enc = fn(Interval.point(x))
            lo = float(np.asarray(enc.lo))
            hi = float(np.asarray(enc.hi))
            if not (lo <= true_lo and true_hi <= hi):
                problems.append(
                    f"{name}({x!r}): enclosure [{lo!r}, {hi!r}] does not "
                    f"contain reference bracket [{true_lo!r}, {true_hi!r}]"
                )

    # -- tier 3: exact rational containment for the basic four --------------- #
    rng = np.random.default_rng(seed)
    ops = (
        ("add", lambda a, b: a + b, lambda fa, fb: fa + fb),
        ("sub", lambda a, b: a - b, lambda fa, fb: fa - fb),
        ("mul", lambda a, b: a * b, lambda fa, fb: fa * fb),
        ("div", lambda a, b: a / b, lambda fa, fb: fa / fb),
    )
    for _ in range(n_random):
        scale = 10.0 ** rng.integers(-6, 7)
        a = float(rng.uniform(-1.0, 1.0) * scale)
        b = float(rng.uniform(-1.0, 1.0) * scale)
        for name, iop, fop in ops:
            if name == "div" and abs(b) < 1e-12:
                continue
            enc = iop(Interval.point(a), Interval.point(b))
            exact = fop(Fraction(a), Fraction(b))
            lo = Fraction(float(np.asarray(enc.lo)))
            hi = Fraction(float(np.asarray(enc.hi)))
            if not (lo <= exact <= hi):
                problems.append(f"{name}({a!r}, {b!r}): exact result escaped")

    if problems:
        detail = "\n  ".join(problems[:10])
        more = "" if len(problems) <= 10 else f"\n  ... and {len(problems) - 10} more"
        raise EnvironmentUnsound(
            f"floating-point environment failed the rounding self-test "
            f"({len(problems)} violation(s)):\n  {detail}{more}\n"
            f"No certification may proceed on this substrate."
        )


def require_sound_environment() -> None:
    """Run the self-test once per process; certification entry points call this.

    Caches success. A failure is NOT cached as permanent state -- it raises
    every time, so nothing downstream can proceed past it.
    """
    global _SELF_TEST_PASSED
    if _SELF_TEST_PASSED is True:
        return
    rounding_self_test()
    _SELF_TEST_PASSED = True
