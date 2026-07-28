"""Distribution-free calibration of the false-alarm side.

Everything in this module concerns one question: given monitor scores from
*nominal* (successful) rollouts only, what threshold fires on at most an
``alpha`` fraction of future nominal rollouts?

Three strengthenings over plain split conformal are implemented:

1. ``rollout_scores`` -- reduce a trajectory to a single score before
   calibrating, so the guarantee is per-*rollout*, not per-timestep. A
   per-timestep guarantee at level alpha admits up to ``T * alpha`` false
   alarms over a horizon of T; that is usually not what anyone wants.

2. ``MondrianCalibrator`` -- group-conditional coverage. Plain split conformal
   gives ``P(no false alarm) >= 1 - alpha`` averaged over the calibration
   distribution. It says nothing about any particular task, object or regime.
   Mondrian conformal gives the claim *conditional on the stratum*.

3. ``robust_level`` -- quantile inflation for a declared distribution-shift
   budget, so the claim survives a bounded change between calibration and
   deployment. When the budget is too large to carry the claim, this raises.

All bounds here are finite-sample and assume only exchangeability of the
calibration scores with the deployment score. No asymptotics, no Gaussianity,
no assumption about the policy or the score function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np

from .errors import (
    DegenerateScores,
    GroupTooSmall,
    InsufficientCalibrationData,
    ShiftBudgetExceeded,
)

__all__ = [
    "rollout_scores",
    "min_calibration_size",
    "SplitConformalCalibrator",
    "MondrianCalibrator",
    "robust_level",
    "kl_worst_case",
]


# --------------------------------------------------------------------------- #
# Trajectory reduction
# --------------------------------------------------------------------------- #


def rollout_scores(trajectories: Sequence[Sequence[float]]) -> np.ndarray:
    """Reduce each trajectory of per-timestep scores to a single rollout score.

    Uses the running maximum. A monitor that fires when *any* timestep exceeds
    the threshold triggers on a rollout exactly when the rollout max exceeds it,
    so calibrating on the max yields an exact rollout-level guarantee with no
    union bound and no multiplicity correction.

    Empty trajectories are rejected: a rollout with no observations cannot be
    scored, and silently mapping it to ``-inf`` would make it un-flaggable.
    """
    if len(trajectories) == 0:
        raise DegenerateScores("no trajectories supplied")
    out = np.empty(len(trajectories), dtype=float)
    for i, traj in enumerate(trajectories):
        arr = np.asarray(traj, dtype=float)
        if arr.size == 0:
            raise DegenerateScores(f"trajectory {i} is empty")
        if not np.all(np.isfinite(arr)):
            raise DegenerateScores(f"trajectory {i} contains non-finite scores")
        out[i] = float(arr.max())
    return out


# --------------------------------------------------------------------------- #
# Sample-size feasibility
# --------------------------------------------------------------------------- #


def min_calibration_size(alpha: float) -> int:
    """Smallest N for which a finite split-conformal threshold exists at ``alpha``.

    The threshold is the ``k``-th ascending order statistic with
    ``k = ceil((N + 1) * (1 - alpha))``. A finite threshold requires ``k <= N``,
    which holds iff ``N + 1 >= 1 / alpha``.
    """
    _check_alpha(alpha)
    n = math.ceil(1.0 / alpha) - 1
    # Guard against floating point landing just under the boundary.
    while math.ceil((n + 1) * (1.0 - alpha)) > n:
        n += 1
    return n


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SplitConformalCalibrator:
    """A calibrated one-sided threshold with a finite-sample coverage guarantee.

    For a fresh nominal score ``S`` exchangeable with the calibration sample::

        P(S <= threshold) >= 1 - alpha

    and, when calibration scores are almost surely distinct::

        P(S <= threshold) <= 1 - alpha + 1 / (n + 1)

    The lower bound is the guarantee. The upper bound is what stops the monitor
    from being trivially conservative -- it certifies that the threshold is not
    much looser than it needs to be, which is what makes the false-alarm rate
    *informative* rather than merely bounded.
    """

    threshold: float
    alpha: float
    n: int
    order_statistic: int

    @property
    def coverage_lower(self) -> float:
        """Proven lower bound on P(no false alarm)."""
        return 1.0 - self.alpha

    @property
    def coverage_upper(self) -> float:
        """Upper bound on P(no false alarm), valid for a.s. distinct scores."""
        return min(1.0, 1.0 - self.alpha + 1.0 / (self.n + 1))

    @property
    def false_alarm_bound(self) -> float:
        """Proven upper bound on P(monitor fires | nominal rollout)."""
        return self.alpha

    def fires(self, score: float) -> bool:
        """True iff this score should trigger abstention."""
        return float(score) > self.threshold

    @classmethod
    def fit(
        cls, scores: Sequence[float] | np.ndarray, alpha: float
    ) -> "SplitConformalCalibrator":
        _check_alpha(alpha)
        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 1:
            raise DegenerateScores("calibration scores must be one-dimensional")
        n = arr.size
        if n == 0:
            raise InsufficientCalibrationData("calibration sample is empty")
        if not np.all(np.isfinite(arr)):
            raise DegenerateScores("calibration scores contain non-finite values")

        k = math.ceil((n + 1) * (1.0 - alpha))
        if k > n:
            raise InsufficientCalibrationData(
                f"alpha={alpha:g} requires at least n={min_calibration_size(alpha)} "
                f"calibration rollouts to admit a finite threshold; got n={n}. "
                f"Collect more nominal data or accept a larger alpha. "
                f"Refusing to return an infinite (never-fires) threshold."
            )

        ordered = np.sort(arr, kind="stable")
        return cls(
            threshold=float(ordered[k - 1]),
            alpha=float(alpha),
            n=int(n),
            order_statistic=int(k),
        )


# --------------------------------------------------------------------------- #
# Mondrian / group-conditional conformal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MondrianCalibrator:
    """Group-conditional thresholds: one calibrated threshold per stratum.

    Guarantee, for every stratum ``g`` present at fit time::

        P(S <= threshold[g] | G = g) >= 1 - alpha

    This is strictly stronger than the marginal claim of plain split conformal,
    and it is the claim that matters operationally -- "safe on average across
    tasks" is not a useful statement to make to whoever owns the robot.

    Unseen strata at deployment are *not* covered by any claim, so
    ``calibrator_for`` raises rather than falling back to a pooled threshold.
    """

    thresholds: Mapping[Hashable, SplitConformalCalibrator]
    alpha: float

    def calibrator_for(self, group: Hashable) -> SplitConformalCalibrator:
        try:
            return self.thresholds[group]
        except KeyError:
            raise GroupTooSmall(
                f"stratum {group!r} was not present at calibration time, so no "
                f"conditional guarantee covers it. Refusing to substitute a "
                f"pooled threshold, which would silently weaken the claim to "
                f"marginal coverage."
            ) from None

    def fires(self, score: float, group: Hashable) -> bool:
        return self.calibrator_for(group).fires(score)

    @property
    def false_alarm_bound(self) -> float:
        return self.alpha

    @classmethod
    def fit(
        cls,
        scores: Sequence[float] | np.ndarray,
        groups: Sequence[Hashable],
        alpha: float,
    ) -> "MondrianCalibrator":
        _check_alpha(alpha)
        arr = np.asarray(scores, dtype=float)
        if arr.size != len(groups):
            raise DegenerateScores(
                f"scores ({arr.size}) and groups ({len(groups)}) length mismatch"
            )

        buckets: dict[Hashable, list[float]] = {}
        for s, g in zip(arr.tolist(), groups):
            buckets.setdefault(g, []).append(s)

        need = min_calibration_size(alpha)
        thin = {g: len(v) for g, v in buckets.items() if len(v) < need}
        if thin:
            raise GroupTooSmall(
                f"alpha={alpha:g} needs >= {need} rollouts per stratum for a "
                f"conditional guarantee; understocked strata: {thin}. "
                f"Either collect more data for these strata, coarsen the "
                f"partition, or raise alpha."
            )

        return cls(
            thresholds={
                g: SplitConformalCalibrator.fit(np.asarray(v), alpha)
                for g, v in buckets.items()
            },
            alpha=float(alpha),
        )


# --------------------------------------------------------------------------- #
# Distribution-shift robustness
# --------------------------------------------------------------------------- #


def kl_worst_case(beta: float, rho: float, *, tol: float = 1e-12) -> float:
    """Worst-case probability under a KL ball.

    Returns ``sup{ z : d(z || beta) <= rho }`` where ``d`` is the binary KL
    divergence. By the data-processing inequality for KL, if
    ``KL(Q || P) <= rho`` and ``P(A) = beta``, then ``Q(A)`` is at most this
    value. So an event that is ``beta``-rare under calibration can be at most
    this common under any deployment distribution inside the ball.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must lie in (0, 1); got {beta!r}")
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative; got {rho!r}")
    if rho == 0.0:
        return beta

    def d(z: float) -> float:
        if z <= 0.0:
            return math.log(1.0 / (1.0 - beta))
        if z >= 1.0:
            return math.log(1.0 / beta)
        return z * math.log(z / beta) + (1.0 - z) * math.log((1.0 - z) / (1.0 - beta))

    if d(1.0 - tol) <= rho:
        return 1.0

    lo, hi = beta, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if d(mid) <= rho:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def robust_level(alpha: float, rho: float, *, divergence: str = "tv") -> float:
    """Inflate the calibration level so the claim survives bounded shift.

    Calibrating at the returned level ``alpha'`` on the calibration
    distribution ``P`` yields a false-alarm bound of ``alpha`` under *any*
    deployment distribution ``Q`` within divergence ``rho`` of ``P``.

    ``divergence="tv"``
        Total variation. For any event, ``|P(A) - Q(A)| <= rho``, so
        ``alpha' = alpha - rho``. Requires ``rho < alpha``.

    ``divergence="kl"``
        Kullback-Leibler. Solves for the largest ``alpha'`` with
        ``kl_worst_case(alpha', rho) <= alpha``.

    Raises ``ShiftBudgetExceeded`` when the budget cannot be carried. This is
    the intended behaviour: a shift larger than the error budget makes the
    guarantee unattainable, and the correct output is a refusal to certify
    rather than a threshold that quietly no longer means what it says.
    """
    _check_alpha(alpha)
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative; got {rho!r}")
    if rho == 0.0:
        return alpha

    if divergence == "tv":
        if rho >= alpha:
            raise ShiftBudgetExceeded(
                f"total-variation budget rho={rho:g} is not smaller than the "
                f"risk level alpha={alpha:g}. Under a shift this large, an event "
                f"of any calibration probability can reach probability alpha at "
                f"deployment, so no threshold carries the claim. Tighten the "
                f"shift budget, raise alpha, or recalibrate on-distribution."
            )
        return alpha - rho

    if divergence == "kl":
        lo, hi = 1e-12, alpha
        if kl_worst_case(lo, rho) > alpha:
            raise ShiftBudgetExceeded(
                f"KL budget rho={rho:g} is too large for alpha={alpha:g}: even a "
                f"vanishing calibration-time false-alarm rate can be inflated "
                f"past alpha inside this ball. Refusing to certify."
            )
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if kl_worst_case(mid, rho) <= alpha:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14:
                break
        return lo

    raise ValueError(f"unknown divergence {divergence!r}; use 'tv' or 'kl'")
