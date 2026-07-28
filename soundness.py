"""The miss-rate side of the certificate.

Standard runtime monitors for learned policies calibrate a failure score on
nominal rollouts and control the false-alarm rate. They provide no bound on
missed failures, and their own authors are explicit that obtaining one would
require failure rollouts to calibrate against.

This module takes a different route. Instead of *estimating* the miss rate from
failure data, it *constructs the score so that a miss is impossible* below a
known level, and then checks that the calibrated threshold sits below it.

    false-alarm side : statistical, distribution-free, from conformal
    miss side        : structural, deterministic, from the score's construction

Neither side alone is a two-sided guarantee. Composed, they are -- and no
failure data is needed anywhere.

The composition is only valid when the threshold clears the soundness level.
When it does not, ``TwoSidedClaim`` refuses to exist and the caller gets a
one-sided certificate that says so on its face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import SoundnessNotEstablished

__all__ = [
    "SoundnessWitness",
    "MarginWitness",
    "CertifiedModelErrorWitness",
    "EmpiricalPowerWitness",
    "TwoSidedClaim",
]


@runtime_checkable
class SoundnessWitness(Protocol):
    """Evidence that the score cannot be small when the specification is violated.

    Implementations must be able to justify ``violation_floor`` as a *theorem*
    about the score function, not as an empirical observation. The whole point
    of this interface is that the guarantee it carries does not degrade under
    distribution shift, because it is not a statistical claim at all.
    """

    def violation_floor(self) -> float:
        """A level ``m`` such that every specification-violating state scores > m."""

    def justification(self) -> str:
        """Human-readable statement of why the floor holds. Goes into the audit log."""

    def miss_probability(self, threshold: float) -> float:
        """Upper bound on P(violation occurs and monitor stays silent)."""


# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MarginWitness:
    """Score is the negated safety function: ``s(x) = -g(x)``, safe iff ``g(x) >= 0``.

    A violation means ``g(x) < 0``, hence ``s(x) > 0``. The floor is therefore
    ``0`` exactly, with no slack and no estimation. Any threshold strictly below
    zero catches every violation.

    Operationally: the calibrated threshold clears zero precisely when nominal
    rollouts keep a strictly positive safety margin at the ``1 - alpha``
    quantile. Systems that habitually skim the constraint boundary will not
    clear it -- correctly, because for such a system no monitor can separate
    nominal operation from violation.
    """

    def violation_floor(self) -> float:
        return 0.0

    def justification(self) -> str:
        return (
            "score s(x) = -g(x) with safety specification g(x) >= 0; "
            "g(x) < 0 implies s(x) > 0 identically"
        )

    def miss_probability(self, threshold: float) -> float:
        return 0.0 if threshold < 0.0 else 1.0


@dataclass(frozen=True, slots=True)
class CertifiedModelErrorWitness:
    """Score built on a *learned* safety function with a certified error bound.

    Given a learned ``g_hat`` and a proven uniform bound
    ``|g_hat(x) - g(x)| <= epsilon`` over the operating domain, define::

        s(x) = epsilon - g_hat(x)

    Then ``s(x) <= 0`` implies ``g_hat(x) >= epsilon`` implies ``g(x) >= 0``:
    silence is *sound*, the monitor never stays quiet through a real violation.
    Conversely ``g(x) < 0`` forces ``g_hat(x) < epsilon`` and so ``s(x) > 0``.
    The floor is ``0``.

    This is where a certified-bounds background pays off directly. ``epsilon``
    is the certified model error, and it enters the score as a pure additive
    conservatism: halving ``epsilon`` shifts every score down by the same amount
    and buys back exactly that much operating envelope before the monitor starts
    abstaining. Tighter bounds are not a cosmetic improvement -- they are the
    difference between a robot that works and one that abstains constantly.
    """

    epsilon: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError(
                f"epsilon must be a finite non-negative certified bound; "
                f"got {self.epsilon!r}"
            )

    def violation_floor(self) -> float:
        return 0.0

    def justification(self) -> str:
        return (
            f"score s(x) = epsilon - g_hat(x) with certified uniform model error "
            f"|g_hat - g| <= epsilon = {self.epsilon:g}; s(x) <= 0 implies "
            f"g(x) >= 0, so silence is sound"
        )

    def miss_probability(self, threshold: float) -> float:
        return 0.0 if threshold < 0.0 else 1.0

    def score(self, g_hat: float | np.ndarray) -> float | np.ndarray:
        """Apply the conservative transform to a learned safety-function value."""
        return self.epsilon - np.asarray(g_hat, dtype=float)


@dataclass(frozen=True, slots=True)
class EmpiricalPowerWitness:
    """Fallback for when failure rollouts *are* available.

    Gives a Clopper-Pearson upper confidence bound on the miss rate from
    observed failures. Honest and distribution-free, but it is a statistical
    claim about the failure distribution you sampled, so it carries none of the
    shift-robustness of the structural witnesses above. Provided because it is
    strictly better than nothing, and clearly labelled because it is strictly
    worse than a proof.
    """

    n_failures: int
    n_missed: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.n_failures <= 0:
            raise ValueError("n_failures must be positive")
        if not (0 <= self.n_missed <= self.n_failures):
            raise ValueError("n_missed must lie in [0, n_failures]")
        if not (0.0 < self.confidence < 1.0):
            raise ValueError("confidence must lie in (0, 1)")

    def violation_floor(self) -> float:
        return float("-inf")  # no structural floor; the claim is statistical

    def justification(self) -> str:
        return (
            f"Clopper-Pearson upper bound at confidence {self.confidence:g} from "
            f"{self.n_missed}/{self.n_failures} observed misses; STATISTICAL "
            f"claim about the sampled failure distribution, not shift-robust"
        )

    def miss_probability(self, threshold: float) -> float:  # noqa: ARG002
        from scipy.stats import beta as _beta

        if self.n_missed == self.n_failures:
            return 1.0
        return float(
            _beta.ppf(
                self.confidence, self.n_missed + 1, self.n_failures - self.n_missed
            )
        )


# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TwoSidedClaim:
    """A composed certificate: bounded false alarms *and* bounded misses.

    Constructed only through :meth:`compose`, which refuses when the calibrated
    threshold fails to clear the witness's floor. There is no way to build one
    that asserts a miss bound it cannot support.
    """

    false_alarm_bound: float
    miss_bound: float
    threshold: float
    violation_floor: float
    justification: str

    @classmethod
    def compose(
        cls,
        *,
        threshold: float,
        false_alarm_bound: float,
        witness: SoundnessWitness,
    ) -> "TwoSidedClaim":
        floor = witness.violation_floor()
        miss = witness.miss_probability(threshold)

        if miss > 0.0 and floor > float("-inf") and threshold >= floor:
            raise SoundnessNotEstablished(
                f"calibrated threshold {threshold:g} does not clear the violation "
                f"floor {floor:g}, so a violating state could score below the "
                f"threshold and pass unflagged. The miss rate is unbounded and no "
                f"two-sided claim is available.\n"
                f"  witness: {witness.justification()}\n"
                f"Remedies: tighten the certified model error, increase the "
                f"nominal safety margin, or accept a larger alpha (a looser "
                f"false-alarm budget lowers the threshold)."
            )

        return cls(
            false_alarm_bound=float(false_alarm_bound),
            miss_bound=float(miss),
            threshold=float(threshold),
            violation_floor=float(floor),
            justification=witness.justification(),
        )

    def summary(self) -> str:
        return (
            f"P(false alarm | nominal) <= {self.false_alarm_bound:.4g}   "
            f"P(miss | violation) <= {self.miss_bound:.4g}   "
            f"threshold={self.threshold:.6g} floor={self.violation_floor:.6g}"
        )
