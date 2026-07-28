"""M5: the two Phase-2 witnesses, binding-checked, and cover membership.

**VerifiedDiscrepancyWitness (W1, direct/one-step).** Spec section 1: "the
certified epsilon plugs straight into Phase 1's ``CertifiedModelErrorWitness``,
unchanged." That is true of the *math*; it is only *safe* once something
checks that the epsilon being plugged in was actually proven for the network
and reference in front of you right now. This wraps ``CertifiedModelErrorWitness``
with exactly the spec A4/A1 checks that make the reuse sound: a weight-hash
mismatch or a reference-identity mismatch refuses at construction ("at
load"), not at some later runtime moment when the unproven bound has already
been trusted.

**PredictiveTubeWitness (W2, predictive/K-step).** Spec Lemma L4: given the
M4 tube's certified lower bound on clearance at every step,
``s = max_t (c_required - lo(IA[h](X_t)))``. ``s <= 0`` implies true clearance
``>= c_required`` for every ``t <= K`` -- the same violation-floor-zero
structure as W1 and Phase 1's ``MarginWitness``, now with lookahead.

Both classes expose a ``covers(point)`` predicate: the thing M5 wires into
``ActionGate`` as its cover-membership check (spec section 3: "the Phase 1
core is untouched except for one addition to the gate"). Outside that cover,
the certificate underneath either witness says nothing at all, and the gate
must abstain rather than let a score computed there be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discrepancy import EpsilonCertificate
from .errors import NetworkCertificateMismatch, ReferenceMismatch
from .nnbound import MLP
from .soundness import CertifiedModelErrorWitness
from .tube import TubeResult, clearance_lower_bounds

__all__ = ["VerifiedDiscrepancyWitness", "PredictiveTubeWitness"]


@dataclass(frozen=True, slots=True)
class VerifiedDiscrepancyWitness:
    """W1: a certified-epsilon witness, bound to the exact network and reference.

    Construct only through :meth:`bind`, which refuses (spec A4, A1) when the
    supplied network's weight-hash or the supplied reference's identity
    string does not match what the certificate was actually proven against.
    """

    certificate: EpsilonCertificate
    inner: CertifiedModelErrorWitness

    @classmethod
    def bind(
        cls, certificate: EpsilonCertificate, net: MLP, reference_id: str
    ) -> "VerifiedDiscrepancyWitness":
        if not certificate.matches_network(net):
            raise NetworkCertificateMismatch(
                "VerifiedDiscrepancyWitness.bind: net's weight-hash does not "
                "match the certificate's bound network. A retrained network, "
                "a fine-tuned network, or a single flipped weight byte each "
                "void the certificate detectably, exactly as spec A4 requires."
            )
        if reference_id != certificate.reference_id:
            raise ReferenceMismatch(
                f"VerifiedDiscrepancyWitness.bind: reference_id {reference_id!r} "
                f"does not match the certificate's bound reference "
                f"{certificate.reference_id!r}. A changed stiffness, timestep, "
                f"or other reference parameter changes this string; the "
                f"certificate is only valid for the exact reference it was "
                f"proven against (spec A1)."
            )
        epsilon = float(np.max(certificate.eps))
        return cls(certificate=certificate, inner=CertifiedModelErrorWitness(epsilon=epsilon))

    # -- SoundnessWitness protocol ------------------------------------------- #

    def violation_floor(self) -> float:
        return self.inner.violation_floor()

    def justification(self) -> str:
        return (
            f"{self.inner.justification()}; epsilon certified (not assumed) by "
            f"{self.certificate.summary()}"
        )

    def miss_probability(self, threshold: float) -> float:
        return self.inner.miss_probability(threshold)

    # -- convenience ---------------------------------------------------------- #

    def score(self, g_hat):
        return self.inner.score(g_hat)

    def covers(self, point) -> bool:
        """Cover-membership predicate: is ``point`` inside the certified cover?"""
        return bool(self.certificate.contains(np.asarray(point, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class PredictiveTubeWitness:
    """W2: the K-step predictive tube witness (spec Lemma L4).

    Construct only through :meth:`build`, from an already-propagated M4
    :class:`~certabstain.tube.TubeResult` (which itself refuses a mismatched
    network -- spec A4 -- before any tube exists to build a witness from).
    """

    tube: TubeResult
    clearance_lo: np.ndarray
    c_required: float

    @classmethod
    def build(cls, tube: TubeResult, clearance_batched, c_required: float) -> "PredictiveTubeWitness":
        clearance_lo = clearance_lower_bounds(tube, clearance_batched)
        return cls(tube=tube, clearance_lo=clearance_lo, c_required=float(c_required))

    # -- SoundnessWitness protocol ------------------------------------------- #

    def violation_floor(self) -> float:
        return 0.0

    def justification(self) -> str:
        return (
            f"score s = max_t(c_required - lo(IA[h](X_t))) over t <= "
            f"K={self.tube.horizon} (requested {self.tube.requested_horizon}); "
            f"s <= 0 implies true clearance >= {self.c_required:g} for every "
            f"t <= K, so silence is sound over the whole certified horizon"
        )

    def miss_probability(self, threshold: float) -> float:
        return 0.0 if threshold < 0.0 else 1.0

    # -- convenience ---------------------------------------------------------- #

    def score(self) -> float:
        """s = max_t (c_required - lo(IA[h](X_t))); spec L4."""
        return float(np.max(self.c_required - self.clearance_lo))

    def covers(self, point) -> bool:
        """Is ``point`` inside the tube's certified reachable envelope?

        True iff it lies in at least one of the tube's boxes up to the
        achieved horizon (``tube.horizon``, which may be short of the
        requested one -- spec section 5's "fails to a smaller domain, never
        a weaker claim" applied to a time horizon). A point outside every
        box is outside everything this witness's guarantee was ever proven
        over, at any certified time step.
        """
        p = np.asarray(point, dtype=np.float64)
        return any(
            bool(np.all(box.contains(p)))
            for box in self.tube.boxes[: self.tube.horizon + 1]
        )
