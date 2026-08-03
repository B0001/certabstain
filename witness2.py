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
from .errors import HorizonTooShort, NetworkCertificateMismatch, ReferenceMismatch
from .nnbound import MLP
from .soundness import CertifiedModelErrorWitness
from .tube import TubeResult, clearance_lower_bounds

__all__ = ["VerifiedDiscrepancyWitness", "PredictiveTubeWitness"]


class _RequireRequested:
    """Sentinel: default to requiring the horizon the tube was asked for.

    A distinct type rather than ``None``, because ``None`` is itself a
    meaningful value here -- it opts *out* of the requirement -- and the two
    must not be spelled the same way.
    """

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics only
        return "<the tube's requested horizon>"


_REQUIRE_REQUESTED = _RequireRequested()


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

    This is where a deployment declares the lookahead it needs (spec 7.8).
    ``propagate_tube`` has its own optional ``required_horizon``, but its real
    callers are the sweeps, which deliberately want the permissive default --
    so a requirement stated only there is stated nowhere a deployment passes.
    A tube truncated to one step would otherwise produce a witness whose score
    maxes over a single step, wearing the K-step interface and composing into a
    claim that reads as predictive.
    """

    tube: TubeResult
    clearance_lo: np.ndarray
    c_required: float
    required_horizon: int | None
    clearance_id: str | None

    @classmethod
    def build(
        cls,
        tube: TubeResult,
        clearance_batched,
        c_required: float,
        *,
        required_horizon: int | None = _REQUIRE_REQUESTED,  # type: ignore[assignment]
        clearance_id: str | None = None,
    ) -> "PredictiveTubeWitness":
        """Build the witness, or refuse if the tube is shorter than declared.

        Parameters
        ----------
        required_horizon
            The lookahead this deployment relies on. Omitted, it defaults to
            ``tube.requested_horizon`` -- you propagated K control boxes, so
            you are taken to need K steps. Pass an explicit ``int`` to require
            fewer, or an explicit ``None`` to accept whatever the tube achieved
            (best effort, recorded in :meth:`justification` so the weaker claim
            is visible in the audit log rather than implied by silence).
        """
        if required_horizon is _REQUIRE_REQUESTED:
            required_horizon = tube.requested_horizon

        if required_horizon is not None:
            # `bool` is an `int` in Python, so `required_horizon=True` would
            # otherwise read as "yes, require it" and quietly mean 1 -- the
            # weakest requirement there is, and near enough to the unsafe
            # default this argument exists to remove.
            if (
                isinstance(required_horizon, bool)
                or int(required_horizon) != required_horizon
                or required_horizon < 1
            ):
                raise ValueError(
                    f"required_horizon must be a positive whole number of "
                    f"steps; got {required_horizon!r}. Pass None to accept the "
                    f"achieved horizon instead of a non-positive requirement."
                )
            required_horizon = int(required_horizon)
            if required_horizon > tube.requested_horizon:
                raise HorizonTooShort(
                    f"this witness is declared to need {required_horizon} "
                    f"steps of lookahead, but the tube was only propagated for "
                    f"{tube.requested_horizon} (that many control boxes were "
                    f"supplied), so the requirement is unreachable by "
                    f"construction. Propagate the tube over the horizon the "
                    f"deployment actually needs."
                )
            if tube.horizon < required_horizon:
                raise HorizonTooShort(
                    f"this witness is declared to need {required_horizon} "
                    f"steps of lookahead; the tube certifies {tube.horizon}. "
                    f"{tube.cover_exit_reason or 'The tube fell short of the requirement.'} "
                    f"A witness built on it would score over t <= "
                    f"{tube.horizon} while presenting the K-step guarantee. "
                    f"Shrink the requirement, widen the certified cover, or "
                    f"re-certify over a domain the tube stays inside."
                )

        # The clearance geometry is a *second* reference, independent of the
        # dynamics one the tube carries, and nothing recorded it. Two witnesses
        # built on the same tube against completely different obstacles produced
        # byte-identical justification strings, so the audit log could not tell
        # which geometry the clearance claim was actually proved for.
        #
        # There is no stored ground truth to check it against -- the tube knows
        # nothing about obstacles -- so the honest move is to record it, not to
        # invent a comparison. A bound method (``clear.interval_batch``) carries
        # its owner, so the usual call site needs no change; a bare function or
        # lambda has no identity and is recorded as undeclared rather than
        # passed off as a named geometry. Same idiom as required_horizon=None:
        # the weaker claim is visible in the audit log, not implied by silence.
        if clearance_id is None:
            owner = getattr(clearance_batched, "__self__", None)
            rid = getattr(owner, "reference_id", None)
            if callable(rid):
                clearance_id = str(rid())

        clearance_lo = clearance_lower_bounds(tube, clearance_batched)
        return cls(
            tube=tube,
            clearance_lo=clearance_lo,
            c_required=float(c_required),
            required_horizon=required_horizon,
            clearance_id=clearance_id,
        )

    # -- SoundnessWitness protocol ------------------------------------------- #

    def violation_floor(self) -> float:
        return 0.0

    def justification(self) -> str:
        declared = (
            "best effort (no horizon requirement declared)"
            if self.required_horizon is None
            else f"declared requirement {self.required_horizon}, met"
        )
        geometry = (
            "undeclared clearance geometry"
            if self.clearance_id is None
            else f"clearance h = {self.clearance_id}"
        )
        return (
            f"score s = max_t(c_required - lo(IA[h](X_t))) over t <= "
            f"K={self.tube.horizon} (requested {self.tube.requested_horizon}; "
            f"{declared}); s <= 0 implies true clearance >= "
            f"{self.c_required:g} for every t <= K, so silence is sound over "
            f"the whole certified horizon; tube certified against reference "
            f"{self.tube.reference_id!r}; {geometry}"
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
