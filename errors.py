"""Error taxonomy.

Every error in this package is a *refusal*, not a warning. The library never
degrades silently: if a guarantee cannot be established, construction fails
loudly rather than returning an object with a weaker-than-advertised claim.

This is the single most important design rule in the package. A monitor that
returns "probably fine" when it cannot prove "fine" is the exact failure mode
this library exists to eliminate.
"""

from __future__ import annotations


class CertAbstainError(Exception):
    """Base class. Catching this catches every refusal in the package."""


# ---------------------------------------------------------------- calibration


class InsufficientCalibrationData(CertAbstainError):
    """Not enough calibration samples to support the requested risk level.

    Split conformal prediction at level ``alpha`` requires the order statistic
    ``k = ceil((N+1)(1-alpha))`` to satisfy ``k <= N``. When it does not, the
    only threshold consistent with the guarantee is ``+inf`` -- i.e. a monitor
    that never fires. We refuse to build that object rather than hand back a
    vacuous certificate.
    """


class DegenerateScores(CertAbstainError):
    """Calibration scores are non-finite, or constant where variation is required."""


class GroupTooSmall(CertAbstainError):
    """A Mondrian (group-conditional) stratum has too few calibration samples.

    Group-conditional coverage is a *per-group* claim. Borrowing strength across
    groups would silently convert it back into a marginal claim, which is the
    weaker guarantee we are trying to escape.
    """


# ------------------------------------------------------------- shift robustness


class ShiftBudgetExceeded(CertAbstainError):
    """The declared distribution-shift budget is too large for the risk level.

    Under a total-variation ball of radius ``rho``, guaranteeing deployment-time
    false-alarm rate ``alpha`` requires calibrating at ``alpha - rho``. When
    ``rho >= alpha`` no threshold on the calibration distribution can carry the
    claim, and the honest output is a refusal.
    """


# -------------------------------------------------------------- two-sidedness


class SoundnessNotEstablished(CertAbstainError):
    """A two-sided certificate was requested but the miss-rate side is unproven.

    The false-alarm side comes from conformal calibration (distribution-free,
    finite-sample). The miss side does *not* come from statistics -- it comes
    from a structural property of the score. If that property does not hold at
    the calibrated threshold, the certificate is one-sided and must say so.
    """


# ---------------------------------------------------------------- gate / runtime


class CertificateError(CertAbstainError):
    """Base class for anything wrong with a certificate presented to the gate."""


class MissingCertificate(CertificateError):
    """An action reached the gate with no certificate attached."""


class ForgedCertificate(CertificateError):
    """Certificate authentication tag did not verify."""


class ReplayedCertificate(CertificateError):
    """Certificate was already consumed. Certificates are strictly single-use."""


class StaleCertificate(CertificateError):
    """Certificate was minted for a different control epoch."""


class BindingMismatch(CertificateError):
    """Certificate does not bind the action/state actually being emitted."""


# ------------------------------------------------------------------ numerics


class EnclosureError(CertAbstainError):
    """An interval enclosure could not be formed or would be unbounded.

    Raised for non-finite endpoints, inverted bounds, division by an interval
    containing zero, domain violations (sqrt below zero), and overflow under
    outward rounding. Returning [-inf, +inf] would be sound but vacuous; the
    honest output is a refusal that names the operation that lost the bound.
    """


class ModeIndeterminate(EnclosureError):
    """A box does not certifiably lie inside the declared operating mode.

    Spec section 7 item 3. Kept distinct from the other enclosure failures
    because the remedy is distinct: split the domain at the motion cone, or
    certify per mode. Nothing is numerically wrong here -- the arithmetic is
    fine and the box is simply straddling a discontinuity the certificate
    never declared.

    Subclasses :class:`EnclosureError`, so existing handlers that catch the
    broader class keep working.
    """


class NonFiniteEnclosure(EnclosureError):
    """A bound went non-finite somewhere on the certification path.

    Spec section 7 item 9. Covers non-finite endpoints, non-finite model or
    network parameters, and an operation whose result crossed the float64
    ceiling under outward rounding: in every case an endpoint is no longer a
    real number, so the enclosure would be vacuous rather than wrong. The
    remedy is to shrink the boxes or fix the upstream value, which is why this
    is worth telling apart from :class:`ModeIndeterminate`.

    Subclasses :class:`EnclosureError`, so existing handlers that catch the
    broader class keep working.
    """


class EnvironmentUnsound(CertAbstainError):
    """The floating-point environment failed the rounding self-test.

    Every certificate rests on two numeric facts: directed rounding via
    nextafter behaves as specified, and libm's exp/tanh are faithfully
    rounded. Both are checked against hard-coded high-precision references
    at startup. If the check fails, certification must not proceed here --
    a certificate minted on an unsound substrate would assert soundness it
    does not have.
    """


# ------------------------------------------------------------- certification


class TargetNotCertified(CertAbstainError):
    """Branch-and-bound exhausted its budget with epsilon above the target.

    The message reports the achieved bound, the empirical floor, and the
    worst uncertified leaf, so the caller can decide: raise the target,
    raise the budget, shrink the domain, or fix the model. What the caller
    cannot do is receive a certificate asserting a bound that was not
    established.
    """


class CoverTooSmall(CertAbstainError):
    """The certified cover falls below the declared minimum domain fraction.

    Partial certification is legitimate -- the certificate carries its exact
    cover and the gate abstains outside it -- but a certificate quietly
    covering a sliver of the intended envelope must not ship. The caller
    declares the minimum; this refusal enforces it.
    """


class HorizonTooShort(CertAbstainError):
    """The certified horizon fell below the horizon the deployment declared.

    Spec section 7.8 has two halves. A tube that leaves the certified cover
    early shrinks its *horizon*, never its claim -- that half is
    ``TubeResult.horizon`` plus ``cover_exit_reason``. But a caller that
    declared it needs K steps of lookahead and received fewer has not got the
    guarantee it asked for, and a truncated ``TubeResult`` is a value it can
    ignore. Declaring ``required_horizon`` converts that shortfall from a
    field the caller may overlook into a refusal it cannot.
    """


class NetworkCertificateMismatch(CertAbstainError):
    """A network was propagated against a certificate that does not bind it.

    Spec A4: the certificate binds a hash of the exact network weight bytes.
    Using it with any other network -- retrained, fine-tuned, or a single
    flipped byte -- would silently reuse an epsilon that was never proven for
    that network. Refused, not warned.
    """


class ReferenceMismatch(CertAbstainError):
    """A witness was bound against a certificate that does not cite this reference.

    Spec A1: the certificate is reference-relative, and binds the reference
    model's identity string (which parameters like stiffness, timestep, or
    geometry canonically determine). A changed reference parameter changes
    that string; using the certificate with a reference it doesn't match
    would silently certify against a system that was never analyzed. Refused,
    not warned.
    """
