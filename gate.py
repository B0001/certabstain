"""The invariant: no actuation without a fresh, valid certificate.

Most safety layers are a function the control loop is *supposed* to call. That
makes safety a convention, and conventions are bypassed -- by a refactor, by a
debug flag left on, by a code path that was added later and did not know about
the check.

Here the check is structural. The gate is the only object that can produce an
actuator command, and it will not produce one without a certificate that:

  * carries a valid authentication tag (unforgeable without the authority's key)
  * was minted for the current control epoch (no staleness)
  * binds the exact observation and action being emitted (no substitution)
  * has not been used before (no replay)

The policy never holds the signing key, so it cannot mint its own permission:
the gate exposes only :class:`CertificateVerifier`, which checks a tag but
cannot create one. There is no ``force=``, no ``override=``, no
``strict=False``, and no way to supply the authority the gate verifies
against -- it mints its own, so nobody else holds the key or the minting
capability for it. Any exception anywhere in the path fails closed to the
configured safe action rather than propagating a live command.

What that does and does not mean: no supported call sequence through this
module's public surface yields an uncertified emission, and an allowlist-based
signature scan (test_certabstain.py) fails the build if the surface grows a
parameter or attribute this design did not bless. It is not a defense against
hostile code running in the same interpreter, which can always reach a private
attribute; the deployment-level form of that separation is an authority in its
own process or an HSM.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Final

import numpy as np

from .errors import (
    BindingMismatch,
    CertAbstainError,
    ForgedCertificate,
    MissingCertificate,
    ReplayedCertificate,
    StaleCertificate,
)
from .soundness import TwoSidedClaim

__all__ = [
    "SafetyCertificate",
    "CertificateAuthority",
    "CertificateVerifier",
    "ActionGate",
    "GateDecision",
]

_DIGEST: Final = hashlib.blake2b


def _canonical(obj: Any) -> bytes:
    """Stable byte encoding for binding. Arrays hash by exact contents and shape."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return b"nd|" + str(arr.dtype).encode() + b"|" + str(arr.shape).encode() + b"|" + arr.tobytes()
    if isinstance(obj, (bytes, bytearray)):
        return b"by|" + bytes(obj)
    if isinstance(obj, str):
        return b"st|" + obj.encode()
    if isinstance(obj, (int, float, bool)):
        return b"sc|" + repr(obj).encode()
    if isinstance(obj, (list, tuple)):
        return b"sq|" + b"\x1f".join(_canonical(x) for x in obj)
    if obj is None:
        return b"nn|"
    raise TypeError(
        f"cannot canonically encode {type(obj).__name__} for certificate binding; "
        f"convert to array, scalar, string or sequence first"
    )


def _bind(observation: Any, action: Any, epoch: int) -> bytes:
    h = _DIGEST(digest_size=32)
    h.update(b"obs")
    h.update(_canonical(observation))
    h.update(b"act")
    h.update(_canonical(action))
    h.update(b"epoch")
    h.update(str(int(epoch)).encode())
    return h.digest()


# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SafetyCertificate:
    """Permission to actuate, valid for exactly one command at one epoch.

    Frozen and slot-based: fields cannot be mutated or monkey-patched after
    minting. The ``tag`` is a MAC over the binding and the claim, so altering
    any field invalidates it.
    """

    binding: bytes
    epoch: int
    nonce: bytes
    tag: bytes
    false_alarm_bound: float
    miss_bound: float | None
    score: float
    threshold: float

    @property
    def is_two_sided(self) -> bool:
        return self.miss_bound is not None

    def summary(self) -> str:
        miss = "UNBOUNDED" if self.miss_bound is None else f"{self.miss_bound:.4g}"
        return (
            f"epoch={self.epoch} score={self.score:.6g} thr={self.threshold:.6g} "
            f"P(false alarm)<={self.false_alarm_bound:.4g} P(miss)<={miss}"
        )


class CertificateAuthority:
    """Mints certificates. Holds the only key; the policy never sees it."""

    __slots__ = ("_key", "_epoch")

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def advance(self) -> int:
        """Begin a new control cycle, invalidating every outstanding certificate."""
        self._epoch += 1
        return self._epoch

    def _tag(
        self,
        binding: bytes,
        nonce: bytes,
        epoch: int,
        fa: float,
        miss: float | None,
        score: float,
        threshold: float,
    ) -> bytes:
        # Every field of SafetyCertificate is covered here. score/threshold are
        # not used to authorize the emission (the gate compares those itself
        # before minting), but they are what an auditor reads back out of the
        # log -- so leaving them outside the MAC would make the certificate's
        # own summary() unauthenticated, re-labellable data.
        msg = b"|".join(
            [
                binding,
                nonce,
                str(int(epoch)).encode(),
                repr(float(fa)).encode(),
                b"none" if miss is None else repr(float(miss)).encode(),
                repr(float(score)).encode(),
                repr(float(threshold)).encode(),
            ]
        )
        return hmac.new(self._key, msg, _DIGEST).digest()

    def mint(
        self,
        *,
        observation: Any,
        action: Any,
        score: float,
        threshold: float,
        false_alarm_bound: float,
        miss_bound: float | None,
    ) -> SafetyCertificate:
        binding = _bind(observation, action, self._epoch)
        nonce = secrets.token_bytes(16)
        tag = self._tag(
            binding, nonce, self._epoch, false_alarm_bound, miss_bound,
            score, threshold,
        )
        return SafetyCertificate(
            binding=binding,
            epoch=self._epoch,
            nonce=nonce,
            tag=tag,
            false_alarm_bound=float(false_alarm_bound),
            miss_bound=None if miss_bound is None else float(miss_bound),
            score=float(score),
            threshold=float(threshold),
        )

    def verify(self, cert: SafetyCertificate) -> bool:
        expected = self._tag(
            cert.binding,
            cert.nonce,
            cert.epoch,
            cert.false_alarm_bound,
            cert.miss_bound,
            cert.score,
            cert.threshold,
        )
        return hmac.compare_digest(expected, cert.tag)


# --------------------------------------------------------------------------- #


class CertificateVerifier:
    """Read-only view of an authority: check a tag, read the epoch. No mint.

    This is what :class:`ActionGate` hands out, because the gate's guarantee is
    that the *policy cannot mint its own permission*. Exposing the authority
    itself would hand out a mint oracle that performs no score, threshold or
    cover check -- enough to manufacture a verifying certificate for any
    observation/action pair on a gate that would otherwise always abstain.

    Scope, stated plainly: this removes minting from the gate's **API
    surface**. It is not memory isolation, and no such claim is made -- in
    CPython, code running in-process can always reach a private attribute or a
    closure cell. The separation this models is deployment-level (the authority
    living in its own process or an HSM, reachable only by an authenticated
    request); in-process, the guarantee is that no supported call sequence
    yields a certificate the gate did not itself authorize.
    """

    __slots__ = ("_authority",)

    def __init__(self, authority: "CertificateAuthority") -> None:
        self._authority = authority

    @property
    def epoch(self) -> int:
        return self._authority.epoch

    def verify(self, cert: SafetyCertificate) -> bool:
        """True iff this certificate's tag authenticates under the CA's key."""
        return self._authority.verify(cert)


# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GateDecision:
    """What the gate actually emitted, and why."""

    action: Any
    abstained: bool
    reason: str
    certificate: SafetyCertificate | None = None

    def __bool__(self) -> bool:
        return not self.abstained


class ActionGate:
    """The only path from a proposed action to an emitted command.

    Construct with a monitor score function, a calibrated threshold, and a safe
    fallback. Call :meth:`step`. There is no other entry point and no way to
    obtain the fallback-free behaviour.
    """

    __slots__ = (
        "_authority", "_threshold", "_fa_bound", "_claim", "_safe_action",
        "_spent", "_log", "_cover",
    )

    def __init__(
        self,
        *,
        threshold: float,
        false_alarm_bound: float,
        safe_action: Any,
        claim: TwoSidedClaim | None = None,
        cover: Callable[[Any], bool] | None = None,
    ) -> None:
        # The gate mints its own authority and never accepts one. An injected
        # authority was an override path in two directions at once: a
        # duck-typed (or subclassed) object whose verify() returns True makes
        # the gate accept anything, and even a genuine CertificateAuthority
        # leaves the *caller* holding mint() -- an oracle that applies no
        # score, threshold or cover check -- for the very instance this gate
        # verifies against. Owning the authority outright is what makes "the
        # policy cannot mint its own permission" true rather than aspirational:
        # a caller's own CertificateAuthority has a different key, so nothing
        # it mints will verify here.
        self._authority = CertificateAuthority()
        thr = float(threshold)
        fa = float(false_alarm_bound)
        if not np.isfinite(thr):
            # `score > threshold` is False for NaN and for +inf, so a
            # non-finite threshold silently certifies every score presented --
            # the strict-off switch this class's docstring denies exists.
            raise ValueError(
                f"threshold must be finite; got {threshold!r}. A non-finite "
                f"threshold makes the abstention comparison vacuously false, "
                f"so every score would certify."
            )
        if not (np.isfinite(fa) and 0.0 <= fa <= 1.0):
            raise ValueError(
                f"false_alarm_bound must be a probability in [0, 1]; got "
                f"{false_alarm_bound!r}. It is recorded in every certificate "
                f"this gate mints, so a meaningless value would be attested."
            )
        self._threshold = thr
        self._fa_bound = fa
        self._claim = claim
        self._safe_action = safe_action
        self._spent: set[bytes] = set()
        self._log: list[GateDecision] = []
        # M5: an optional cover-membership predicate on the observation. Any
        # witness whose guarantee is only valid inside a certified domain
        # (VerifiedDiscrepancyWitness, PredictiveTubeWitness) supplies its own
        # ``covers`` method here; the gate abstains -- with a fixed, checkable
        # reason -- before it ever trusts a score computed outside that
        # domain. This is the one change M5 makes to Phase 1's gate.
        self._cover = cover

    @property
    def verifier(self) -> CertificateVerifier:
        """A mint-free view of the authority, for auditing emitted decisions."""
        return CertificateVerifier(self._authority)

    @property
    def log(self) -> tuple[GateDecision, ...]:
        return tuple(self._log)

    @property
    def miss_bound(self) -> float | None:
        return None if self._claim is None else self._claim.miss_bound

    # -- the invariant ------------------------------------------------------ #

    def _admit(self, cert: SafetyCertificate, observation: Any, action: Any) -> None:
        """Raise unless this certificate authorises exactly this emission now."""
        if cert is None:
            raise MissingCertificate("no certificate presented")
        if not self._authority.verify(cert):
            raise ForgedCertificate("certificate authentication tag did not verify")
        if cert.epoch != self._authority.epoch:
            raise StaleCertificate(
                f"certificate minted for epoch {cert.epoch}, current epoch is "
                f"{self._authority.epoch}"
            )
        if cert.nonce in self._spent:
            raise ReplayedCertificate("certificate already consumed")
        if not hmac.compare_digest(
            cert.binding, _bind(observation, action, self._authority.epoch)
        ):
            raise BindingMismatch(
                "certificate does not bind this observation/action pair"
            )

    def step(
        self,
        *,
        observation: Any,
        proposed_action: Any,
        score: float,
    ) -> GateDecision:
        """Emit ``proposed_action`` iff it can be certified; otherwise abstain.

        Fails closed on *any* internal error. A monitor that crashes is a
        monitor that cannot vouch for the action, which is indistinguishable
        from a monitor that says no.
        """
        try:
            # Inside the try: an authority whose advance() raises must abstain
            # like any other internal failure. Outside it, the exception
            # escaped step() and returned no GateDecision at all -- which a
            # caller wrapping step() in its own except could turn back into an
            # emission, defeating the promise in this docstring.
            self._authority.advance()

            if self._cover is not None and not self._cover(observation):
                decision = GateDecision(self._safe_action, True, "left certified domain")
                self._log.append(decision)
                return decision

            s = float(score)
            if not np.isfinite(s):
                decision = GateDecision(
                    self._safe_action, True, f"non-finite monitor score ({score!r})"
                )
                self._log.append(decision)
                return decision

            if s > self._threshold:
                decision = GateDecision(
                    self._safe_action,
                    True,
                    f"score {s:.6g} exceeds calibrated threshold "
                    f"{self._threshold:.6g}",
                )
                self._log.append(decision)
                return decision

            cert = self._authority.mint(
                observation=observation,
                action=proposed_action,
                score=s,
                threshold=self._threshold,
                false_alarm_bound=self._fa_bound,
                miss_bound=self.miss_bound,
            )
            self._admit(cert, observation, proposed_action)
            self._spent.add(cert.nonce)

            decision = GateDecision(
                proposed_action, False, "certified", certificate=cert
            )
            self._log.append(decision)
            return decision

        except CertAbstainError as exc:
            decision = GateDecision(
                self._safe_action, True, f"certificate refused: {exc}"
            )
            self._log.append(decision)
            return decision
        except Exception as exc:  # noqa: BLE001 -- fail closed, deliberately broad
            decision = GateDecision(
                self._safe_action, True, f"failed closed on {type(exc).__name__}: {exc}"
            )
            self._log.append(decision)
            return decision

    # -- statistics --------------------------------------------------------- #

    @property
    def abstention_rate(self) -> float:
        if not self._log:
            return 0.0
        return sum(d.abstained for d in self._log) / len(self._log)
