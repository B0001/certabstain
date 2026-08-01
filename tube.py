"""K-step interval reachable tube (M4): spec Lemma L3.

Given a certified discrepancy bound (an :class:`~certabstain.discrepancy.
EpsilonCertificate`, from M3) for a learned one-step dynamics model ``g_hat``
against a trusted reference ``g``, this module propagates a box that is
guaranteed to contain the true reference trajectory for as many steps as the
precondition holds:

    X_t x U_t subset Cover
        ==>  g(X_t, U_t) subset IA[g_hat](X_t x U_t) (+) [-eps, eps]^n

Iterating while the precondition holds yields a tube containing every true
trajectory from X_0 under the declared open-loop control boxes U_0..U_{K-1}.
If the tube (or its controls) would leave the certified cover at step t*, the
certified horizon stops at t* - 1 (or at 0, if the very first joint box is
already uncertified) -- reported, never papered over: spec section 5's rule
that certification fails to a *smaller* claim, never a *weaker* one, applies
to time horizons exactly as it does to spatial domains.

Alongside the direct interval propagation, a scalar discrete-Grönwall bound is
computed from certified Jacobian enclosures (nnbound.jacobian_bounds): a mean-
value-theorem argument bounding the deviation of the true state from a
deterministic nominal (uncertified) reference rollout of g_hat. Both bounds
are sound, so their intersection is sound and never looser than either alone
-- the same "sound-meet-sound" composition nnbound.py already uses for
CROWN-intersect-IBP.

Clearance evaluation (the module's other M4 responsibility): given the tube,
compute the certified lower bound of an interval-valued clearance function at
every step, for later use by the W2 predictive witness (M5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .discrepancy import EpsilonCertificate
from .errors import EnclosureError, HorizonTooShort, NetworkCertificateMismatch
from .interval import Interval, require_sound_environment
from .nnbound import MLP, crown_bounds, jacobian_bounds

__all__ = ["TubeResult", "propagate_tube", "cover_contains_box", "clearance_lower_bounds"]


def _concat_iv(a: Interval, b: Interval) -> Interval:
    """Concatenate two 1-D intervals along their only axis. Exact: no rounding."""
    return Interval(np.concatenate([a.lo, b.lo]), np.concatenate([a.hi, b.hi]))


def cover_contains_box(cert: EpsilonCertificate, box: Interval, *, rtol: float = 1e-9) -> bool:
    """Sound (fail-closed) test: does ``box`` lie entirely inside ``cert``'s cover?

    The certifier's cover leaves are pairwise disjoint by construction (branch-
    and-bound keeps only leaves; a refined box is replaced by its children,
    never kept alongside them), so full coverage of ``box`` holds iff the
    volumes of ``box`` intersected with every leaf sum to the volume of
    ``box`` itself. A shortfall -- beyond a tiny relative tolerance for the
    summation's own rounding -- means some genuine sliver of ``box`` is
    uncovered, and this returns False rather than rounding it away: a step
    that should abstain must never be waved through by a tolerance bug.
    """
    box_vol = float(np.prod(box.hi - box.lo))
    if box_vol <= 0.0:
        return True  # degenerate zero-volume box: trivially covered or moot
    if cert.cover_lo.shape[0] == 0:
        return False
    lo = np.maximum(cert.cover_lo, box.lo[None, :])
    hi = np.minimum(cert.cover_hi, box.hi[None, :])
    widths = np.maximum(hi - lo, 0.0)
    covered = float(np.prod(widths, axis=1).sum())
    return covered >= box_vol * (1.0 - rtol)


@dataclass(frozen=True, slots=True)
class TubeResult:
    """The certified K-step tube, and the Grönwall alternative alongside it.

    ``boxes[t]`` soundly encloses the true reference state at time ``t``, for
    every ``t <= horizon``. ``horizon`` may fall short of ``requested_horizon``
    exactly when the containment-in-cover precondition failed; the reason is
    then in ``cover_exit_reason`` rather than silently truncating the tube.
    """

    boxes: tuple[Interval, ...]
    gronwall_boxes: tuple[Interval, ...]
    widths: np.ndarray            # (horizon+1, n_states), the reported boxes' widths
    gronwall_widths: np.ndarray   # (horizon+1, n_states), the Grönwall alternative alone
    horizon: int
    requested_horizon: int
    cover_exit_reason: str | None


def propagate_tube(
    net: MLP,
    cert: EpsilonCertificate,
    X0: Interval,
    controls: list[Interval],
    *,
    n_states: int,
    required_horizon: int | None = None,
    experimental: bool = False,
) -> TubeResult:
    """Propagate the certified interval tube (spec L3) for up to ``len(controls)`` steps.

    Parameters
    ----------
    net
        The learned one-step dynamics model ``g_hat`` the certificate was
        issued for. Checked against ``cert``'s bound weight-hash (spec A4);
        any other network raises :class:`NetworkCertificateMismatch`.
    cert
        The M3 :class:`EpsilonCertificate` for ``net`` against the trusted
        reference, over a domain covering ``(state, control)``.
    X0
        The certified initial state box, shape ``(n_states,)``.
    controls
        The declared open-loop control boxes ``U_0 .. U_{K-1}`` (v1: open-loop
        only, spec section 11); ``K = len(controls)`` is the requested horizon.
    n_states
        Dimensionality of the state; ``net.n_inputs - n_states`` is inferred
        as the control dimension.
    required_horizon
        The lookahead the deployment declares it needs. When given and the
        certified horizon comes in below it, this raises
        :class:`HorizonTooShort` (spec section 7.8) instead of returning a
        truncated tube the caller might use as though it reached ``K``.
        Left ``None``, the horizon shrinks and the shortfall is reported in
        ``cover_exit_reason`` for the caller to judge -- the right default for
        sweeps and studies that are *measuring* where the horizon collapses.

    Returns
    -------
    TubeResult
    """
    # Argument sanity first, before any work or any other refusal: a caller
    # that asked for a nonsensical horizon should hear about that, not about
    # the environment or the weight hash.
    if required_horizon is not None:
        # 0 and negatives would satisfy `horizon < required_horizon` nowhere
        # and silently behave as None -- a caller computing `k - 1` and landing
        # on 0 would get no refusal at all.
        if int(required_horizon) != required_horizon or required_horizon < 1:
            raise ValueError(
                f"required_horizon must be a positive whole number of steps; "
                f"got {required_horizon!r}. Pass None for the shrink-and-report "
                f"behaviour instead of a non-positive requirement."
            )
        required_horizon = int(required_horizon)

    require_sound_environment()
    if not cert.matches_network(net):
        raise NetworkCertificateMismatch(
            "propagate_tube: net does not match the network certify_epsilon "
            "bound this certificate to (weight-hash mismatch). Re-certify, or "
            "supply the exact network the epsilon was proven for."
        )
    if X0.lo.shape != (n_states,):
        raise EnclosureError(
            f"X0 has shape {X0.lo.shape}; expected ({n_states},) for n_states={n_states}"
        )
    eps = np.asarray(cert.eps, dtype=np.float64)
    if eps.shape[0] != n_states:
        raise EnclosureError(
            f"certificate eps has {eps.shape[0]} output dims; expected n_states={n_states} "
            f"(propagate_tube assumes g_hat predicts the next state directly)"
        )
    eps_iv = Interval(-eps, eps)

    K = len(controls)
    if required_horizon is not None and required_horizon > K:
        raise HorizonTooShort(
            f"required_horizon={required_horizon} exceeds the {K} control box(es) "
            f"supplied, so the requested lookahead is unreachable by construction. "
            f"Declare controls for every step the deployment needs."
        )
    boxes: list[Interval] = [X0]
    g_boxes: list[Interval] = [X0]
    widths: list[np.ndarray] = [np.array(X0.hi - X0.lo)]
    gronwall_widths: list[np.ndarray] = [np.array(X0.hi - X0.lo)]

    x_bar = 0.5 * (X0.lo + X0.hi)                       # nominal (uncertified) trajectory
    r = float(np.max(0.5 * (X0.hi - X0.lo))) if n_states else 0.0  # L-infinity radius bound

    horizon = 0
    exit_reason: str | None = None

    for t in range(K):
        U_t = controls[t]
        joint = _concat_iv(boxes[-1], U_t)
        if not cover_contains_box(cert, joint):
            exit_reason = (
                f"state x U_{t} left the certified cover before step {t}; "
                f"certified horizon stops at {horizon}"
            )
            break

        y_hat = crown_bounds(net, joint, experimental=experimental)
        x_next = y_hat + eps_iv  # Minkowski widening by [-eps, eps]: spec L3

        J = jacobian_bounds(net, joint, experimental=experimental)
        Jmag = np.maximum(np.abs(J.lo), np.abs(J.hi))
        L_x = float(Jmag[:, :n_states].sum(axis=1).max()) if n_states else 0.0
        L_u = float(Jmag[:, n_states:].sum(axis=1).max()) if Jmag.shape[1] > n_states else 0.0
        u_center = 0.5 * (U_t.lo + U_t.hi)
        u_halfwidth = float(np.max(0.5 * (U_t.hi - U_t.lo))) if U_t.lo.size else 0.0

        x_bar_next = net.forward(np.concatenate([x_bar, u_center]))
        r_next = L_x * r + L_u * u_halfwidth + float(np.max(eps))
        g_next = Interval(x_bar_next - r_next, x_bar_next + r_next)

        # Both x_next (CROWN+eps) and g_next (Grönwall) soundly enclose the
        # true next state; their intersection is sound too, and only ever
        # tighter -- the same argument nnbound.crown_bounds uses for
        # CROWN-intersect-IBP. An empty intersection here would mean the two
        # "sound" bounds contradict each other -- a real bug, not a case to
        # paper over -- and Interval's own constructor raises on it.
        combined = Interval(
            np.maximum(x_next.lo, g_next.lo), np.minimum(x_next.hi, g_next.hi)
        )

        boxes.append(combined)
        g_boxes.append(g_next)
        widths.append(np.array(combined.hi - combined.lo))
        gronwall_widths.append(np.full(n_states, 2.0 * r_next))
        x_bar, r = x_bar_next, r_next
        horizon = t + 1

    if required_horizon is not None and horizon < required_horizon:
        raise HorizonTooShort(
            f"certified horizon is {horizon} step(s); the deployment declared it "
            f"requires {required_horizon} (of {K} requested). "
            f"{exit_reason or 'The tube did not reach the required horizon.'} "
            f"Shrink the required horizon, widen the certified cover, or "
            f"re-certify over a domain the tube stays inside."
        )

    return TubeResult(
        boxes=tuple(boxes),
        gronwall_boxes=tuple(g_boxes),
        widths=np.array(widths),
        gronwall_widths=np.array(gronwall_widths),
        horizon=horizon,
        requested_horizon=K,
        cover_exit_reason=exit_reason,
    )


def clearance_lower_bounds(
    tube: TubeResult,
    clearance_batched: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Certified lower bound of an interval-valued clearance ``h`` at every tube step.

    ``clearance_batched`` follows the same ``(lo (B,d), hi (B,d)) -> (lo, hi)
    (B,m)`` contract as ``CircleClearance.interval_batch`` and the ``ref``
    callable in ``discrepancy.py``. Returns ``lo(IA[h](X_t))`` for
    ``t = 0 .. tube.horizon``, the quantity spec L4's score is built from
    (``s = max_t (c_required - lo(IA[h](X_t)))``); composing that score is the
    W2 witness's job (M5), not this module's.
    """
    lo = np.stack([b.lo for b in tube.boxes])
    hi = np.stack([b.hi for b in tube.boxes])
    clo, _ = clearance_batched(lo, hi)
    clo = np.asarray(clo, dtype=np.float64)
    return clo.reshape(lo.shape[0], -1)[:, 0]
