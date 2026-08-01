"""The discrepancy certifier (M3): epsilon produced, not assumed.

Given a learned network g_hat, a batched interval reference g, and a declared
domain D (optionally intersected with a mode region M), this module computes
a certified per-output bound

    sup over Cover of |g_hat_i(x) - g_i(x)|  <=  eps_i

by adaptive branch-and-bound, where every per-leaf bound is the two-enclosure
bound of spec Lemma L2 evaluated with sound arithmetic:

    u_B,i = max( hi[g_hat_i](B) - lo[g_i](B),  hi[g_i](B) - lo[g_hat_i](B) )

Enclosures for the network come from batched IBP on the M0 substrate during
refinement, with an optional CROWN polish on the worst surviving leaves
(both are sound, so the elementwise minimum is sound). Enclosures for the
reference come from its interval twin.

Partial certification, per spec section 5: boxes that certifiably lie
outside the mode are excluded; boxes that straddle the mode boundary are
split until a minimum width and then excluded -- certification fails to a
SMALLER DOMAIN, never to a weaker claim. The certificate carries its exact
cover, and refuses to exist at all if the cover falls below the declared
minimum fraction of the domain.

The certificate binds the exact network (weight-bytes hash) and the exact
reference (identity string): a retrained network or a changed reference
parameter voids it detectably, not silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import (
    CoverTooSmall,
    EnclosureError,
    NonFiniteEnclosure,
    TargetNotCertified,
)
from .gate import _canonical
from .interval import Interval, _down, _up, require_sound_environment
from .nnbound import MLP, crown_bounds

__all__ = ["EpsilonCertificate", "certify_epsilon", "weights_hash", "MODE_IN",
           "MODE_OUT", "MODE_STRADDLE"]

MODE_IN, MODE_OUT, MODE_STRADDLE = 1, -1, 0

# ref(lo (B,d), hi (B,d)) -> (rlo (B,m), rhi (B,m)), sound enclosures
RefFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
# mode(lo (B,d), hi (B,d)) -> (B,) ints in {MODE_IN, MODE_OUT, MODE_STRADDLE}
ModeFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def weights_hash(net: MLP) -> str:
    """BLAKE2b over the canonical serialization of every parameter tensor."""
    h = hashlib.blake2b(digest_size=32)
    h.update(net.activation.encode())
    for W, b in net.weights:
        h.update(_canonical(W))
        h.update(_canonical(b))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Batched sound IBP (refinement workhorse)
# --------------------------------------------------------------------------- #


def _batched_ibp(net: MLP, lo: np.ndarray, hi: np.ndarray):
    """Sound IBP over a stack of boxes: (B, n_in) -> (B, n_out) bounds.

    Same enclosure argument as interval.matvec, vectorized over boxes; the
    directed accumulation loop runs over the input width of each layer.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        for k, (W, b) in enumerate(net.weights):
            tl = np.where(W[None, :, :] >= 0.0, lo[:, None, :], hi[:, None, :])
            th = np.where(W[None, :, :] >= 0.0, hi[:, None, :], lo[:, None, :])
            TL = _down(tl * W[None, :, :])
            TH = _up(th * W[None, :, :])
            acc_lo = np.zeros((lo.shape[0], W.shape[0]))
            acc_hi = np.zeros((lo.shape[0], W.shape[0]))
            for j in range(W.shape[1]):
                acc_lo = _down(acc_lo + TL[:, :, j])
                acc_hi = _up(acc_hi + TH[:, :, j])
            acc_lo = _down(acc_lo + b)
            acc_hi = _up(acc_hi + b)
            if k < len(net.weights) - 1:
                if net.activation == "relu":
                    lo = np.maximum(acc_lo, 0.0)
                    hi = np.maximum(acc_hi, 0.0)
                else:
                    lo = np.maximum(_down(np.tanh(acc_lo), 2), -1.0)
                    hi = np.minimum(_up(np.tanh(acc_hi), 2), 1.0)
            else:
                return acc_lo, acc_hi
    raise AssertionError("unreachable")


def _gap_bound(net, ref: RefFn, lo, hi):
    """Two-enclosure per-box bound on |g_hat - g|: (B, n_out)."""
    nlo, nhi = _batched_ibp(net, lo, hi)
    rlo, rhi = ref(lo, hi)
    # All four endpoints, not just two: the returned bound reads nhi and rlo as
    # well, so checking only (nlo, rhi) let a half-infinite reference through
    # and minted a certificate with eps = inf -- vacuous, and sound only in the
    # useless sense. Spec 7.9 says non-finite *anywhere* on the path refuses.
    if not all(np.all(np.isfinite(a)) for a in (nlo, nhi, rlo, rhi)):
        raise NonFiniteEnclosure("bound computation lost finiteness; shrink boxes")
    return np.maximum(_up(nhi - rlo), _up(rhi - nlo))


# --------------------------------------------------------------------------- #
# The certificate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EpsilonCertificate:
    """A certified discrepancy bound, valid exactly on its cover.

    ``eps[i]`` bounds sup over the cover of |g_hat_i - g_i|. The cover is the
    explicit list of certified leaf boxes; everything in the requested domain
    but outside the cover is UNCERTIFIED, and the runtime witness (M5) must
    treat it as abstain territory. Bindings: the network's weight-bytes hash
    and the reference identity string.
    """

    eps: np.ndarray
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    cover_lo: np.ndarray          # (n_leaves, d)
    cover_hi: np.ndarray
    cover_fraction: float         # volume(cover) / volume(domain), diagnostic
    net_hash: str
    reference_id: str
    empirical_floor: np.ndarray   # max sampled |g_hat - g| per output dim
    n_leaf_evals: int
    n_leaves: int
    target: float | None

    def __post_init__(self) -> None:
        for name in ("eps", "domain_lo", "domain_hi", "cover_lo", "cover_hi",
                     "empirical_floor"):
            # np.array copies; np.asarray would hand back the caller's own
            # object when it is already float64, leaving the base array
            # writable through any view the caller kept -- so the read-only
            # flag below would protect nothing.
            arr = np.array(getattr(self, name), dtype=np.float64)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        # A certificate asserting an infinite bound asserts nothing. Refuse at
        # construction so no such object can be handed out, whatever path
        # produced the number.
        if not np.all(np.isfinite(self.eps)):
            raise NonFiniteEnclosure(
                f"certificate eps is non-finite ({self.eps!r}); an unbounded "
                f"epsilon is vacuous, not conservative. Shrink the domain or "
                f"fix the reference enclosure."
            )

    def matches_network(self, net: MLP) -> bool:
        return weights_hash(net) == self.net_hash

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Elementwise: does each point lie inside the certified cover?"""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        inside = np.zeros(p.shape[0], dtype=bool)
        for i in range(0, self.cover_lo.shape[0], 4096):
            L = self.cover_lo[i : i + 4096]
            H = self.cover_hi[i : i + 4096]
            hit = np.all(
                (L[None, :, :] <= p[:, None, :]) & (p[:, None, :] <= H[None, :, :]),
                axis=2,
            )
            inside |= hit.any(axis=1)
        return inside if points.ndim > 1 else inside[0]

    def summary(self) -> str:
        e = ", ".join(f"{v:.4g}" for v in np.atleast_1d(self.eps))
        f = ", ".join(f"{v:.4g}" for v in np.atleast_1d(self.empirical_floor))
        return (
            f"eps=[{e}] over {self.n_leaves} leaves "
            f"(cover {self.cover_fraction:.1%} of domain, "
            f"{self.n_leaf_evals} leaf evals); empirical floor [{f}]; "
            f"net {self.net_hash[:12]}...; ref {self.reference_id}"
        )


# --------------------------------------------------------------------------- #
# The certifier
# --------------------------------------------------------------------------- #


def _empirical_floor(net, ref_float, domain_lo, domain_hi, mode_float, n, seed):
    rng = np.random.default_rng(seed)
    floor = None
    remaining = n
    while remaining > 0:
        m = min(remaining, 500_000)
        pts = rng.uniform(domain_lo, domain_hi, size=(m, domain_lo.shape[0]))
        if mode_float is not None:
            pts = pts[mode_float(pts)]
            if pts.shape[0] == 0:
                remaining -= m
                continue
        gap = np.abs(net.forward(pts) - _as_2d(ref_float(pts)))
        gmax = gap.max(axis=0)
        floor = gmax if floor is None else np.maximum(floor, gmax)
        remaining -= m
    if floor is None:
        raise CoverTooSmall(
            "no sampled point of the domain lies in the declared mode; the "
            "requested region is empty as far as sampling can tell."
        )
    return floor


def _as_2d(y):
    y = np.asarray(y, dtype=np.float64)
    return y[:, None] if y.ndim == 1 else y


def certify_epsilon(
    net: MLP,
    ref: RefFn,
    domain: Interval,
    *,
    reference_id: str,
    ref_float: Callable[[np.ndarray], np.ndarray] | None = None,
    target: float | None = None,
    max_leaf_evals: int = 1_000_000,
    mode: ModeFn | None = None,
    mode_float: Callable[[np.ndarray], np.ndarray] | None = None,
    min_cover_fraction: float = 0.90,
    min_width_frac: float = 1e-3,
    floor_samples: int = 10_000_000,
    crown_polish: bool = True,
    batch: int = 1024,
    seed: int = 0,
) -> EpsilonCertificate:
    """Certify sup |g_hat - g| over domain-intersect-mode, or refuse.

    Raises TargetNotCertified when the budget runs out above ``target``
    (reporting the achieved bound, the empirical floor, and the worst leaf),
    and CoverTooSmall when mode exclusion leaves less than
    ``min_cover_fraction`` of the domain certified.
    """
    require_sound_environment()
    dlo = np.asarray(domain.lo, dtype=np.float64)
    dhi = np.asarray(domain.hi, dtype=np.float64)
    if dlo.ndim != 1 or dlo.shape[0] != net.n_inputs:
        raise EnclosureError(
            f"domain shape {dlo.shape} does not match net input {net.n_inputs}"
        )
    dwidth = dhi - dlo
    min_w = dwidth * min_width_frac
    dvol = float(np.prod(dwidth))

    floor = (
        _empirical_floor(net, ref_float, dlo, dhi, mode_float, floor_samples, seed)
        if ref_float is not None
        else np.zeros(net.n_outputs)
    )

    min_w = dwidth * min_width_frac
    excluded_vol = 0.0
    evals = 0

    def _split(L, H):
        w = (H - L) / dwidth[None, :]
        ax = np.argmax(w, axis=1)
        rows = np.arange(L.shape[0])
        mid = 0.5 * (L[rows, ax] + H[rows, ax])
        L2, H2 = L.copy(), H.copy()
        H2[rows, ax] = mid
        L3, H3 = L.copy(), H.copy()
        L3[rows, ax] = mid
        return np.concatenate([L2, L3]), np.concatenate([H2, H3])

    def _bound_chunked(L, H):
        nonlocal evals
        outs = []
        for i in range(0, L.shape[0], batch):
            outs.append(_gap_bound(net, ref, L[i : i + batch], H[i : i + batch]))
            evals += min(batch, L.shape[0] - i)
        return np.concatenate(outs) if outs else np.zeros((0, net.n_outputs))

    # -- phase A: mode filtering tiles the domain into certified-IN leaves -- #
    # (children of an IN box are IN because the predicates are monotone under
    #  inclusion, so mode is checked only here, never during refinement)
    pend_lo = dlo[None, :].copy()
    pend_hi = dhi[None, :].copy()
    in_lo_parts, in_hi_parts = [], []
    while pend_lo.shape[0] > 0:
        take = min(batch, pend_lo.shape[0])
        L, H = pend_lo[:take], pend_hi[:take]
        pend_lo, pend_hi = pend_lo[take:], pend_hi[take:]
        if mode is None:
            in_lo_parts.append(L)
            in_hi_parts.append(H)
            continue
        status = mode(L, H)
        out = status == MODE_OUT
        excluded_vol += float(np.prod(H[out] - L[out], axis=1).sum())
        keep = status == MODE_IN
        in_lo_parts.append(L[keep])
        in_hi_parts.append(H[keep])
        strad = status == MODE_STRADDLE
        tiny = np.all((H - L) <= min_w[None, :], axis=1)
        drop = strad & tiny
        excluded_vol += float(np.prod(H[drop] - L[drop], axis=1).sum())
        resplit = strad & ~tiny
        if np.any(resplit):
            sl, sh = _split(L[resplit], H[resplit])
            pend_lo = np.concatenate([pend_lo, sl])
            pend_hi = np.concatenate([pend_hi, sh])

    leaf_lo = np.concatenate(in_lo_parts) if in_lo_parts else np.zeros((0, dlo.shape[0]))
    leaf_hi = np.concatenate(in_hi_parts) if in_hi_parts else np.zeros((0, dlo.shape[0]))
    leaf_u = _bound_chunked(leaf_lo, leaf_hi)

    # -- phase B: worst-first refinement ------------------------------------ #
    K = max(1, batch // 2)
    while evals < max_leaf_evals and leaf_lo.shape[0] > 0:
        umax = leaf_u.max(axis=1) if leaf_u.size else np.zeros(0)
        refinable = ~np.all((leaf_hi - leaf_lo) <= min_w[None, :], axis=1)
        if target is not None:
            refinable &= umax > target
            if not np.any(umax > target):
                break
        if not np.any(refinable):
            break
        idx = np.flatnonzero(refinable)
        if idx.shape[0] > K:
            worst = idx[np.argpartition(-umax[idx], K - 1)[:K]]
        else:
            worst = idx
        sl, sh = _split(leaf_lo[worst], leaf_hi[worst])
        su = _bound_chunked(sl, sh)
        keep = np.ones(leaf_lo.shape[0], dtype=bool)
        keep[worst] = False
        leaf_lo = np.concatenate([leaf_lo[keep], sl])
        leaf_hi = np.concatenate([leaf_hi[keep], sh])
        leaf_u = np.concatenate([leaf_u[keep], su])

    done_lo, done_hi, done_u = [leaf_lo], [leaf_hi], [leaf_u]

    cov_lo = np.concatenate(done_lo) if done_lo else np.zeros((0, dlo.shape[0]))
    cov_hi = np.concatenate(done_hi) if done_hi else np.zeros((0, dlo.shape[0]))
    cov_u = np.concatenate(done_u) if done_u else np.zeros((0, net.n_outputs))

    cover_fraction = (
        float(np.prod(cov_hi - cov_lo, axis=1).sum() / dvol) if dvol > 0 else 0.0
    )
    if cover_fraction < min_cover_fraction:
        raise CoverTooSmall(
            f"certified cover is {cover_fraction:.1%} of the domain, below the "
            f"declared minimum {min_cover_fraction:.0%}. The mode predicate "
            f"excluded too much (excluded volume fraction "
            f"{excluded_vol / dvol:.1%}). Coarsen the mode, shrink the domain "
            f"to the mode, or lower min_cover_fraction deliberately."
        )

    # CROWN polish: tighten the leaves that define the maximum
    if crown_polish and cov_u.shape[0] > 0:
        umax = cov_u.max(axis=1)
        order = np.argsort(-umax)[: min(512, cov_u.shape[0])]
        for i in order:
            fin = crown_bounds(
                net,
                Interval(cov_lo[i], cov_hi[i]),
                experimental=(net.activation == "tanh"),
            )
            rlo, rhi = ref(cov_lo[i][None, :], cov_hi[i][None, :])
            u2 = np.maximum(_up(fin.hi - rlo[0]), _up(rhi[0] - fin.lo))
            cov_u[i] = np.minimum(cov_u[i], u2)  # both sound; min is sound
        evals += order.shape[0]

    eps = cov_u.max(axis=0) if cov_u.shape[0] else np.zeros(net.n_outputs)

    if target is not None and float(eps.max()) > target:
        wi = int(np.argmax(cov_u.max(axis=1)))
        raise TargetNotCertified(
            f"budget exhausted at {evals} leaf evaluations with achieved "
            f"eps={np.array2string(eps, precision=4)} > target {target:g}.\n"
            f"  empirical floor (sampled): "
            f"{np.array2string(floor, precision=4)}\n"
            f"  worst leaf: lo={np.array2string(cov_lo[wi], precision=4)} "
            f"hi={np.array2string(cov_hi[wi], precision=4)} "
            f"bound={cov_u[wi].max():.4g}\n"
            f"Remedies: raise the target above the floor, raise the budget, "
            f"shrink the domain, or improve the model."
        )

    return EpsilonCertificate(
        eps=eps,
        domain_lo=dlo,
        domain_hi=dhi,
        cover_lo=cov_lo,
        cover_hi=cov_hi,
        cover_fraction=cover_fraction,
        net_hash=weights_hash(net),
        reference_id=reference_id,
        empirical_floor=floor,
        n_leaf_evals=evals,
        n_leaves=int(cov_lo.shape[0]),
        target=target,
    )
