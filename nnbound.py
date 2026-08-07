"""Sound network bounds over the interval substrate (M1).

IBP, backward CROWN, and interval Jacobians for small MLPs. ReLU is fully
supported; tanh sits behind ``experimental=True`` and uses the parallel-slope
(endpoint-derivative) relaxation rather than tangent-optimal lines.

The departure from standard CROWN implementations is the soundness model.
Published CROWN code computes relaxation slopes and the backward linear
algebra in ordinary floating point, making its bounds "sound up to float".
Here every relaxation coefficient is carried as an *interval enclosing the
exact real coefficient*, and all backward linear algebra uses directed
rounding on the M0 substrate. The argument:

  1. The exact-real relaxation (chord/tangent slopes computed from certified
     pre-activation bounds) is a valid linear bound on the activation --
     validity only needs the pre-activation range to be *contained* in the
     certified one, which it is.
  2. Every arithmetic step of the backward pass encloses the exact-real
     coefficients of that valid relaxation.
  3. Concretization over the input box with interval arithmetic then soundly
     bounds the exact linear functional, hence the network.

So the emitted bounds are certified, not approximately certified. The cost is
a few ulps of width and python loops sized by network width -- the networks
this package certifies are small by design (spec: input dim <= 8, width <= 64).

Nothing here is trained or clever about the network; ``MLP`` is a plain
container. The *forward* pass is ordinary float64 and is NOT certified -- it
is the object under analysis, not part of the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .errors import EnclosureError, NonFiniteEnclosure
from .interval import Interval, _down, _freeze, _up, affine

__all__ = ["MLP", "ibp_bounds", "crown_bounds", "jacobian_bounds"]

_ACTIVATIONS = ("relu", "tanh")


# --------------------------------------------------------------------------- #
# The network container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MLP:
    """A plain MLP: affine layers with an elementwise activation between them.

    ``weights`` is a tuple of ``(W, b)`` pairs, ``W`` of shape (out, in).
    The final affine layer has no activation.
    """

    weights: tuple
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_ACTIVATIONS}; got {self.activation!r}"
            )
        if len(self.weights) < 1:
            raise ValueError("MLP needs at least one affine layer")
        frozen = []
        prev = None
        for k, (W, b) in enumerate(self.weights):
            W = np.array(W, dtype=np.float64, copy=True)
            b = np.array(b, dtype=np.float64, copy=True)
            if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
                raise ValueError(f"layer {k}: W must be (out, in), b (out,)")
            if prev is not None and W.shape[1] != prev:
                raise ValueError(
                    f"layer {k}: expected input width {prev}, got {W.shape[1]}"
                )
            if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b))):
                raise NonFiniteEnclosure(f"layer {k}: non-finite parameters")
            W = _freeze(W)
            b = _freeze(b)
            frozen.append((W, b))
            prev = W.shape[0]
        object.__setattr__(self, "weights", tuple(frozen))

    @property
    def n_inputs(self) -> int:
        return self.weights[0][0].shape[1]

    @property
    def n_outputs(self) -> int:
        return self.weights[-1][0].shape[0]

    @property
    def n_hidden_layers(self) -> int:
        return len(self.weights) - 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Ordinary float64 forward pass. NOT certified; this is the analysand."""
        z = np.asarray(x, dtype=np.float64)
        for k, (W, b) in enumerate(self.weights):
            z = z @ W.T + b
            if k < len(self.weights) - 1:
                z = np.maximum(z, 0.0) if self.activation == "relu" else np.tanh(z)
        return z

    @classmethod
    def random(
        cls,
        sizes: tuple[int, ...],
        activation: str = "relu",
        rng: np.random.Generator | None = None,
        scale: float = 1.0,
    ) -> "MLP":
        rng = rng or np.random.default_rng()
        ws = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(size=(fan_out, fan_in)) * scale * np.sqrt(2.0 / fan_in)
            b = rng.normal(size=fan_out) * 0.1
            ws.append((W, b))
        return cls(tuple(ws), activation=activation)


def _check_tanh_flag(net: MLP, experimental: bool) -> None:
    if net.activation == "tanh" and not experimental:
        raise ValueError(
            "tanh bounds are experimental in v1 (parallel-slope relaxation, "
            "looser than tangent-optimal). Pass experimental=True to "
            "acknowledge and proceed."
        )


# --------------------------------------------------------------------------- #
# IBP
# --------------------------------------------------------------------------- #


def _act_interval(activation: str, zhat: Interval) -> Interval:
    if activation == "relu":
        return zhat.maximum(0.0)  # exact: monotone endpoint evaluation
    return zhat.tanh()


def _ibp_forward(net: MLP, box: Interval) -> tuple[Interval, list[Interval]]:
    if box.lo.ndim != 1 or box.lo.shape[0] != net.n_inputs:
        raise EnclosureError(
            f"input box has shape {box.lo.shape}; net expects ({net.n_inputs},)"
        )
    z = box
    preacts: list[Interval] = []
    out: Interval | None = None
    for k, (W, b) in enumerate(net.weights):
        zhat = affine(W, z, b)
        if k < len(net.weights) - 1:
            preacts.append(zhat)
            z = _act_interval(net.activation, zhat)
        else:
            out = zhat
    assert out is not None
    return out, preacts


def ibp_bounds(net: MLP, box: Interval, *, experimental: bool = False) -> Interval:
    """Interval bound propagation. Cheap, sound, and the baseline CROWN beats."""
    _check_tanh_flag(net, experimental)
    return _ibp_forward(net, box)[0]


# --------------------------------------------------------------------------- #
# Interval-coefficient helpers for the backward pass
# --------------------------------------------------------------------------- #


def _ee_mul(alo, ahi, blo, bhi):
    """Elementwise interval multiply on raw (lo, hi) arrays, broadcasting."""
    c = np.stack(
        np.broadcast_arrays(alo * blo, alo * bhi, ahi * blo, ahi * bhi)
    )
    return _down(c.min(axis=0)), _up(c.max(axis=0))


def _ee_add(alo, ahi, blo, bhi):
    return _down(alo + blo), _up(ahi + bhi)


def _split_sign(lo, hi):
    """Exact split of an interval coefficient into its positive/negative parts."""
    return (
        (np.maximum(lo, 0.0), np.maximum(hi, 0.0)),
        (np.minimum(lo, 0.0), np.minimum(hi, 0.0)),
    )


def _rowsum(lo, hi):
    """Directed sum over the last axis."""
    acc_lo = np.zeros(lo.shape[:-1])
    acc_hi = np.zeros(hi.shape[:-1])
    for j in range(lo.shape[-1]):
        acc_lo = _down(acc_lo + lo[..., j])
        acc_hi = _up(acc_hi + hi[..., j])
    return acc_lo, acc_hi


def _imatmul_exact(Llo, Lhi, W):
    """(O, J) interval coefficients times (J, M) exact matrix -> (O, M)."""
    O, J = Llo.shape
    M = W.shape[1]
    acc_lo = np.zeros((O, M))
    acc_hi = np.zeros((O, M))
    for j in range(J):
        t1 = Llo[:, j : j + 1] * W[j]
        t2 = Lhi[:, j : j + 1] * W[j]
        acc_lo = _down(acc_lo + _down(np.minimum(t1, t2)))
        acc_hi = _up(acc_hi + _up(np.maximum(t1, t2)))
    return acc_lo, acc_hi


def _idot_exactvec(Llo, Lhi, v):
    """(O, J) interval coefficients dotted with an exact (J,) vector -> (O,)."""
    acc_lo = np.zeros(Llo.shape[0])
    acc_hi = np.zeros(Lhi.shape[0])
    for j in range(Llo.shape[1]):
        t1 = Llo[:, j] * v[j]
        t2 = Lhi[:, j] * v[j]
        acc_lo = _down(acc_lo + _down(np.minimum(t1, t2)))
        acc_hi = _up(acc_hi + _up(np.maximum(t1, t2)))
    return acc_lo, acc_hi


def _concretize(Llo, Lhi, mu_lo, mu_hi, box: Interval):
    """Bound Lambda @ x + mu over the input box. Returns (val_lo, val_hi)."""
    acc_lo = np.array(mu_lo, copy=True)
    acc_hi = np.array(mu_hi, copy=True)
    for j in range(Llo.shape[1]):
        c = np.stack(
            [
                Llo[:, j] * box.lo[j],
                Llo[:, j] * box.hi[j],
                Lhi[:, j] * box.lo[j],
                Lhi[:, j] * box.hi[j],
            ]
        )
        acc_lo = _down(acc_lo + _down(c.min(axis=0)))
        acc_hi = _up(acc_hi + _up(c.max(axis=0)))
    return acc_lo, acc_hi


# --------------------------------------------------------------------------- #
# Activation relaxations (interval-valued coefficients)
# --------------------------------------------------------------------------- #
# Each relaxation is a dict of raw (lo, hi) arrays enclosing the exact real
# coefficients of a linear bound valid on the certified pre-activation range:
#   upper:  sigma(z) <= au * z + cu       lower:  sigma(z) >= al * z + cl


def _relax_relu(pre: Interval) -> dict[str, np.ndarray]:
    l, u = pre.lo, pre.hi
    H = l.shape[0]
    pos = l >= 0.0
    neg = u <= 0.0
    uns = ~(pos | neg)

    au_lo = np.where(pos, 1.0, 0.0)
    au_hi = au_lo.copy()
    cu_lo = np.zeros(H)
    cu_hi = np.zeros(H)
    # adaptive lower slope: identity when the box leans positive
    al_lo = np.where(pos, 1.0, np.where(neg, 0.0, np.where(u >= -l, 1.0, 0.0)))
    al_hi = al_lo.copy()
    cl_lo = np.zeros(H)
    cl_hi = np.zeros(H)

    if np.any(uns):
        U = Interval.point(u[uns])
        L = Interval.point(l[uns])
        D = U - L  # strictly positive: unstable means l < 0 < u
        A = U / D  # chord slope u / (u - l)
        C = (-(U * L)) / D  # chord intercept -u*l / (u - l), >= 0
        au_lo[uns], au_hi[uns] = A.lo, A.hi
        cu_lo[uns], cu_hi[uns] = C.lo, C.hi

    return dict(
        au_lo=au_lo, au_hi=au_hi, cu_lo=cu_lo, cu_hi=cu_hi,
        al_lo=al_lo, al_hi=al_hi, cl_lo=cl_lo, cl_hi=cl_hi,
    )


def _relax_tanh(pre: Interval) -> dict[str, np.ndarray]:
    """Parallel-slope relaxation for the S-shaped tanh.

    With lam = min(tanh'(l), tanh'(u)) -- the minimum of tanh' over [l, u],
    since tanh' is unimodal with its maximum at 0 -- the function
    g(z) = tanh(z) - lam*z is nondecreasing on [l, u], so

        lam*z + (tanh(l) - lam*l)  <=  tanh(z)  <=  lam*z + (tanh(u) - lam*u).

    Sound, simple, and every quantity is computable with M0 primitives; the
    tangent-optimal CROWN relaxation is tighter and is deferred (spec 11).
    """
    L = Interval.point(pre.lo)
    U = Interval.point(pre.hi)
    Tl = L.tanh()
    Tu = U.tanh()
    lam = (Interval.point(1.0) - Tl.sqr()).minimum(Interval.point(1.0) - Tu.sqr())
    lam = Interval(np.maximum(lam.lo, 0.0), np.minimum(lam.hi, 1.0))  # exact clamp
    Cu = Tu - lam * U
    Cl = Tl - lam * L
    return dict(
        au_lo=lam.lo, au_hi=lam.hi, cu_lo=Cu.lo, cu_hi=Cu.hi,
        al_lo=lam.lo, al_hi=lam.hi, cl_lo=Cl.lo, cl_hi=Cl.hi,
    )


def _build_relax(activation: str, pre: Interval) -> dict[str, np.ndarray]:
    return _relax_relu(pre) if activation == "relu" else _relax_tanh(pre)


# --------------------------------------------------------------------------- #
# Backward CROWN
# --------------------------------------------------------------------------- #


def _backward(
    net: MLP,
    relaxes: list[dict[str, np.ndarray]],
    W0: np.ndarray,
    b0: np.ndarray,
) -> tuple[tuple, tuple]:
    """Propagate the linear quantity W0 @ z_k + b0 back to the input.

    ``relaxes`` holds the relaxations for hidden layers 0..k-1 (so k =
    len(relaxes)). Returns two chains of interval coefficients over the input:
    (upper_Lambda_lo, upper_Lambda_hi, upper_mu_lo, upper_mu_hi) and the
    lower-chain counterpart.
    """
    up = dict(Llo=np.array(W0), Lhi=np.array(W0), mlo=np.array(b0), mhi=np.array(b0))
    lo = dict(Llo=np.array(W0), Lhi=np.array(W0), mlo=np.array(b0), mhi=np.array(b0))

    for k in range(len(relaxes) - 1, -1, -1):
        r = relaxes[k]
        Wk, bk = net.weights[k]
        for chain, own, other in ((up, "u", "l"), (lo, "l", "u")):
            # through the activation: lambda+ takes this chain's own-side
            # relaxation, lambda- takes the opposite side
            (plo, phi), (nlo, nhi) = _split_sign(chain["Llo"], chain["Lhi"])
            a_own = (r[f"a{own}_lo"][None, :], r[f"a{own}_hi"][None, :])
            a_oth = (r[f"a{other}_lo"][None, :], r[f"a{other}_hi"][None, :])
            c_own = (r[f"c{own}_lo"][None, :], r[f"c{own}_hi"][None, :])
            c_oth = (r[f"c{other}_lo"][None, :], r[f"c{other}_hi"][None, :])

            s1 = _ee_mul(plo, phi, *a_own)
            s2 = _ee_mul(nlo, nhi, *a_oth)
            Zlo, Zhi = _ee_add(*s1, *s2)  # coefficients on the pre-activation

            i1 = _ee_mul(plo, phi, *c_own)
            i2 = _ee_mul(nlo, nhi, *c_oth)
            ic_lo, ic_hi = _ee_add(*i1, *i2)
            add_lo, add_hi = _rowsum(ic_lo, ic_hi)
            mlo = _down(chain["mlo"] + add_lo)
            mhi = _up(chain["mhi"] + add_hi)

            # through the affine layer
            Llo, Lhi = _imatmul_exact(Zlo, Zhi, Wk)
            blo, bhi = _idot_exactvec(Zlo, Zhi, bk)
            chain["Llo"], chain["Lhi"] = Llo, Lhi
            chain["mlo"], chain["mhi"] = _down(mlo + blo), _up(mhi + bhi)

    return (
        (up["Llo"], up["Lhi"], up["mlo"], up["mhi"]),
        (lo["Llo"], lo["Lhi"], lo["mlo"], lo["mhi"]),
    )


def _intersect(a: Interval, b: Interval) -> Interval:
    """Both arguments are sound enclosures of the same set, so this is too."""
    return Interval(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def _crown_machinery(
    net: MLP, box: Interval
) -> tuple[Interval, Interval, Interval, list[Interval]]:
    """Returns (final, pure_crown_out, ibp_out, certified_preacts)."""
    with np.errstate(over="ignore", invalid="ignore"):
        ibp_out, ibp_pre = _ibp_forward(net, box)
        relaxes: list[dict[str, np.ndarray]] = []
        preacts: list[Interval] = []
        for k in range(net.n_hidden_layers):
            Wk, bk = net.weights[k]
            if k == 0:
                pk = affine(Wk, box, bk)  # first layer: exact affine image box
            else:
                (uL, uH, um_lo, um_hi), (lL, lH, lm_lo, lm_hi) = _backward(
                    net, relaxes, Wk, bk
                )
                _, hi = _concretize(uL, uH, um_lo, um_hi, box)
                lo, _ = _concretize(lL, lH, lm_lo, lm_hi, box)
                pk = _intersect(Interval(lo, hi), ibp_pre[k])
            preacts.append(pk)
            relaxes.append(_build_relax(net.activation, pk))

        WL, bL = net.weights[-1]
        (uL, uH, um_lo, um_hi), (lL, lH, lm_lo, lm_hi) = _backward(
            net, relaxes, WL, bL
        )
        _, out_hi = _concretize(uL, uH, um_lo, um_hi, box)
        out_lo, _ = _concretize(lL, lH, lm_lo, lm_hi, box)
        pure = Interval(out_lo, out_hi)
        return _intersect(pure, ibp_out), pure, ibp_out, preacts


def crown_bounds(
    net: MLP,
    box: Interval,
    *,
    experimental: bool = False,
    return_details: bool = False,
):
    """Certified output bounds: backward CROWN intersected with IBP.

    The result is never looser than IBP (the intersection guarantees it) and
    is typically much tighter whenever unstable neurons exist. With
    ``return_details=True`` also returns a dict with the pure CROWN and IBP
    enclosures and the certified per-layer pre-activation bounds.
    """
    _check_tanh_flag(net, experimental)
    final, pure, ibp, preacts = _crown_machinery(net, box)
    if return_details:
        return final, {"crown": pure, "ibp": ibp, "preact": preacts}
    return final


# --------------------------------------------------------------------------- #
# Interval Jacobians
# --------------------------------------------------------------------------- #


def jacobian_bounds(
    net: MLP, box: Interval, *, experimental: bool = False
) -> Interval:
    """Sound enclosure of the network Jacobian over the box, shape (out, in).

    ReLU uses {0}, {1}, or [0, 1] per neuron from certified pre-activation
    signs (with [0, 1] at the kink, covering the Clarke subdifferential);
    tanh uses 1 - tanh(z)^2 evaluated in interval arithmetic and clamped to
    [0, 1]. The chain product runs right-to-left with directed rounding.
    """
    _check_tanh_flag(net, experimental)
    _, _, _, preacts = _crown_machinery(net, box)

    with np.errstate(over="ignore", invalid="ignore"):
        WL, _ = net.weights[-1]
        Jlo = np.array(WL, copy=True)
        Jhi = np.array(WL, copy=True)
        for k in range(net.n_hidden_layers - 1, -1, -1):
            pk = preacts[k]
            if net.activation == "relu":
                dlo = np.where(pk.lo > 0.0, 1.0, 0.0)
                dhi = np.where(pk.hi < 0.0, 0.0, 1.0)
            else:
                D = Interval.point(1.0) - pk.tanh().sqr()
                dlo = np.maximum(D.lo, 0.0)
                dhi = np.minimum(D.hi, 1.0)
            Jlo, Jhi = _ee_mul(Jlo, Jhi, dlo[None, :], dhi[None, :])
            Wk, _ = net.weights[k]
            Jlo, Jhi = _imatmul_exact(Jlo, Jhi, Wk)
        return Interval(Jlo, Jhi)


# --------------------------------------------------------------------------- #
# Convenience training (NOT certified -- produces the analysand)
# --------------------------------------------------------------------------- #


def fit_mlp(
    sizes: tuple[int, ...],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    activation: str = "relu",
    steps: int = 3000,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> MLP:
    """Minimal Adam/MSE trainer in numpy. Deliberately plain: the trained
    network is the object under certification, and nothing about training
    enters any guarantee."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    Ws = []
    for fi, fo in zip(sizes[:-1], sizes[1:]):
        Ws.append(
            [rng.normal(size=(fo, fi)) * np.sqrt(2.0 / fi), np.zeros(fo)]
        )
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in Ws]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in Ws]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def act(z):
        return np.maximum(z, 0.0) if activation == "relu" else np.tanh(z)

    def dact(z):
        return (z > 0).astype(np.float64) if activation == "relu" else 1 - np.tanh(z) ** 2

    for t in range(1, steps + 1):
        idx = rng.integers(0, X.shape[0], size=batch)
        zs, a = [], X[idx]
        acts = [a]
        for k, (W, bb) in enumerate(Ws):
            z = a @ W.T + bb
            zs.append(z)
            a = act(z) if k < len(Ws) - 1 else z
            acts.append(a)
        g = 2.0 * (a - Y[idx]) / batch
        for k in range(len(Ws) - 1, -1, -1):
            if k < len(Ws) - 1:
                g = g * dact(zs[k])
            gW = g.T @ acts[k]
            gb = g.sum(axis=0)
            g = g @ Ws[k][0]
            for slot, grad in ((0, gW), (1, gb)):
                m[k][slot] = b1 * m[k][slot] + (1 - b1) * grad
                v[k][slot] = b2 * v[k][slot] + (1 - b2) * grad * grad
                mh = m[k][slot] / (1 - b1**t)
                vh = v[k][slot] / (1 - b2**t)
                Ws[k][slot] = Ws[k][slot] - lr * mh / (np.sqrt(vh) + eps)
    return MLP(tuple((W, bb) for W, bb in Ws), activation=activation)
