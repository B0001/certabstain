"""The first fully certified composition (M3 acceptance, final item).

Phase 1's demo assumed a certified model error epsilon and showed what the
two-sided monitor buys once you have one. This script closes the loop: the
same monitor, the same witness, the same conformal calibration -- but epsilon
comes out of the branch-and-bound certifier, bound to the exact network hash
and reference identity, with its cover and budget on the record.

Chain, end to end:

  reference   CircleClearance -- analytic, interval-twinned (M2)
  analysand   a small ReLU net trained to imitate it (uncertified, on purpose)
  epsilon     certify_epsilon over the declared domain (M3): produced
  witness     CertifiedModelErrorWitness(epsilon)  -- silence is sound (P1)
  false-alarm conformal calibration on nominal rollouts only (P1)
  claim       P(miss | violation) = 0 on the cover; P(false alarm) <= alpha

One precondition is stated rather than enforced here: every evaluated state
must lie inside the certificate's cover (here the cover is the full domain).
M5 moves that check into the gate, where it stops being a sentence in a demo
and becomes an invariant.
"""

from __future__ import annotations

import numpy as np

from certabstain import (
    CertifiedModelErrorWitness,
    Interval,
    TargetNotCertified,
    build_monitor,
    certify_epsilon,
)
from certabstain.nnbound import fit_mlp
from certabstain.reference import CircleClearance

RNG = np.random.default_rng(3)

DOMAIN = Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
CLEAR = CircleClearance(ox=0.0, oy=0.0, r=0.15)
ALPHA = 0.05
HORIZON = 25
N_CAL = 999


def rule(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


# ------------------------------------------------------------------ 1
rule("1. Train the analysand (uncertified, deliberately imperfect)")
X = RNG.uniform(DOMAIN.lo, DOMAIN.hi, size=(40_000, 2))
net = fit_mlp((2, 32, 32, 1), X, CLEAR.value(X), steps=4000, seed=0)
print(f"trained (2, 32, 32, 1) ReLU net on {X.shape[0]} samples of h(x, y)")

# ------------------------------------------------------------------ 2
rule("2. Produce epsilon (the line Phase 1 had to assume)")
cert = certify_epsilon(
    net,
    CLEAR.interval_batch,
    DOMAIN,
    reference_id=CLEAR.reference_id(),
    ref_float=CLEAR.value,
    target=None,
    max_leaf_evals=400_000,
    floor_samples=2_000_000,
)
eps = float(cert.eps[0])
print(cert.summary())
print(f"\nepsilon = {eps:.4g}  (empirical floor {float(cert.empirical_floor[0]):.4g}, "
      f"ratio {eps / float(cert.empirical_floor[0]):.2f}x)")

# ------------------------------------------------------------------ 3
rule("3. Compose: certified witness + conformal false-alarm side")
witness = CertifiedModelErrorWitness(epsilon=eps)


def nominal_traj() -> np.ndarray:
    """A rollout that keeps real clearance: a bounded random walk in the
    annulus h >= 0.12, scored through the LEARNED clearance."""
    pts = np.empty((HORIZON, 2))
    p = RNG.uniform(-0.45, 0.45, size=2)
    while CLEAR.value(p[None, :])[0] < 0.12:
        p = RNG.uniform(-0.45, 0.45, size=2)
    for t in range(HORIZON):
        step = RNG.normal(scale=0.02, size=2)
        q = np.clip(p + step, DOMAIN.lo + 1e-6, DOMAIN.hi - 1e-6)
        if CLEAR.value(q[None, :])[0] >= 0.12:
            p = q
        pts[t] = p
    return pts


cal = [witness.score(net.forward(nominal_traj())[:, 0]) for _ in range(N_CAL)]
monitor = build_monitor(
    nominal_trajectories=cal,
    alpha=ALPHA,
    witness=witness,
    safe_action="HALT",
)
print(monitor.describe())

# ------------------------------------------------------------------ 4
rule("4. Evaluate the composed claim")
n_eval = 4000
fa = sum(
    bool(np.any(witness.score(net.forward(nominal_traj())[:, 0]) > monitor.threshold))
    for _ in range(n_eval)
)
print(f"false alarms on nominal rollouts   {fa / n_eval:6.2%}   bound {ALPHA:.0%}")

viol = 0
missed = 0
while viol < 20_000:
    p = RNG.uniform(-0.15, 0.15, size=(4096, 2))
    inside = CLEAR.value(p) < 0.0
    p = p[inside]
    if p.shape[0] == 0:
        continue
    s = witness.score(net.forward(p)[:, 0])
    missed += int(np.count_nonzero(s <= monitor.threshold))
    viol += p.shape[0]
print(f"missed violations ({viol} states)  {missed / viol:6.2%}   bound 0%  "
      f"(proven on the cover; miss bound {monitor.claim.miss_bound:g})")

# ------------------------------------------------------------------ 5
rule("5. The certificate refuses what it cannot support")
try:
    certify_epsilon(
        net,
        CLEAR.interval_batch,
        DOMAIN,
        reference_id=CLEAR.reference_id(),
        ref_float=CLEAR.value,
        target=float(cert.empirical_floor[0]) * 0.5,
        max_leaf_evals=60_000,
        floor_samples=500_000,
    )
    print("unexpectedly certified an impossible target")
except TargetNotCertified as e:
    print(f"target below the floor -> {type(e).__name__}")
    print("  " + str(e).splitlines()[0])

print(
    "\nPhase 1 assumed epsilon; this run produced it, hash-bound it to the "
    "network\nand reference, and the same two-sided claim now rests on a "
    "certificate\ninstead of a promise. M5 moves the cover check into the gate."
)
