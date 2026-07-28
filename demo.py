"""Demo: what the two-sided certificate buys, what it costs, and where it stops.

Scenario. A learned policy drives a system with safety specification
``g(x) >= 0``. We never observe ``g``; we observe a learned ``g_hat``.

Read this alongside the two claims, which are different in kind:

    miss bound        STRUCTURAL. Holds for any deployment distribution,
                      conditional on epsilon being a genuine bound over the
                      operating domain. Immune to distribution shift.

    false-alarm bound DISTRIBUTIONAL. Holds on the calibration distribution,
                      or within a declared shift budget. Degrades under
                      undeclared shift, exactly like every other conformal
                      monitor -- which is what ``shift_budget`` is for.

Conflating those two is how monitors end up over-promising. This demo keeps
them apart, including in the places where certabstain's guarantee runs out.
"""

from __future__ import annotations

import numpy as np

from certabstain import (
    ActionGate,
    CertifiedModelErrorWitness,
    CircleClearance,
    Interval,
    ShiftBudgetExceeded,
    SoundnessNotEstablished,
    SplitConformalCalibrator,
    VerifiedDiscrepancyWitness,
    build_monitor,
    certify_epsilon,
    rollout_scores,
)
from certabstain.nnbound import fit_mlp

RNG = np.random.default_rng(11)

ALPHA = 0.05
HORIZON = 25
N_CAL = 999
N_EVAL = 4000

ERR_CALIB = 0.02   # model error seen during calibration
ERR_SHIFT = 0.30   # model error out-of-distribution, still inside the certified bound
ERR_BREAK = 1.00   # model genuinely breaks down, outside the certified bound
EPS_CERT = 0.35    # certified uniform bound over the declared operating domain


def nominal_g(n: int) -> np.ndarray:
    return RNG.uniform(0.80, 2.00, n)


def violating_g(n: int) -> np.ndarray:
    g = RNG.uniform(0.80, 2.00, n)
    g[RNG.integers(0, n)] = RNG.uniform(-0.20, -0.001)
    return g


def observe(g: np.ndarray, err: float) -> np.ndarray:
    return g + RNG.uniform(-err, err, g.shape)


def rates(score_fn, threshold: float, err: float) -> tuple[float, float]:
    fa = sum(
        bool(np.any(score_fn(observe(nominal_g(HORIZON), err)) > threshold))
        for _ in range(N_EVAL)
    )
    miss = sum(
        not bool(np.any(score_fn(observe(violating_g(HORIZON), err)) > threshold))
        for _ in range(N_EVAL)
    )
    return fa / N_EVAL, miss / N_EVAL


def rule(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


REGIMES = (
    ("in-distribution     ", ERR_CALIB),
    ("shifted             ", ERR_SHIFT),
    ("model breakdown     ", ERR_BREAK),
)

# ------------------------------------------------------------------ run 1
rule("1. Baseline conformal monitor on the raw learned score")

raw = lambda g_hat: -g_hat  # noqa: E731
base = SplitConformalCalibrator.fit(
    rollout_scores([raw(observe(nominal_g(HORIZON), ERR_CALIB)) for _ in range(N_CAL)]),
    ALPHA,
)
print(f"threshold {base.threshold:+.4f}")
print(f"implicitly assumes model error stays under {abs(base.threshold):.2f} -- never stated anywhere\n")
print(f"{'regime':<22}{'model err':>10}{'false alarms':>14}{'missed violations':>20}")
for label, err in REGIMES:
    fa, miss = rates(raw, base.threshold, err)
    print(f"{label:<22}{err:>10.2f}{fa:>13.1%}{miss:>19.1%}")
print(f"\nbound advertised:              {ALPHA:>12.0%}{'none':>19}")
print("Both failures are silent. The monitor reports nothing unusual in any row.")

# ------------------------------------------------------------------ run 2
rule("2. certabstain: certified error folded into the score")

w = CertifiedModelErrorWitness(epsilon=EPS_CERT)
mon = build_monitor(
    nominal_trajectories=[
        w.score(observe(nominal_g(HORIZON), ERR_CALIB)) for _ in range(N_CAL)
    ],
    alpha=ALPHA,
    witness=w,
    safe_action="HALT",
)
print(mon.describe())
print(f"\n{'regime':<22}{'model err':>10}{'false alarms':>14}{'missed violations':>20}")
for label, err in REGIMES:
    fa, miss = rates(w.score, mon.threshold, err)
    note = "  <-- epsilon violated" if err > EPS_CERT else ""
    print(f"{label:<22}{err:>10.2f}{fa:>13.1%}{miss:>19.1%}{note}")

print(
    f"\nRows 1-2 (model error within the certified epsilon={EPS_CERT}): zero misses, as proven.\n"
    "\n"
    "Row 3 is the honest limit, and it is worth staring at. The certified bound\n"
    "is violated there, so the miss guarantee is void -- yet the measured miss\n"
    "rate is still 0.0%. That is the trap this whole library exists to close:\n"
    "an empirical rate that looks fine is not a bound that holds. The number\n"
    "would not have warned you. Only the assumption on epsilon would have, which\n"
    "is why epsilon must come from a proof over the declared operating domain\n"
    "rather than from a measurement on nominal data.\n"
    "\n"
    "Note also that false alarms rise with shift in every row, for certabstain\n"
    "as much as for the baseline. That bound is distributional; carrying it\n"
    "under shift means declaring a budget, and section 4 shows what happens\n"
    "when the budget you need exceeds the one you can afford."
)

# ------------------------------------------------------------------ run 3
rule("3. The gate: no actuation without a fresh, bound certificate")

for label, g_true in (("clear ", 1.4), ("breach", -0.05)):
    d = mon.step(
        observation=np.array([g_true]),
        proposed_action=np.array([0.4, -0.2]),
        score=float(w.score(g_true + RNG.uniform(-ERR_SHIFT, ERR_SHIFT))),
    )
    tag = "ABSTAIN -> HALT" if d.abstained else "emit           "
    print(f"{label}  {tag}  {d.reason}")

# ------------------------------------------------------------------ run 4
rule("4. Refusals: decline at build time rather than void at runtime")

base_args = dict(
    nominal_trajectories=[
        w.score(observe(nominal_g(HORIZON), ERR_CALIB)) for _ in range(N_CAL)
    ],
    alpha=ALPHA,
    witness=w,
    safe_action="HALT",
)
for desc, override in (
    ("shift budget >= alpha", dict(shift_budget=0.08)),
    (
        "nominal skims the constraint",
        dict(nominal_trajectories=[list(RNG.uniform(-0.2, 0.4, HORIZON)) for _ in range(400)]),
    ),
    ("alpha too small for n", dict(alpha=0.0005)),
):
    args = dict(base_args)
    args.update(override)
    try:
        build_monitor(**args)
        print(f"{desc}: unexpectedly succeeded")
    except Exception as e:  # noqa: BLE001
        print(f"{desc}\n  -> {type(e).__name__}: {str(e).splitlines()[0]}\n")

# ------------------------------------------------------------------ run 5
rule("5. The cost: certified bound vs. required margin")

print("A two-sided claim needs nominal clearance above 2*epsilon at the")
print("(1-alpha) quantile. A looser certified bound prices you out entirely.\n")
print(f"{'epsilon':>9}{'needs margin >':>17}{'nominal in [0.80, 2.00]':>28}")
for eps in (0.05, 0.15, 0.30, 0.35, 0.42, 0.50, 1.00):
    wi = CertifiedModelErrorWitness(epsilon=eps)
    trajs = [
        wi.score(observe(nominal_g(HORIZON), min(eps, ERR_CALIB))) for _ in range(N_CAL)
    ]
    try:
        m = build_monitor(
            nominal_trajectories=trajs, alpha=ALPHA, witness=wi, safe_action="HALT"
        )
        verdict = f"certified, thr {m.threshold:+.3f}"
    except SoundnessNotEstablished:
        verdict = "REFUSED"
    print(f"{eps:9.2f}{2 * eps:17.2f}{verdict:>28}")

print(
    "\nNote the last row. Covering the model-breakdown regime honestly (epsilon=1.0)\n"
    "would demand 2.0 of clearance, which this system does not have -- so the\n"
    "library refuses to certify rather than issuing the claim that row 3 of\n"
    "section 2 would have quietly broken.\n"
    "\nThat is the whole design: every unit shaved off the certified bound is\n"
    "returned directly as operating envelope, and when the bound is too loose to\n"
    "support a claim, you find out at build time instead of in an incident report."
)

# ------------------------------------------------------------------ run 6
rule("6. Phase 2 (M3-M5): the silent-void row, closed")

print(
    "Section 2's row 3 (model breakdown) was Phase 1's one honest weakness:\n"
    "epsilon=0.35 was ASSUMED, the true error (1.00) blew past it, and the\n"
    "measured miss rate still read 0.0% -- nothing on that row would have\n"
    "warned you. Phase 2 makes epsilon a PRODUCED, certified bound over a\n"
    "declared domain, and wires cover-membership into the gate: a state the\n"
    "certificate was never proven over now abstains by name, instead of\n"
    "silently returning a clean-looking number.\n"
)

CLEAR = CircleClearance(ox=0.0, oy=0.0, r=0.15)
_domain = Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
_X = RNG.uniform(_domain.lo, _domain.hi, size=(20_000, 2))
_net = fit_mlp((2, 16, 16, 1), _X, CLEAR.value(_X), steps=2000, seed=0)
_cert = certify_epsilon(
    _net,
    CLEAR.interval_batch,
    _domain,
    reference_id=CLEAR.reference_id(),
    ref_float=CLEAR.value,
    target=None,
    max_leaf_evals=80_000,
    floor_samples=200_000,
)
w2 = VerifiedDiscrepancyWitness.bind(_cert, _net, CLEAR.reference_id())
print(f"produced (not assumed): eps={float(np.max(_cert.eps)):.4g}  "
      f"cover={_cert.cover_fraction:.1%} of the declared domain\n")

gate2 = ActionGate(
    threshold=0.0, false_alarm_bound=ALPHA, safe_action="HALT", cover=w2.covers
)
for label, point in (
    ("in-distribution     ", np.array([0.3, 0.3])),
    ("near the domain edge ", np.array([0.45, 0.45])),
    ("model breakdown      ", np.array([10.0, 10.0])),  # driven off the domain entirely
):
    g_hat = float(CLEAR.value(point[None, :])[0])
    d = gate2.step(
        observation=point, proposed_action=np.array([0.0]), score=w2.score(g_hat)
    )
    tag = "ABSTAIN -> HALT" if d.abstained else "emit           "
    print(f"{label}  {tag}  {d.reason}")

print(
    "\nThe third row is the same scenario Section 2's row 3 called the trap:\n"
    "a state so far outside anything ever analyzed that the old library had\n"
    "nothing to say about it but a score. Now it reads \"left certified\n"
    "domain\" -- a detected abstention, not a silent void."
)
