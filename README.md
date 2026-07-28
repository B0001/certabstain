# certabstain

Certified runtime abstention for learned robot policies.

Every published runtime monitor for VLA and generative manipulation policies
controls false alarms and says nothing about misses. Their authors are explicit
about why: a miss bound is treated as requiring failure rollouts to calibrate
against, and failure rollouts are exactly what you do not have.

This library gets a two-sided guarantee anyway, by refusing to treat the miss
side as a statistical problem at all.

```
false-alarm side   statistical   conformal calibration on nominal rollouts only
                                 distribution-free, finite-sample, shift-inflatable

miss side          structural    soundness property of the score itself
                                 deterministic, no failure data, immune to shift

enforcement        architectural unforgeable single-use certificates gate actuation
                                 no override path exists in the API
```

Neither side alone is a two-sided guarantee. Composed, they are.

## The idea in one paragraph

Given a learned safety function `g_hat` with a **certified** uniform error
bound `|g_hat - g| <= epsilon` over the operating domain, define the monitor
score as `s(x) = epsilon - g_hat(x)`. Then `s(x) <= 0` implies
`g_hat(x) >= epsilon` implies `g(x) >= 0`: **silence is sound**, the monitor
cannot stay quiet through a real violation. Conversely any violation `g(x) < 0`
forces `s(x) > 0`. So every violating state scores above a known floor, and a
calibrated threshold below that floor has a miss rate of exactly zero — proven,
not measured, and with no failure data anywhere in the pipeline. Conformal
calibration is then used only for what it is actually good at: bounding false
alarms on the nominal distribution.

The composition is checked at construction. If the calibrated threshold does
not clear the floor, `TwoSidedClaim.compose` raises rather than returning a
certificate that asserts a bound it cannot support.

## Install and run

```bash
cd certabstain
pip install numpy scipy pytest
python -m pytest tests/ -q          # 29 tests
PYTHONPATH=. python demo/demo.py
```

## Usage

```python
from certabstain import build_monitor, CertifiedModelErrorWitness

monitor = build_monitor(
    nominal_trajectories=calib_rollouts,  # per-timestep scores, SUCCESSES ONLY
    alpha=0.01,                           # false-alarm budget per rollout
    witness=CertifiedModelErrorWitness(epsilon=0.05),
    shift_budget=0.002,                   # total-variation ball radius
    safe_action=STOP,
)
print(monitor.describe())

decision = monitor.step(observation=obs, proposed_action=a, score=s)
if decision.abstained:
    handoff(decision.reason)
```

## What is guaranteed

| Claim | Kind | Holds under shift? | Needs failure data? |
|---|---|---|---|
| `P(false alarm \| nominal rollout) <= alpha` | statistical | only within declared budget | no |
| `P(miss \| violation) = 0` | structural | **yes**, unconditionally | no |
| certificate unforgeable / single-use / epoch- and action-bound | architectural | n/a | no |

The false-alarm bound is **per rollout**, not per timestep. Calibrating on the
rollout max gives an exact whole-trajectory guarantee with no union bound; a
per-timestep bound at `alpha` admits up to `T * alpha` false alarms over a
horizon of `T`, which is usually not what anyone means.

`MondrianCalibrator` upgrades marginal coverage to **group-conditional**
coverage — `P(no false alarm | stratum g) >= 1 - alpha` for every stratum. This
is the claim that matters operationally; "safe on average across tasks" is not
a useful thing to tell whoever owns the robot.

## What is *not* guaranteed

Stated plainly, because the failure mode this library targets is exactly the
habit of not stating these:

- **The miss bound is conditional on `epsilon` being a genuine bound over the
  declared operating domain.** If `epsilon` is estimated from nominal data
  rather than proven, the guarantee is worthless. Section 2 of the demo shows a
  regime where the bound is violated, the guarantee is void, and the *measured*
  miss rate is still 0.0% — the number does not warn you.
- **The false-alarm bound degrades under undeclared distribution shift**, for
  this library as much as for any other conformal monitor. Declare a
  `shift_budget` or accept that the bound is calibration-distribution-only.
- Under a total-variation budget `rho`, the claim requires `rho < alpha`.
  Beyond that the library raises `ShiftBudgetExceeded` rather than issuing a
  threshold that no longer means what it says.
- A two-sided claim requires nominal clearance above roughly `2 * epsilon`.
  Systems that habitually skim the constraint boundary get
  `SoundnessNotEstablished` — correctly, because for such a system no monitor
  can separate nominal operation from violation.

## The invariant

The gate is not a function the control loop is supposed to call. It is the only
object that can produce an actuator command, and it will not produce one
without a certificate that verifies, is bound to the current epoch, binds the
exact observation and action, and has not been used before. The policy never
holds the signing key. There is no `force=`, `override=`, or `strict=False`
anywhere in the API, and a test asserts that the public surface stays that way.
Any exception in the path fails closed to the safe action.

## Novelty relative to prior art

Published monitors (SAFE, FIPER, FAIL-Detect, Sentinel, UNISafe) all sit on the
statistical side and each names one or more of these as open: instance-level
conditional coverage, shift-robust calibration, low-FPR operation, and a
false-negative bound without failure data. The general concept "robot detects
it is uncertain and asks for help" is prior art and partly patented (Dexterity,
US10824142B2 and continuations). Novelty here is not the concept — it is the
composition: **moving the miss side off statistics entirely and onto a
soundness property of the score, so the two-sided claim needs no failure data
and does not decay under shift**, plus the architectural non-bypassability of
the gate. File a provisional on the composition and the gate before publishing;
get a real freedom-to-operate search first.

## Layout

```
certabstain/
  conformal.py   split conformal, Mondrian group-conditional, shift inflation
  soundness.py   witnesses and the two-sided composition
  gate.py        certificate authority and the non-bypassable action gate
  errors.py      every failure mode is a distinct, catchable refusal
tests/           29 tests; coverage theorems validated by Monte Carlo
demo/            baseline vs. certified, including where the guarantee stops
```

## Next

1. Reproduce SAFE/FIPER/FAIL-Detect on the public SAFE rollout sets and drop
   this monitor in alongside them on the same data.
2. Replace `CertifiedModelErrorWitness` with a real certified bound on a
   learned contact/dynamics model — the reachable-tube route.
3. Mechanize the soundness argument in Lean so the composition itself is
   machine-checked, not just the code that implements it.
