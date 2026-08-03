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

Development happens in a project-local venv at `certabstain/.venv`, not the
shared workspace one directory up. From `~/Downloads/certabstain`:

```bash
# full suite -- 136 tests
VIRTUAL_ENV="$(pwd)/.venv" uv run --active --no-sync pytest -q

# bare scripts need PYTHONPATH so `certabstain` resolves as a package
# one directory up; pytest above does not
VIRTUAL_ENV="$(pwd)/.venv" PYTHONPATH="$(dirname "$(pwd)")" \
    uv run --active --no-sync python demo.py
```

Runtime dependency is numpy alone; `matplotlib`, `scipy`, `onnx`,
`onnxruntime`, and `mpmath` are dev-group extras used by the sweeps, the
Clopper-Pearson interval in `soundness.py`, and the VNNLIB export.

`TECHNICAL_NOTE.md` is the Phase 2 write-up: the L1-L4 lemma chain, the
stiffness-boundary result, the planar-pushing numbers, and which script
reproduces which artifact.

## Usage

```python
from certabstain import (
    ActionGate, VerifiedDiscrepancyWitness, build_monitor, certify_epsilon,
)

# 1. PRODUCE epsilon -- branch-and-bound the network against an
#    interval-extendable reference over a declared domain. Refuses if the
#    certified cover falls below the declared minimum fraction.
cert = certify_epsilon(
    net, clearance.interval_batch, domain,
    reference_id=clearance.reference_id(),
    ref_float=clearance.value,
    target=None, max_leaf_evals=80_000,
)

# 2. BIND it -- weight hash and reference identity re-checked here, so a
#    single flipped weight byte refuses at load rather than at runtime.
witness = VerifiedDiscrepancyWitness.bind(cert, net, clearance.reference_id())

# 3. Calibrate the false-alarm side on nominal rollouts only.
monitor = build_monitor(
    nominal_trajectories=calib_rollouts,  # per-timestep scores, SUCCESSES ONLY
    alpha=0.01,                           # false-alarm budget per rollout
    witness=witness,
    shift_budget=0.002,                   # total-variation ball radius
    safe_action=STOP,
)
print(monitor.describe())

decision = monitor.step(
    observation=obs, proposed_action=a, score=witness.score(g_hat)
)
if decision.abstained:
    handoff(decision.reason)
```

Cover-membership abstention means a state outside the certified domain reports
`left certified domain` instead of a score the certificate never covered.
`build_monitor` wires this for you from the witness — a witness whose
guarantee holds only inside a certified domain (W1/W2) supplies its own
`covers`, and one with no certified region (`CertifiedModelErrorWitness`) has
none, so nothing is invented. Building the gate directly, pass it yourself:

```python
gate = ActionGate(
    threshold=0.0, false_alarm_bound=0.01, safe_action=STOP, cover=witness.covers,
)
```

Until 2026-08-03 `build_monitor` dropped the predicate, so this check was only
ever active on the hand-built route above; see §6 of `TECHNICAL_NOTE.md`.

`CertifiedModelErrorWitness(epsilon=...)` still exists and takes `epsilon` on
faith. It is the Phase 1 interface, kept for the demo's cautionary section;
prefer the produced bound above.

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
holds the signing key: the gate mints its own authority and accepts none from
outside, and hands out only a `CertificateVerifier`, which checks a tag but
cannot mint one. There is no `force=`, `override=`, or `strict=False` anywhere
in the API; a non-finite `threshold` is refused at construction, because
`score > nan` is false and would certify everything; and an allowlist-based
signature scan pins every public member and parameter of every class the gate
module exports, plus `__slots__`, so neither a new parameter nor a public
instance attribute can appear unnoticed. Any exception in the path — including
one raised by the authority itself — fails closed to the safe action.

The honest scope: that is a statement about the **API surface**, not memory
isolation. No supported call sequence yields an uncertified emission or a
certificate the gate did not authorize. Code already running inside the same
Python process can always reach a private attribute; the deployment form of
that separation is an authority in its own process or an HSM.

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

Flat package; tests sit beside the modules they cover.

```
certabstain/
  conformal.py              split conformal, Mondrian group-conditional, shift inflation
  soundness.py              witnesses and the two-sided composition
  gate.py                   certificate authority and the non-bypassable action gate
  errors.py                 every failure mode is a distinct, catchable refusal

  interval.py               sound interval arithmetic under outward rounding      (L1)
  nnbound.py                IBP / CROWN bound propagation for the network
  reference.py              interval-extendable reference models + identity strings
  discrepancy.py            adaptive branch-and-bound certified epsilon           (L2)
  tube.py                   K-step interval tube + discrete-Gronwall comparison   (L3)
  witness2.py               predictive witness, cover-membership gating           (L4)
  certified_composition.py  the produced-epsilon two-sided claim
  vnnlib.py                 VNNLIB/ONNX export for out-of-band re-verification

  test_*.py                 136 tests: theorem validation (Monte Carlo +
                            adversarial corners), refusal coverage, freeze tests
  demo.py                   baseline vs. certified, including where the guarantee stops
  stiffness_sweep.py        M2 single-step enclosure vs. stiffness
  tube_sweep.py             M4 certified-horizon collapse
  scaling_study.py          M6 input-dimension and horizon scaling
  artifacts/                committed results; artifacts/vnnlib/ is the external cross-check
```

## Next

Phase 2 is done: `epsilon` is now produced by `certify_epsilon` and the
reachable-tube route is implemented (`tube.py`, `witness2.py`), certified
through planar pushing per-mode.

The out-of-band α,β-CROWN re-verification of `artifacts/vnnlib/`, listed here
as outstanding until 2026-08-03, **has been run: `unsat` on all 24 instances**
(α,β-CROWN 0.7.0, CPU, `double_fp: true`; verdicts in
`artifacts/abcrown_run_2026-08-03.json`). Run it in float64 — at the verifier's
float32 default, `net_15` returns a spurious `sat`, explained in
`artifacts/vnnlib/RUN.md`. This is one verifier on one machine on one run, so
it is a cross-check, not an independent audit.

What remains:

1. Reproduce SAFE/FIPER/FAIL-Detect on the public SAFE rollout sets and drop
   this monitor in alongside them on the same data.
2. Mechanize the soundness argument in Lean so the composition itself is
   machine-checked, not just the code that implements it.
3. Professional freedom-to-operate search before any filing (see
   `PROVISIONAL_OUTLINE.md`, which is an engineering draft for attorney
   review and not legal advice).
