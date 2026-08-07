# certabstain Phase 2 — Technical Note

**M7 write-up · verified epsilon, the lemma chain, the stiffness boundary, and
how to reproduce every number in it.**

This is an internal engineering note, not a legal document. Every quantitative
claim below traces to a JSON artifact or a test in this repository; none is
estimated or recalled. Section 5 gives exact commands to regenerate all of
them.

---

## 1. Overview

Phase 1 built a two-sided runtime-abstention monitor for learned robot
policies: a statistical false-alarm bound from conformal calibration on
nominal rollouts, and a structural miss bound of exactly zero, built into the
monitor score's own construction rather than measured from failure data.
`TwoSidedClaim.compose` refuses to exist unless the calibrated threshold
clears the score's proven violation floor — so the composition itself is
checked, not just each side separately. Phase 1's one honest gap was that the
miss-side guarantee was *conditional on an assumed model-error bound* ε; the
demo could show a violated ε producing a measured miss rate of 0.0% that
never warned anyone.

Phase 2, per SPEC.md section 1, removes the assumption: it builds the
machinery that **produces** ε for a learned contact-dynamics model as a
machine-checked artifact — a certified deterministic bound, not a fitted or
sampled one — and wires it into the existing witness and gate so the
composition is certified end to end, still with no failure data anywhere in
the pipeline. Everything below this line is the mechanics and the measured
behavior of that machinery.

---

## 2. The lemma chain (L1–L4)

Four lemmas, each carried by a specific function or class, each validated by
Monte Carlo containment tests in the test suite.

### L1 — Enclosure (`interval.py`)

*Invariant:* for the interval extension `IA[f]` of any primitive this module
provides, and any box `B`: for every real `z ∈ B`, `f(z) ∈ IA[f](B)`, in exact
arithmetic and under outward rounding.

`Interval` is the substrate everything else is built on — a closed box of
float64 endpoints, immutable after construction. `+ - * /` and `sqrt` get
their soundness for free from IEEE-754 correct rounding plus one directed
`nextafter` step per side; `exp`/`tanh` carry no such libm guarantee, so
they're treated as merely *faithful* (within 1 ulp) and widened by an extra
fixed number of `nextafter` steps — an assumption that is not trusted but
*checked*: `rounding_self_test` validates every primitive against brackets
computed offline at 700-digit precision and against exact `Fraction`
arithmetic, and `require_sound_environment` refuses to let any certification
proceed if that self-test fails. Every downstream lemma inherits its
soundness from this one: nothing above L1 does its own floating-point
reasoning, it only composes L1-sound primitives.

### L2 — One-step discrepancy bound (`discrepancy.py`)

*Invariant:* on a leaf box `B`, with `IA[ĝ_i](B)` the network's enclosure and
`IA[g_i](B)` the reference's enclosure,

```
sup_B (ĝ_i - g_i) ≤ hi(IA[ĝ_i](B)) - lo(IA[g_i](B))
sup_B (g_i - ĝ_i) ≤ hi(IA[g_i](B)) - lo(IA[ĝ_i](B))
u_{B,i} = max of the two
```

`_gap_bound` computes exactly this two-enclosure bound per box, with
`_batched_ibp` supplying the network side (batched IBP over the M0
substrate, optionally polished by CROWN on the worst surviving leaves —
CROWN-intersect-IBP is sound because both are sound and their elementwise
minimum can only tighten). `certify_epsilon` is the adaptive branch-and-bound
loop that drives this to a certificate: it refines boxes until every leaf's
`u_{B,i}` bound is below target or the leaf-eval budget is exhausted,
excludes boxes provably outside the declared mode, shrinks and drops
mode-straddling boxes rather than including them (spec §5 — certification
fails to a *smaller* domain, never a *weaker* claim), and refuses outright if
the resulting cover falls below a declared minimum fraction of the domain
(default 90%). The output, `EpsilonCertificate`, binds a BLAKE2b hash of the
exact network weight bytes and a reference-identity string (stiffness,
timestep, geometry) alongside `eps` and the exact cover-box tree — a
retrained network or a changed reference parameter voids the certificate
detectably (`matches_network`), not silently.

### L3 — The K-step tube (`tube.py`)

*Invariant:* if `X_t × U_t ⊆ Cover`, then `g(X_t, U_t) ⊆ IA[ĝ](X_t × U_t) ⊕
[-ε, ε]^n`. Iterating while the precondition holds yields a tube containing
every true trajectory from `X_0` under the declared open-loop control boxes.

`propagate_tube` is this loop. At each step it first checks
`cover_contains_box` — a sound, fail-closed area-based test that the joint
state-control box lies entirely inside the certificate's cover (a box whose
volume-inside-cover falls even fractionally short is treated as uncovered,
never rounded away) — and if the box has left the cover, the certified
horizon simply stops there (`cover_exit_reason` records why); it never
reports a horizon it can't back. When the precondition holds, it takes the
CROWN-bounded network image `y_hat` widened by `[-ε, ε]` (the certified
discrepancy from L2, Minkowski-summed on), **and** independently computes a
scalar discrete-Grönwall bound from certified Jacobian enclosures
(`nnbound.jacobian_bounds`) around a nominal, uncertified center rollout. Both
bounds are individually sound, so `propagate_tube` intersects them
(`combined = Interval(max(lo,lo), min(hi,hi))`) — sound-meet-sound, the same
composition pattern `nnbound.py` uses for CROWN-intersect-IBP — and takes
whichever is tighter at each step. An empty intersection here is treated as a
genuine contradiction (`Interval`'s constructor raises), not a case to smooth
over. `clearance_lower_bounds` then evaluates a certified interval clearance
function at every tube step, producing the `lo(IA[h](X_t))` sequence L4's
score is built from.

### L4 — Sound silence, predictive witness (`witness2.py`)

*Invariant:* `s = max over t ≤ K of (c_required - lo(IA[h](X_t)))`; `s ≤ 0`
implies true clearance `≥ c_required` for every `t ≤ K`.

`PredictiveTubeWitness.build` consumes a `TubeResult` and a batched interval
clearance function and stores exactly the `clearance_lo` sequence L3
produced; `.score()` computes `s`. `.violation_floor()` is `0.0` and
`.miss_probability(threshold)` is `0.0` for any non-negative threshold — the
same violation-floor-zero shape as Phase 1's `MarginWitness`, now with
lookahead over the whole certified horizon rather than a single step.
`.covers(point)` reports whether a point lies in any tube box up to the
*achieved* horizon (which may be short of the requested one — again, §5's
rule applied to a time axis) — this predicate is what M5 wires into
`ActionGate` as its cover-membership check, so that a state driven outside
everything the tube was ever proven over produces an abstention rather than
a score computed on a certificate that has already gone silent about that
region. `VerifiedDiscrepancyWitness` (W1, the direct one-step counterpart)
carries the same binding discipline for the non-predictive case: its `.bind`
classmethod is the only constructor, and it refuses at bind time — "at
load," in the spec's words — on a weight-hash or reference-identity
mismatch, rather than letting an unproven bound get trusted at some later
runtime moment.

---

## 3. The stiffness-boundary result

This is the central empirical finding of the whole verification route, and
the spec's own framing (§9) is blunt about why it matters: the same
stiffness that breaks *learning* contact dynamics (Parmar–Halm–Posa) breaks
naive *verification* of it too, because the one-step map steepens near the
contact guard and interval enclosures widen with it. Three milestones measure
three different faces of that same obstruction.

### M2 — single-step enclosure vs. stiffness (`artifacts/stiffness_sweep.json`)

On `SpringDamper2D`'s `vy'` component, across `k` from `10` to `10^7` (six
orders of magnitude), two boxes are measured: `straddle` (crossing the
contact guard) and `contact` (strictly inside contact), plus a single
in-contact `point` isolating pure rounding cost.

The true reachable width (`mc_width`, estimated by 4×10^5 Monte Carlo
samples) blows up by between one and two orders of magnitude over the sweep —
`straddle_mc_width` goes from `1.002` at `k=10` to `51.0` at `k=10^7`;
`contact_mc_width` from `0.997` to `80.99` over the same range. What stays
close to flat is *sharpness*, the ratio `enc_width / mc_width`: it sits
between roughly `1.0002` and `1.02` across the entire sweep for both boxes —
i.e. our interval enclosure never inflates the *true* reachable width by more
than ~2%. `point_width` (pure rounding cost, no reachable-set contribution)
stays at `~1e-16` to `~1e-13` throughout — negligible next to the
physics-driven blowup. Read together: the enclosure tracks the true
reachable set almost exactly at every stiffness tested; the width explosion
is the physics itself getting steeper, not our specific bound getting
looser. This is the "boundary of the method is a deliverable, not a secret"
result the spec asks M2 to publish as-is.

### M4 — certified horizon collapse (`tube_sweep.json`)

The natural follow-on: what does single-step widening do to a *K-step*
certified tube? Same `(y, vy)` subsystem, `K_MAX = 10`, clearance band
`0.15`. Measured achieved horizon vs. `k`:

| k | achieved horizon (of 10 requested) |
|---|---|
| 10.0 | 7 |
| 28.48 | 4 |
| 81.11 | 2 |
| 231.0 | 1 |
| 658 – 10^6 (all remaining rows) | 1 |

The certified horizon collapses from 7 steps at the lowest stiffness tested
down to a floor of 1 step by `k ≈ 231` and stays there for every higher
stiffness sampled, up to `k = 10^6`. Every single row in the sweep — *including
the least-stiff one, `k=10`* — also trips the script's own kill-criterion
check (`kill_by_5: true` on all twelve rows). That check has two clauses
(`tube_sweep.py:108`): the tube failed to reach step 5 at all, *or* its
`vy_width` at step 5 exceeds the declared `0.15` clearance band. Which clause
fires is worth separating, because they say different things. At `k=10` — the
best case measured here — the tube does reach step 5 and the width clause is
what trips it: `vy_widths[5] = 0.333`, more than double the band. For the
other eleven rows the achieved horizon is ≤ 4, so the tube never reaches step
5 and the horizon clause is what fires; those rows have no step-5 width to
compare (at `k=81.11`, for instance, the tube stops after 3 steps at a width
of `0.124`, still *inside* the band when it ran out of certified cover). Read
plainly, per spec §5's discipline of reporting shrinkage rather than hiding
it: within the parameter range this sweep covers, the M4 kill-criterion
checkpoint (spec §6: "if the tube engulfs the clearance band by K=5 at low
stiffness, stop and execute mitigations") fires everywhere it was tested, not
only at the high-stiffness end. The follow-on M4/M5/M6 milestones proceeded
using tighter, smaller-`D`, per-mode configurations (see §4 below) rather
than this sweep's wide unmitigated boxes — the sweep exists specifically to
show what happens *without* those mitigations.

### M6 — dimension and horizon scaling (`scaling_study.json`)

Two separate scaling questions, both explicitly asked for by spec §6.

**Input-dimension scaling**, at a fixed leaf-eval budget (a synthetic linear
contraction with a closed-form true sup-gap, so this isolates branch-and-bound
geometry from any training noise): the ratio of certified ε to the true gap
grows from `1.06×` at `d=2` to `5.00×` at `d=8` (the spec's own v1 input-dim
cap), at essentially constant leaf-eval count (~50,700). Branch-and-bound
splits one axis per refinement, so the same
budget buys an exponentially coarser per-axis width as dimension grows (the
same budget, and wall times of 0.14 s–0.30 s across the six dimensions) —
exactly the effect predicted in spec §9's risk list ("BnB blowup in input
dimension").

**Horizon scaling**, reusing the exact net/certificate/domain from the M4
acceptance test that reached `K=10` with zero tube escapes: pushed further,
the tube survives to an achieved horizon of **11** against a requested
horizon of 40, exiting the cover at step 11. So the free-flight tube used for
the M4 acceptance criterion has almost no slack beyond the horizon that test
already demanded — one extra step, not several.

### Where the route's boundary actually is

Composing the three: single-step enclosures at fixed, per-mode, in-contact
boxes stay sharp (M2, sharpness ≲1.02) even as stiffness rises six orders of
magnitude — the width growth there is physical, not a verification artifact.
But iterating that per-step map into a tube compounds the width geometrically
against a fixed cover, so the certified *horizon* collapses hard and early
(M4: 7→1 steps by `k≈231`, and the kill-criterion trips even at the mildest
stiffness tested). Separately, and orthogonally, the same branch-and-bound
machinery loses ground as input dimension grows at a fixed budget (M6:
1.06×→5.00× from d=2 to d=8), and even a favorable low-stiffness tube has
only a single certified step of headroom past whatever horizon was
targeted (M6: 11 vs. 10 requested). None of these is a bug; they are the
Parmar–Halm–Posa obstruction and BnB's own combinatorics showing up exactly
where the spec predicted they would.

Spec §9's mitigation order, in the sequence it's meant to be tried: **shrink
boxes near the contact manifold** (adaptive BnB already does this
automatically via mode exclusion and straddle-shrinking); **reduce dt**
(shrinks the one-step map's steepness directly); **certify per-mode**, letting
the mode predicate carry the discontinuity instead of asking one enclosure to
span it (this is exactly what made M6's pushing results below work at all —
each of stick/slide-left/slide-right is its own certificate over its own
mode); and, not yet implemented, **v2's smoothed-reference idea** — an
analytic smoothing-error term on top of a smoothed reference model, in the
style of reachable-tube constructions, meant to decouple the enclosure's
tightness from the raw stiffness of the true contact law.

---

## 4. M6 planar-pushing results

`pushing_conservatism_report.json` consolidates the three per-mode
certificates for `PusherSlider` (a=0.05, c=0.03, mu=0.3, dt=0.01,
py_max=0.04), each over its own mode's one-step `(py, vpx, vpy) → py2`
subsystem, mode membership handled exactly at the box level via
`PusherSlider.mode_certificate` (not by resampling):

| mode | ε | cover fraction | measured abstention | 2·α bar (α=0.05) | meets M6 bar |
|---|---|---|---|---|---|
| **stick** | 0.001001 | 99.78% | 0.16% | 10% | yes |
| slide_left | 0.005007 | 99.15% | 5.65% | 10% | yes |
| slide_right | 0.002947 | 99.15% | 3.70% | 10% | yes |

Spec §6's accept bar is explicit that only the **sticking mode** is required
to clear nominal abstention ≤ 2α; the other two modes get an honest write-up
whichever way they land. All three modes cleared the bar here. The stick
certificate is also the tightest in absolute ε by a wide margin (0.001001,
roughly 3–5× smaller than the other two modes'), at ε within ~2.4× of its
3×10^5-sample empirical floor and a cover of 99.8% of the declared domain over
126,368 leaves. Note that on the ε/floor *ratio* — how much conservatism
branch-and-bound added on top of what sampling could already see — stick is
not the best of the three: slide_right is tighter at 1.81× (against stick's
2.45× and slide_left's 2.72×). The two readings differ because each mode's
floor is against its own reachable set, and stick's is by far the smallest.
This is consistent with §3's read that per-mode certification is exactly
the mitigation that lets a route which collapses badly on the raw
free-flight tube still produce a usable certificate once the mode predicate
does the discontinuity's work. `slide_right`'s report additionally carries a
second-seed check, re-run as part of the test rather than recorded as prose: a
different data/fit seed (23/11 vs. 7/0) fits a little less tightly (train MAE
1.6×10^-4 vs. 8.8×10^-5) and certifies a looser ε (0.00682 vs. 0.00295, at the
same 99.15% cover), and its measured abstention rate of 3.8% still clears the
2α bar — so the pass on this mode is not a single-seed artifact.

---

## 5. Reproduction

The project is developed in a project-local, isolated virtualenv at
`certabstain/.venv` rather than through the (broken, shared) uv workspace at
`~/Downloads`. From `~/Downloads/certabstain`:

```bash
# run the full test suite
VIRTUAL_ENV="$(pwd)/.venv" uv run --active --no-sync pytest -q

# run a bare script directly (PYTHONPATH needed so `certabstain` resolves
# as a package one directory up; not needed for pytest above)
VIRTUAL_ENV="$(pwd)/.venv" PYTHONPATH="$(dirname "$(pwd)")" \
    uv run --active --no-sync python stiffness_sweep.py
```

Which artifact reproduces which result:

| Script / test | Produces | Section |
|---|---|---|
| `stiffness_sweep.py` | `artifacts/stiffness_sweep.json` (+ `.png`) | §3, M2 |
| `tube_sweep.py` | `artifacts/tube_sweep.json` (+ `.png`) | §3, M4 |
| `scaling_study.py` | `artifacts/scaling_study.json` (+ `.png`) | §3, M6 |
| `test_pushing_stick.py` | `artifacts/pushing_stick_report.json` | §4, M6 |
| `test_pushing_slide_left.py` | `artifacts/pushing_slide_left_report.json` | §4, M6 |
| `test_pushing_slide_right.py` | `artifacts/pushing_slide_right_report.json` | §4, M6 |
| `pushing_conservatism_report.py` | `artifacts/pushing_conservatism_report.json` (+ `.png`) | §4, M6 |

The three per-mode pushing reports are written as a side effect of running
their respective test modules under pytest (each ends by dumping its own
report JSON), not by standalone scripts; `pushing_conservatism_report.py`
then only reads and consolidates those three files — run the tests first if
regenerating from scratch.

Every number in §3 and §4 above was re-derived from these artifacts, and each
artifact re-derived from the script or test named here, as a check on this
note itself. Two things that check turned up and this note now reflects:
`stiffness_sweep.py` writes to `artifacts/`, while the cited copy of
`stiffness_sweep.json` sat at the repo root, left over from an earlier revision
of the script — re-running reproduced it to the last ulp, and the root
duplicate has since been removed so the cited path and the reproduction path
are the same one; and `pushing_slide_right_report.json`, alone
among the three, had no code that produced it (its figures were transcribed by
hand) until `test_pushing_slide_right.py` gained the same report-dumping step
its two siblings already had. Regenerating it reproduced every substantive
field exactly, so §4's table is unchanged — but the provenance now matches
what the paragraph above claims for it.

`artifacts/vnnlib/` holds a separate, independent cross-check: VNNLIB/ONNX
instances generated by `certabstain.vnnlib.generate_artifact_set`, pairing
each certified network with a property asserting the *negation* of this
repo's own computed IBP/CROWN output bounds. Per `artifacts/vnnlib/RUN.md`,
these are re-verified out-of-band with α,β-CROWN — expected verdict
`unsat`/`safe` on every instance, and a single `sat` would be a soundness
counterexample and a release blocker. Nothing in that directory depends on
this repo's code; it exists precisely so a third party can check our CROWN
implementation without trusting our code to check itself.

**Result (2026-08-03): `unsat` on all 24 instances**, α,β-CROWN 0.7.0 on CPU
with `double_fp: true`; per-instance verdicts in
`artifacts/abcrown_run_2026-08-03.json`.

Two things about that run are worth stating rather than burying. First, it
could not have succeeded before that date: every literal in all 24 instances
was written as `np.float64(0.5)` rather than `0.5` — `export_vnnlib`
formatted numpy scalars with `{v!r}`, and NumPy ≥ 2 reprs them wrapped. No
VNNLIB grammar accepts that, so every instance would have died on a parse
error, which is neither `unsat` nor `sat`. The well-formedness test did not
catch it because it interpolated the *same* formatter on both sides of its
assertion, and so was tautological.

Second, **at α,β-CROWN's float32 default, `net_15` returns `sat`, and that
verdict is spurious.** The PGD stage reports an attack margin of exactly
`0.00000000` while printing `Total number of violation: 0` in the same
breath — a tie at the boundary, not an escape. `net_15`'s true f64 margin is
`3.886e-15` against an f32 forward error of `5.704e-07` for these models
(recorded per instance in `bounds.json`), roughly 1.5×10⁸ times larger, so at
float32 the margin is simply not representable. With `double_fp: true` it
verifies `unsat` like the other 23, and evaluating its f64 ONNX at 60,005
points in the box via onnxruntime finds no violation. This is a precision
artifact of the verifier, not an unsound bound — but a reader who runs the
default configuration will see a `sat`, so it is documented here and in
`RUN.md` rather than left to surprise them.

---

## 6. Status and what's not yet done

M0 through M6 are complete; the full test suite is green (`pytest -q`,
run from the isolated `.venv` as above). This document — the technical note
with the lemma chain, the stiffness-boundary result, and reproduction
instructions — is itself the write-up half of M7 (spec §6). The VNNLIB
cross-check (§5) is the other verification-facing M7 deliverable; the
artifacts exist under `artifacts/vnnlib/` and the α,β-CROWN run was completed
on 2026-08-03 with `unsat` on all 24 instances. That closes the deliverable,
but it is one verifier on one machine on one run: it is a cross-check against
an independent implementation, not an independent audit, and re-running it is
worth more than citing this line.

An adversarial audit of the claims in this note, in `SPEC.md`, and in the
gate itself was run as part of M7, and three things it found are worth
recording rather than quietly fixing:

- **The gate's non-bypassability claim was, as originally written, false.**
  A public `authority` property handed out a `mint()` oracle that applied no
  score, threshold, or cover check, and the `authority=` constructor argument
  accepted any object whose `verify()` returned `True`. Both are closed: the
  property is now a mint-free `CertificateVerifier`, and the `authority=`
  argument was **removed outright** rather than type-checked. A type check was
  the first attempt and was not enough — a subclass overriding `verify()`
  still passes `isinstance`, and even a genuine `CertificateAuthority` leaves
  the *caller* holding `mint()` for the very instance the gate verifies
  against. The gate now constructs its own authority and accepts none, which
  is what makes "the policy cannot mint its own permission" true: a caller's
  own authority has a different key, so nothing it mints verifies here. The
  certificate's MAC
  now covers `score` and `threshold` rather than leaving them re-labellable,
  and the signature scan that enforces all of this is now an allowlist over
  the whole gate module instead of a six-name denylist over one class. The
  claim is now stated with its real scope: it is a property of the **API
  surface**, not memory isolation.
- **Spec §7 item 8's refusal half did not exist.** The certified horizon
  shrank and reported, but nothing refused when it came in under the horizon
  a deployment declared it needed. Both layers now carry the requirement:
  `propagate_tube` takes it as an optional early-out, and
  `PredictiveTubeWitness.build` — the layer a deployment actually passes
  through — requires the tube's requested horizon by default and raises
  `HorizonTooShort` otherwise. Accepting a truncated tube silently was not a
  cosmetic gap: it produced a witness scoring over a single step while
  presenting the K-step interface, which composes into a claim that reads as
  predictive. Best-effort is still available, as an explicit
  `required_horizon=None` that says so in the witness's own justification
  string.
- **Spec §7's "each is a distinct exception" was true only by message text**
  for items 3 and 9, which shared a bare `EnclosureError`. They now have
  distinct subclasses.

A second pass on 2026-08-03 found four more, and they share a shape worth
naming: *a guarantee enforced on one path but not its sibling.* In every case
the safety property existed, had a test, and was reachable — just not from the
call the documentation tells a user to make.

- **Spec §7.7 was dead on the documented path.** `build_monitor` constructed
  its `ActionGate` without `cover=`, silently dropping the cover predicate
  that W1/W2 witnesses carry. Every caller that actually got the check —
  `demo.py`, the M5 tests, the driver skill — bypassed `build_monitor` and
  built the gate by hand. Stepped at a state far outside the certified cover,
  a monitor built the documented way returned `abstained=False,
  reason='certified'` and **emitted a signed `SafetyCertificate`**: not merely
  a missing abstention, but an affirmative certificate for an action at a
  state where the guarantee had never been established.
- **`clearance_lower_bounds` trusted its caller's enclosure.** It discarded
  the returned upper end outright (`clo, _`), so ordering went unchecked. W2
  scores `s = max_t(c_required - clo)`, so an overstated lower bound makes
  clearance look larger and the score smaller. On the M4 4-D control tube with
  `c_required` set so the true clearance does *not* clear it, a correct
  clearance scores `+0.022` and the gate fires; an inverted one scores
  `-4.978` and the gate stays silent. A real danger signal turned into
  silence. Refused now, not clamped.
- **`nan` disabled three refusals**, because every comparison against it is
  False. `target=nan` was the worst: the same comparison drives refinement, so
  `umax > nan` marked nothing refinable and the search stopped at the crude
  initial bound. A net certifying to `eps=0.048` with `target=None` minted a
  certificate at `eps=2.94` under `target=nan` — sixty times weaker, recorded
  as having met its target. `min_cover_fraction` and the shift budget were the
  same defect with milder consequences.
- **Reference identity had nowhere to survive on the tube/W2 path.**
  `propagate_tube` enforced the network binding (A4) but `TubeResult` dropped
  the certificate's `reference_id`, and the clearance geometry — a *second*,
  independent reference — was never recorded at all. Two W2 witnesses built on
  the same tube against completely different obstacles produced byte-identical
  `justification()` strings, so the audit log could not say which geometry the
  claim was proved for. Both are recorded now. There is no ground truth to
  check the clearance against, so this records rather than inventing a
  comparison, and an unnamed geometry reads as `undeclared` rather than
  passing for a named one.

None of this changed a certified number: the §3 and §4 results stand exactly
as reported, and were re-derived from their artifacts as part of the same
pass. Each guard added in the second pass was checked by disabling it and
confirming its test fails — a guard whose test passes either way is not a
guard.

A third pass on 2026-08-07 found the same shape one level down: not a call
site forgetting to pass a flag, but a safety invariant that lived only in a
classmethod factory (`bind()`, `build()`, `compose()`, `fit()`) while the
dataclass's own constructor — reachable directly, with no flag to forget —
enforced nothing.

- **`Interval.lo`/`.hi`, `MLP` weights, and `EpsilonCertificate`'s arrays were
  not actually immutable.** `setflags(write=False)` looks like a freeze, but
  an array that owns its data can flip `WRITEABLE` back to `True` on request
  — confirmed empirically: `iv.lo.setflags(write=True); iv.lo[0] = 999.0`
  silently succeeded on what §2 above calls "immutable after construction,"
  including into an inverted `lo > hi` state the constructor itself would
  have refused. Fixed by round-tripping every such array through `bytes`
  (`interval._freeze`), which produces a view whose base has no `WRITEABLE`
  flag to flip; `setflags(write=True)` on the result now raises `ValueError`
  instead of succeeding. Covered by
  `test_interval_endpoints_cannot_be_unfrozen_via_setflags`,
  `test_mlp_weights_cannot_be_unfrozen_via_setflags`, and
  `test_certificate_arrays_cannot_be_unfrozen_via_setflags`. "Immutable after
  construction" in §2 is now true of the object, not just of the one call
  path that used to set the flag.
- **`TwoSidedClaim`'s floor-clearance check ran only in `compose()`.** Calling
  the dataclass constructor directly — `TwoSidedClaim(miss_bound=0.9,
  threshold=1.0, violation_floor=0.5, ...)` — built cleanly with no error,
  producing an object that claims a zero miss rate while actually admitting
  one. Every field the check needs (`threshold`, `violation_floor`,
  `miss_bound`) is already a stored field, so the fix moves the check into
  `__post_init__`, which every construction path runs. `compose()` is
  unchanged; it now duplicates a check that no longer depends on going
  through it. Covered by
  `test_two_sided_claim_direct_construction_reruns_composes_check`.
- **`PredictiveTubeWitness.build()`'s horizon check had the same hole**,
  fixed the same way (moved into `__post_init__`, all inputs already stored
  fields) — covered by
  `test_direct_construction_bypassing_build_still_enforces_horizon`.
  **`VerifiedDiscrepancyWitness.bind()`'s spec A1/A4 checks** need `net` and
  `reference_id` values that are not stored on the resulting witness, so they
  cannot be re-derived from stored fields the way the others were; that
  factory now uses a private sentinel-token field only `bind()` can set,
  matching the existing `_RequireRequested` idiom already in the same file
  for "unset" defaults. Direct construction now raises `TypeError` before any
  witness is produced. Covered by
  `test_direct_construction_bypassing_bind_is_rejected`.
- **`SplitConformalCalibrator.fit()` and `MondrianCalibrator.fit()` had the
  same hole for their structural checks** (`alpha` range, `n >= 1`,
  `order_statistic` in range, finite `threshold`; for Mondrian, every
  stratum's `alpha` agreeing with the calibrator's own). Direct construction
  — e.g. `SplitConformalCalibrator(threshold=-999.0, alpha=5.0, n=-1,
  order_statistic=999)` — previously built cleanly and reported
  `coverage_lower=-4.0`, a probability outside `[0, 1]`. Fixed with
  `__post_init__` re-checks on both classes, covered by
  `test_split_conformal_direct_construction_rejects_nonsense_fields` and
  `test_mondrian_direct_construction_rejects_inconsistent_alpha`. This is
  narrower than `fit()`'s full check: `fit()` also verifies `threshold` is a
  genuine order statistic of the raw calibration sample, and the raw sample
  is not a stored field, so that half cannot be re-derived from a
  already-built calibrator. What's re-checked is every invariant that *can*
  be checked from stored fields alone — recorded as a residual gap below,
  not fixed by pretending otherwise.

The same audit swept every other frozen dataclass in the package
(`gate.py`'s `SafetyCertificate` and `GateDecision`, `tube.py`'s
`TubeResult`, `reference.py`'s three parameter dataclasses, `nnbound.py`'s
`MLP.random()`, `discrepancy.py`'s `EpsilonCertificate`/`certify_epsilon()`)
for the same shape and found no further instance: `SafetyCertificate`'s
guarantee is an HMAC tag checked by `CertificateVerifier`, not constructor
validation, so a forged direct construction still fails `verify()`;
`GateDecision` and `TubeResult` assert no invariant of their own; the
reference models and `MLP.random()` have no factory beyond
`__post_init__` to bypass; and `certify_epsilon()`'s two checks that are not
re-derivable from `EpsilonCertificate`'s stored fields (`CoverTooSmall`,
`TargetNotCertified`) are refusals about whether the *caller's* preferences
were met, not about the soundness of the resulting `eps`/cover — bypassing
them yields a certificate with a looser-than-requested or undersized-cover
bound, not an unsound one, and `__post_init__` already re-checks the one
invariant that is load-bearing for every consumer: `eps` must be finite.

Two residual gaps, left as gaps rather than papered over: (1) the
`SplitConformalCalibrator`/`MondrianCalibrator` re-check above cannot verify
`threshold` is a real order statistic of *some* calibration sample, because
the raw sample is never stored on the object; a caller can still hand-pick a
structurally-valid-looking `(threshold, n, order_statistic)` triple that
`fit()` would never have produced.

A 2026-08-07 addition narrows this gap without closing it. `fit()` now also
records `sample_digest`, a BLAKE2b-256 hash of the sorted calibration sample
(same construction as `discrepancy.weights_hash`), and both calibrators gained
`verify_sample(scores)` (`MondrianCalibrator.verify_sample(group, scores)`
delegates to the named stratum's calibrator). It recomputes the digest from
an independently-supplied `scores` array and checks both that the digest
matches and that `sorted(scores)[order_statistic - 1] == threshold`. A pass
proves `threshold` is a genuine order statistic of a sample that hashes to
`sample_digest` — strictly more than the bare structural re-check above can
show. It does not prove:

- that the `scores` handed to `verify_sample` are the actual rollout data
  collected at calibration time — only that they hash to what `fit()`
  recorded. A digest proves consistency with supplied bytes, not chain of
  custody, the same way `weights_hash` proves byte-identity to a specific
  network but not that the network is the intended policy.
- that `order_statistic` is the *correct* index for this `alpha`/`n` via
  `k = ceil((n + 1) * (1 - alpha))`. `order_statistic` and `threshold` can
  still be edited together to point at a different-but-real position in the
  same digested sample, and `verify_sample` accepts it — demonstrated
  directly by `test_verify_sample_known_gap_same_index_different_k_still_verifies`,
  which is a record of the gap, not a bug report.
- anything, for a calibrator with `sample_digest=None` — every calibrator
  built by direct construction without one, and every one built before this
  field existed, has nothing recorded to check against, so `verify_sample`
  fails closed (`False`) rather than treating "nothing to compare" as
  success.

Covered by `test_verify_sample_accepts_the_real_calibration_sample`,
`test_verify_sample_rejects_a_different_sample`,
`test_verify_sample_rejects_a_forged_threshold`,
`test_verify_sample_false_when_no_digest_was_recorded`,
`test_verify_sample_known_gap_same_index_different_k_still_verifies`, and
`test_mondrian_verify_sample_delegates_per_stratum`.

(2) all of the sentinel-token and
`__post_init__` guards in this pass are, like the gate's non-bypassability
claim, statements about the supported API surface — code already running in
the same process can still reach `object.__setattr__` or a private module
attribute directly. Neither gap is new; both are the same "API surface, not
memory isolation" scope the gate claim already declares in §2 and the
project README, extended here to the same trust boundary applying to every
frozen dataclass in the package, not just the gate's.

The provisional-patent outline is a separate document, drafted in parallel
by a different workstream, and is explicitly out of scope for this note —
per spec §12, filing it is gated on a professional freedom-to-operate search
over the relevant dockets, which is a precondition to filing, not a
formality to skip. Nothing here should be read as, or substituted for, that
outline or that search.
