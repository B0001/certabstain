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
(`tube_sweep.py:107`): the tube failed to reach step 5 at all, *or* its
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
these are meant to be re-verified out-of-band with α,β-CROWN — expected
verdict `unsat`/`safe` on every instance, and a single `sat` would be a
soundness counterexample and a release blocker. Nothing in that directory
depends on this repo's code; it exists precisely so a third party can check
our CROWN implementation without trusting our code to check itself.

---

## 6. Status and what's not yet done

M0 through M6 are complete; the full test suite is green (`pytest -q`,
run from the isolated `.venv` as above). This document — the technical note
with the lemma chain, the stiffness-boundary result, and reproduction
instructions — is itself the write-up half of M7 (spec §6). VNNLIB
cross-check artifacts (§5) are the other verification-facing M7 deliverable
and already exist under `artifacts/vnnlib/`, awaiting an out-of-band
α,β-CROWN run by whoever picks that up.

An adversarial audit of the claims in this note, in `SPEC.md`, and in the
gate itself was run as part of M7, and three things it found are worth
recording rather than quietly fixing:

- **The gate's non-bypassability claim was, as originally written, false.**
  A public `authority` property handed out a `mint()` oracle that applied no
  score, threshold, or cover check, and the `authority=` constructor argument
  accepted any object whose `verify()` returned `True`. Both are closed (a
  mint-free `CertificateVerifier`, and a type check), the certificate's MAC
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

None of this changed a certified number: the §3 and §4 results stand exactly
as reported, and were re-derived from their artifacts as part of the same
pass.

The provisional-patent outline is a separate document, drafted in parallel
by a different workstream, and is explicitly out of scope for this note —
per spec §12, filing it is gated on a professional freedom-to-operate search
over the relevant dockets, which is a precondition to filing, not a
formality to skip. Nothing here should be read as, or substituted for, that
outline or that search.
