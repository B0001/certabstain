# certabstain Phase 2 — Verified ε

**Spec v0.1 · 2026-07-26 · status: draft for approval**

Phase 1 delivered a two-sided abstention monitor whose zero-miss guarantee is
conditional on an *assumed* certified model-error bound ε. Phase 2 removes the
assumption: build the machinery that **produces** ε for a learned contact-
dynamics or clearance model, as a machine-checked artifact, and wire it into
the existing witness and gate so the composition is certified end to end.

The route (per the July 2026 survey) is Route A: verify the discrepancy between
the learned model and a trusted reference by sound interval computation over a
declared operating domain. No unknown Lipschitz constant, no unknown RKHS norm
— the two assumptions that survive are explicit and testable.

---

## 1. Objective and the claim

Build a certifier that, given a trained network `ĝ`, an interval-extendable
reference model `g`, and a declared domain `D` inside a declared contact mode
`M`, outputs an **EpsilonCertificate** establishing, per output dimension `i`:

```
sup over (x,u) in Cover  of  | ĝ_i(x,u) − g_i(x,u) |   ≤   ε_i
```

where `Cover ⊆ D ∩ M` is the certified sub-region (see §5 on partial
certification), and the inequality is sound with respect to exact real
arithmetic: every operation on the certification path uses outward-rounded
interval arithmetic, so floating point can only loosen ε, never invalidate it.

The certificate additionally binds: a hash of the exact network weight bytes,
the reference model identity and parameters (stiffness, timestep, geometry),
the cover geometry, and the per-dimension ε vector. The runtime witness
refuses if any binding fails — a retrained network, a changed timestep, or a
state outside the cover each void the certificate *detectably* instead of
silently.

Downstream, two witnesses consume the certificate:

**W1 — direct (one-step).** `ĥ` is a learned clearance/safety function; the
certified ε plugs straight into Phase 1's `CertifiedModelErrorWitness`,
unchanged. This is the first end-to-end certified composition and lands early
(M3).

**W2 — predictive (K-step tube).** `ĝ` is a learned one-step dynamics model.
At runtime, propagate an interval tube K steps under `ĝ ⊕ [−ε, ε]`; since the
true reference trajectory lies inside the tube while the tube remains in the
cover, a certified positive clearance lower bound over the whole tube makes
silence sound over the horizon. Same violation-floor-zero structure as Phase 1,
now with lookahead (M5).

---

## 2. Assumption ledger

Every assumption is declared, defended, and where possible runtime-checked.
Anything not on this list is not assumed.

**A1 — the reference model is the ground truth of the guarantee.** The
certificate is *reference-relative*: it certifies `ĝ` against a declared
analytic contact model, not against physical reality. Sim-to-real is
explicitly outside the deterministic claim (see Non-goals). Defense: the
reference is implemented twice — float64 for data generation, interval
arithmetic for certification — with a consistency test that every float
evaluation lies inside its interval evaluation. Consequence: MuJoCo/Drake are
**demoted to sanity checks**; they are black boxes and cannot be interval-
extended, so they cannot be the reference the proof is about.

**A2 — the certified region lies inside one declared contact mode** (or a
declared finite mode set, each certified separately). Defense: mode membership
is an interval-checkable predicate; the branch-and-bound excludes boxes that
straddle the mode boundary. Runtime check: the gate abstains outside the
certified cover, so the assumption's failure is an abstention, not a silent
void.

**A3 — float soundness by outward rounding.** All certification-path
arithmetic uses directed-rounding interval ops (nextafter-based). Startup
self-test verifies rounding behavior on a battery of boundary cases and
refuses to certify if the environment misbehaves.

**A4 — the deployed network is the certified network.** The certificate binds
a BLAKE2b hash of the canonical weight serialization; the witness recomputes
and compares at load. One flipped byte voids it.

---

## 3. Architecture

New modules extend the existing package; the Phase 1 core is untouched except
for one addition to the gate (cover-membership check).

```
certabstain/
  interval.py      outward-rounded interval arithmetic (the substrate)
  nnbound.py       IBP + CROWN-style linear bounds; Jacobian enclosures
                   for small MLPs (ReLU first; tanh behind a flag)
  reference.py     interval-extended analytic contact models + mode
                   predicates; float twin for data generation
  discrepancy.py   branch-and-bound certifier → EpsilonCertificate
  tube.py          K-step interval tube + clearance evaluation
  witness2.py      VerifiedDiscrepancyWitness (W1 binding checks) and
                   PredictiveTubeWitness (W2)
  vnnlib.py        export network + property to VNNLIB/ONNX for
                   cross-checking against α,β-CROWN
```

Dependency policy: **numpy + stdlib only on the certified path.** Training the
small nets is done in numpy (they are tiny by design). mpmath is a dev/test
dependency (high-precision oracle). onnx is optional, export-only. No torch
requirement anywhere.

Owning the verifier instead of depending on auto_LiRPA is deliberate: the
networks are small, the soundness story must include our own rounding
discipline end to end, and the external cross-check (vnnlib.py + α,β-CROWN
run out-of-band) covers the "did we get CROWN right" risk without importing a
GPU stack into the certified path.

---

## 4. The certified chain

Four lemmas, each carried by code and each validated by the test plan.

**L1 (enclosure).** For the interval extension `IA[f]` of any supported `f`
and any box `B`: for all `z ∈ B`, `f(z) ∈ IA[f](B)`, in exact arithmetic and
under outward rounding.

**L2 (one-step discrepancy).** On a leaf box `B`, with per-dimension
enclosures `IA[ĝ](B)` and `IA[g](B)`:

```
sup_B (ĝ_i − g_i) ≤ hi(IA[ĝ_i](B)) − lo(IA[g_i](B))
sup_B (g_i − ĝ_i) ≤ hi(IA[g_i](B)) − lo(IA[ĝ_i](B))
u_{B,i} = max of the two;   ε_i = max over leaves of u_{B,i}
```

v1 uses this two-enclosure bound; a v1.1 tightening subtracts correlated
linear parts (CROWN linear bounds on `ĝ` against a linearization of `g`)
before enclosing the remainder.

**L3 (tube).** If `X_t × U_t ⊆ Cover`, then the true reference successor set
satisfies `g(X_t, U_t) ⊆ IA[ĝ](X_t × U_t) ⊕ [−ε, ε]^n`. Iterating while the
containment-in-Cover precondition holds yields a tube containing every true
trajectory from `X_0` under the declared controls. If the tube exits the
cover at step `t*`, the certificate's horizon is `t* − 1` — reported, never
papered over. A scalar discrete-Grönwall bound via certified Jacobian
enclosures is computed alongside; the tighter of the two is used.

**L4 (sound silence, W2).** Score `s = max over t ≤ K of
(c_required − lo(IA[h](X_t)))`. Then `s ≤ 0` implies true clearance
≥ c_required for all `t ≤ K`. Violation floor 0; Phase 1's
`TwoSidedClaim.compose` applies unchanged, and conformal calibration of `s`
on nominal rollouts supplies the false-alarm side exactly as before.

---

## 5. Partial certification

Branch-and-bound will not always certify all of `D` within budget (mode
boundaries, stiff regions). Design decision: **certification fails to a
smaller domain, never to a weaker claim.** The certificate carries the exact
certified cover (a box tree); the gate checks cover membership every step and
abstains outside it. A requested-but-uncertified region is an abstain region
by construction. A hard refusal still triggers if the certified cover falls
below a declared minimum fraction of `D` (default 90%, configurable), so a
certificate that quietly covers 3% of the intended envelope cannot ship.

This converts Phase 1's one honest weakness — the demo row where ε was
violated and nothing warned you — into a detected abstention: the guarantee's
precondition is now itself runtime-checkable.

---

## 6. Milestones

Effort assumes solo work; ranges are estimates, acceptance criteria are not.

**M0 — Interval substrate (3–5 days).** `interval.py` with +, −, ×, ÷, exp,
tanh, sqrt, min/max, matrix ops, all outward-rounded; rounding self-test.
*Accept:* 10⁶ randomized expression trees × random points: zero containment
violations; agreement with 100-digit mpmath enclosures on a 10⁴ subset;
boundary-case battery (denormals, sign flips at zero, nextafter edges) passes.

**M1 — Network bounds (1–2 weeks).** IBP over boxes; CROWN backward linear
bounds; interval Jacobians. ReLU fully; tanh behind `--experimental` with
certified tangent-line relaxations.
*Accept:* 1,000 random MLPs (≤4 layers, width ≤64, input dim ≤8), 10⁵ sampled
inputs per box: zero enclosure violations; CROWN strictly tighter than IBP on
≥90% of boxes with median width ratio reported; Jacobian enclosures contain
finite-difference and analytic gradients at all samples; VNNLIB export
cross-checked against α,β-CROWN on ≥20 instances with no contradiction.

**M2 — Reference models (1 week).** (a) 2-D point mass with stiff
spring-damper ground contact, semi-implicit Euler, closed form; (b)
quasistatic planar pushing, ellipsoidal limit surface, motion-cone mode
predicates (stick / slide-left / slide-right). Float twin + interval twin.
*Accept:* 10⁶ random states per model: float result ∈ interval result, always;
stiffness sweep documented — enclosure width vs. contact stiffness curve
published as-is (this is the Parmar–Halm–Posa obstruction made measurable, and
it defines where this route dies; we publish the boundary, we don't hide it).

**M3 — Discrepancy certifier + first end-to-end (1–2 weeks).** Adaptive
branch-and-bound over `D ∩ M`; `EpsilonCertificate` with cover tree, bindings,
budget stats; refusal paths.
*Accept:* on the spring-damper task, a learned clearance net (input dim ≤4)
certified with ε ≤ 3× the empirical sup-gap (10⁷ samples) within ≤10⁶ leaf
evaluations; setting the target below the empirical gap forces a refusal that
reports the achieved bound and the worst uncertified leaf; **the Phase 1 demo
reruns with ε produced, not assumed** — first fully certified composition.

**M4 — Tube (1 week).** K-step propagation + clearance evaluation + Grönwall
comparison; certified-horizon shrinkage on cover exit.
*Accept:* K = 10 on spring-damper: 10⁵ Monte Carlo true rollouts, zero tube
escapes; tube-width vs. K curve reported. **Kill-criterion checkpoint:** if
the tube engulfs the clearance band by K = 5 at *low* stiffness, stop and
execute mitigations (shrink `D`, smaller net, smaller dt) before any further
milestone.

**M5 — Witnesses + gate integration (1 week).** `VerifiedDiscrepancyWitness`
(W1), `PredictiveTubeWitness` (W2), cover-membership check in `ActionGate`.
*Accept:* one flipped weight byte ⇒ refusal at load; reference-parameter
mismatch ⇒ refusal; state driven out of the cover ⇒ abstention with reason
"left certified domain"; demo's old silent-void row now reads as a detected
abstention; all Phase 1 tests still green.

**M6 — Planar pushing scale-up (1–2 weeks).** Full pipeline on the pushing
model, per-mode certification unioned over the three modes, conservatism
report (certified ε vs. empirical gap vs. abstention rate), input-dim and
horizon scaling study.
*Accept:* certified two-sided monitor on at least the sticking mode with
nominal abstention ≤ 2× the conformal α; honest write-up of any mode that
fails to certify and why.

**M7 — Write-up + IP (1 week).** Technical note with the lemma chain, the
stiffness-boundary result, and reproduction scripts; VNNLIB artifacts;
provisional outline. Not legal advice; a professional FTO search is a
precondition to filing (see §10).

---

## 7. Refusal surface

Enumerated so no failure can be soft. Each is a distinct exception or a
distinct abstention reason, and each has a test.

1. Branch-and-bound budget exhausted with ε above target → refuse; report
   achieved ε and worst leaf.
2. Certified cover below the declared minimum fraction of `D` → refuse.
3. Requested box not expressible inside any declared mode → refuse
   (`ModeIndeterminate`).
4. Weight-hash mismatch at witness load → refuse.
5. Reference model identity/parameter mismatch → refuse.
6. Rounding self-test failure at startup → refuse to certify anything.
7. Runtime state outside the certified cover → abstain (gate).
8. Tube exits cover before the required horizon → certified horizon shrinks;
   refuse if below the horizon the deployment declared (`HorizonTooShort`).
   The requirement is declared at **witness construction**
   (`PredictiveTubeWitness.build`), defaulting to the tube's requested
   horizon, because that is the layer a deployment passes through;
   `propagate_tube` takes the same argument as an optional early-out, but its
   real callers are the sweeps, which want the permissive default. A
   requirement stated only there would be stated nowhere that deploys.
9. Non-finite anywhere on the certification path → refuse/abstain, as in
   Phase 1 (`NonFiniteEnclosure`; at the gate, the abstention reason
   "non-finite monitor score").

Items 3 and 9 are subclasses of `EnclosureError` rather than the bare class,
so the "distinct exception" above holds under a *type* reading and not only by
matching message text — while any existing handler catching `EnclosureError`
keeps working. The bare class is retained for the conditions that are neither:
domain violations (division by an interval containing zero, `sqrt` below zero)
and caller-side shape mismatches.

---

## 8. Test plan

Three tiers, mirroring Phase 1's discipline. **Theorem validation:** Monte
Carlo containment tests for L1–L3 (the numbers in M0/M1/M4), plus adversarial
corner sampling (box vertices, mode-boundary-adjacent points, near-zero
crossings for ReLU). **Refusal coverage:** one test per item in §7. **Freeze
tests:** the certificate dataclass is frozen/slotted; the gate's public
surface remains exactly Phase 1's plus nothing (inspect-based test, extended
from Phase 1); no bypass parameter anywhere in the `gate` module's public
surface, enforced by the same signature scan.

The scan is an **allowlist**, deliberately: it names every permitted public
attribute and every permitted parameter of every class exported by `gate`
(`ActionGate`, `CertificateAuthority`, `CertificateVerifier`,
`SafetyCertificate`, `GateDecision`), so a new one fails the test rather than
passing under a name no denylist anticipated. A mutation study is what forced
this: against a denylist version, a new public `CertificateAuthority.
mint_unchecked`, a new `allow_any=` parameter on `mint`, and a public
*instance* attribute honoured by `step` all shipped green. The last of those
is why the scan also asserts `__slots__` — `dir()` is a class scan and cannot
see instance attributes.

Two limits of the freeze tier are worth stating rather than implying:
`__slots__` prevents *new* attributes, not assignment to existing private
ones, and no in-process Python check can prevent code that already runs in
the interpreter from reaching a private attribute. The gate's claim is
therefore about its **API surface** — no supported call sequence yields an
uncertified emission or a certificate the gate did not itself authorize —
plus deployment separation of the certificate authority. Memory isolation
against hostile in-process code is not claimed and would need a separate
process or an HSM.

External cross-check: the VNNLIB instances and α,β-CROWN results are committed
as artifacts (`artifacts/vnnlib/`, verdicts in
`artifacts/abcrown_run_2026-08-03.json`) so a third party can re-verify
without our code. Run α,β-CROWN with `double_fp: true`; at its float32
default one instance returns a spurious `sat`, explained in
`artifacts/vnnlib/RUN.md`.

---

## 9. Risks and kill criteria

**Stiffness kills interval tightness** (the central technical risk). As
contact stiffness rises, the reference's interval enclosures widen and ε goes
vacuous — the same obstruction that breaks learning breaks naive verification.
Mitigations, in order: shrink boxes near the contact manifold (adaptive BnB
already does this), reduce dt, certify per-mode with the mode predicate doing
the discontinuity's work, and (v2) a smoothed reference with an analytic
smoothing-error term à la the reachable-tube construction. Kill criterion is
M4's checkpoint; the stiffness-boundary curve from M2 is the early-warning
instrument.

**BnB blowup in input dimension.** v1 hard-caps input dim at 8; the pushing
task needs ≤5. Beyond that is out of scope until zonotope-based enclosures
(v2).

**CROWN implementation error.** Covered by the α,β-CROWN cross-check and the
containment tests; any contradiction is a release blocker.

**Grönwall vacuity.** Expected for L̂ > 1 over long horizons; that is why the
tube is primary and the scalar bound is only a comparison.

---

## 10. Non-goals (v1)

Sim-to-real certification — the deterministic claim is reference-relative,
full stop; an optional conformal layer over real-world residuals may be added
later but is statistical, separately labeled, and never feeds the miss bound.
Multi-contact beyond the declared mode set. Input dim > 8. GPU anything.
MuJoCo/Drake as reference models (sanity comparison only). Torch dependence.
Certifying the mode *classifier* for learned mode boundaries — flagged by the
survey as open and a candidate Phase 3 invention, not smuggled into v1.

---

## 11. Open design questions (decide by end of M1)

Per-output-dim ε vector vs. scalar max (leaning vector — tubes benefit).
Control input handling: v1 open-loop declared `U_t` boxes; affine feedback
tubes (Li & Chou style) deferred to v2. tanh relaxation certification depth
(ship ReLU-only if tanh slips). Cover representation: box tree v1; merge to
zonotopes v2.

---

## 12. IP checkpoints

File the provisional after M5 (first fully certified composition with the
predictive witness) and before any public write-up. Claim emphasis, per the
survey's prior-art read: (i) a certified deterministic error bound on a
*learned* function as the load-bearing input to (ii) a deterministic
(proven-not-measured) miss bound, with (iii) domain-carrying, weight-hash-
bound single-use certificates enforced by a non-bypassable gate. Design
around ModelPlex (US 10,872,187): its ε bounds a hand-authored hybrid model
with no learned-function certification step — keep that distinction explicit
in the claims. Professional FTO search over G05B/G06N and the runtime-
assurance dockets (Boeing/Collins, NASA, NVIDIA, TRI) is a precondition to
filing, not a nice-to-have.
