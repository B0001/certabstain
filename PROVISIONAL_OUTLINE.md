# Provisional Patent Outline — certabstain (draft for attorney review)

> **THIS IS NOT LEGAL ADVICE.** This document is an engineering-drafted
> outline, prepared by the project's own engineer, for handoff to a
> registered patent attorney. It is not a filing-ready application, it does
> not constitute legal advice, and it makes **no claim about patentability,
> validity, novelty, or freedom-to-operate**. Per SPEC.md section 12: *"a
> professional FTO search is a precondition to filing, not a nice-to-have."*
> That search — over CPC classes G05B/G06N and the runtime-assurance dockets
> named below (Boeing/Collins, NASA, NVIDIA, TRI) — has not been performed
> and must occur, together with attorney review, before any application is
> drafted or filed. Nothing below should be read as an assertion that any
> claim idea is allowable, novel, or non-obvious.

---

## 1. Field of the invention

Runtime safety monitoring and verification for control systems that use
learned (machine-learned) components — specifically, methods and systems for
producing a **deterministic (non-statistical) bound on the probability of a
missed safety violation** for a control loop whose dynamics or
safety-relevant output is computed, in part, by a trained neural network.
The field spans certified numerical verification of neural networks (interval
arithmetic, branch-and-bound bound propagation), runtime assurance
architectures, and certificate-gated actuation.

---

## 2. Background / problem with prior art

Published runtime monitors for learned robot policies — SAFE, FIPER,
FAIL-Detect, Sentinel, and UNISafe (see the accompanying prior-art survey,
`invention_idea.md`) — calibrate a failure score against nominal or
successful rollouts and report a **statistical, marginal false-alarm bound**.
None of them reports a bound on missed failures. This is not an oversight:
several of these papers state explicitly, in their own text, that a
miss-rate (true-positive) guarantee would require calibration against
*failure* rollout data, which is exactly the data that is unavailable for a
safety-critical system (that is the point of building a monitor in the first
place). This is a structural gap in the published art, not merely an
unaddressed detail — a monitor calibrated only on successes cannot, by
construction, bound how often it stays silent during a failure it has never
seen.

The closest patented prior art identified by the survey (`compass_artifact_
wf-b56dd5e1..._text_markdown.md`, Part 7) is **ModelPlex, US 10,872,187 B2**
("Verified runtime validation of verified cyber-physical system models,"
Platzer & Mitsch, CMU/DARPA lineage). ModelPlex synthesizes, via theorem
proving, a provably-correct runtime monitor from a verified **hand-authored**
hybrid model, with an ε-bounded plant/disturbance term and a provably-safe
fallback on non-compliance. That patent's ε bounds deviation of the *real
plant from a hand-authored analytic model* — there is no step in it that
certifies the error of a *learned, trained* function against anything. This
distinction is the crux of the intended claim differentiation and should be
kept sharp and explicit in any drafted claims: **ModelPlex's monitored
quantity is never the output of a trained network; this system's is.**

---

## 3. Summary of the invention

The system composes three elements that, per the survey's prior-art read
(SPEC.md section 12), do not appear together as an integrated unit in the
published or patented literature:

**(i) A certified, deterministic error bound on a *learned* function.**
Given a trained network `ĝ`, an interval-extendable reference model `g`
(one that can itself be evaluated with sound interval arithmetic, not a
black-box simulator), and a declared operating domain inside a declared
contact/operating mode, an adaptive branch-and-bound procedure computes a
per-output-dimension bound `ε_i` such that `sup |ĝ_i − g_i| ≤ ε_i` over a
certified cover of that domain, sound with respect to exact arithmetic under
outward (directed) rounding. This is distinguished from statistical/marginal
bounds (conformal prediction) and from bounds that require an assumed,
unverifiable regularity constant (a Lipschitz constant or an RKHS-norm
bound) — no such assumption appears anywhere on this certification path.

**(ii) A deterministic, proven-not-measured miss bound built on that error
bound.** Rather than estimating a miss rate from failure data, the monitor's
score is constructed so that a specification violation cannot produce a
small score — a *violation floor* that is a theorem about the score's
construction, not an empirical observation. In the one-step (direct) case
the score takes the form `s = ε − ĝ(x)` (informally: the certified error
margin less the network's own output), so `s ≤ 0` is only possible when the
true value could be within the violation region; propagated over a horizon,
the same construction extends to a K-step interval tube under `ĝ ⊕ [−ε, ε]`,
whose certified clearance lower bound over all `t ≤ K` gives the same
zero-violation-floor structure with lookahead. The statistical (false-alarm)
side and this structural (miss) side are then composed into a single
two-sided claim only when a calibrated threshold clears the violation floor;
otherwise the composition declines to exist and reports a one-sided
certificate instead of silently overstating its guarantee.

**(iii) Domain-carrying, weight-hash-bound, single-use certificates enforced
by a non-bypassable gate.** The certificate produced by (i)/(ii) binds: a
cryptographic hash of the exact deployed network weights, an identity string
for the exact reference model and its parameters, the exact certified cover
geometry, and the per-dimension ε vector. A gate component is architecturally
the *only* path by which an actuation command can be emitted, and it refuses
to emit one without a certificate that authenticates (via a keyed MAC held
only by a separate authority, never by the policy), is bound to the current
control epoch and to the exact observation/action pair, and has not been
consumed before. There is no override, no force flag, no strict-off
parameter anywhere in the gate's public surface — a property that is itself
enforced by an automated signature scan, not merely a code-review convention.

---

## 4. Independent claim sketches

*(Outline form only — plain engineering description of coverage a claim
might seek, not proposed legal claim language. An attorney would need to
draft actual claim language, choose claim type/breadth, and check each
against FTO results before anything is filed.)*

**Claim idea A — method, certified learned-function discrepancy bound.**
A computer-implemented method for producing a sound, deterministic bound on
the discrepancy between a trained neural network's output and a reference
model's output over a declared operating domain, comprising: obtaining
interval (or other sound outward-rounded) enclosures of both the network and
the reference model over sub-regions of the domain via adaptive
branch-and-bound refinement; excluding or further splitting sub-regions that
straddle a declared operating-mode boundary; and producing a certificate
recording, for each output dimension, an upper bound on the discrepancy that
holds over the resulting certified cover, together with a binding to the
exact network weights and reference parameters used. *This is the claim
that most directly targets the gap the survey identifies — a deterministic
bound whose subject is a learned/trained function — and is the sharpest
point of departure from ModelPlex, whose analogous ε bounds only a
hand-authored model.*

**Claim idea B — method, two-sided composition with a structural miss
floor.** A method for producing a runtime monitor with a bounded probability
of both false alarms and missed violations without calibration against
failure data, comprising: constructing a monitor score from a certified
discrepancy bound such that the score cannot fall below a computable
threshold when a specification violation is present (a "violation floor");
separately calibrating a false-alarm bound on the same score using only
nominal-rollout data via a distribution-free (e.g., conformal) procedure;
and composing the two only when the calibrated threshold clears the
violation floor, refusing to produce a two-sided claim otherwise.

**Claim idea C — method, K-step predictive tube extension.** A method for
extending a one-step certified discrepancy bound into a certified,
multi-step, deterministic clearance guarantee, comprising: propagating an
interval state tube forward under the learned dynamics model enlarged by
the certified per-step error bound; evaluating a safety/clearance function
against the tube at each step; computing a comparison bound via a discrete
Grönwall-type argument and using the tighter of the two; and shrinking the
certified horizon (rather than the claim's validity) when the tube would
exit the certified operating domain before the declared horizon.

**Claim idea D — system, certificate-gated, non-bypassable actuation
architecture.** A system comprising an actuation gate that is the sole path
to command emission and that will not emit a command absent a certificate
that (a) is authenticated by a key held only by a separate certificate
authority and never exposed to the controlling policy, (b) binds a hash of
the deployed network weights, an identity of the reference model and its
parameters, and the geometry of a certified operating-domain cover, (c) is
scoped to a single control epoch and a single observation/action pair, and
(d) is rejected on any binding mismatch, replay, or staleness, with no
override, force, or strict-disable parameter present in the gate's
interface. *This claim is architecture-general — it does not, by itself,
depend on the learned-function certification of Claim idea A — and is
therefore the claim idea most likely to run into closer prior art in the
general runtime-assurance space (Simplex-style architectures, ASTM F3269,
Boeing/Collins and NASA runtime-assurance frameworks, and the Dexterity
uncertainty-triggered handoff family, US 10,824,142 B2 / US 12,045,052 B2,
flagged in `invention_idea.md`). This is a specific, named item for the FTO
search to check, not a conclusion this outline draws.*

**Claim idea E — method/system, partial and per-mode certification with a
refusal floor.** A method in which a certification procedure that cannot
certify an entire declared domain within a computational budget certifies a
sub-region instead — never a weaker numerical claim over the full region —
and refuses to produce any certificate at all if the certified sub-region
falls below a declared minimum fraction of the requested domain; extended to
a system in which several such certificates, one per declared operating
mode, are unioned to cover a multi-mode system, with a gate that abstains
whenever the current state is outside every certified mode's cover.

---

## 5. Dependent claim ideas (brief)

- Per-output-dimension ε as a vector rather than a single scalar max.
- K-step tube certification combined with a discrete-Grönwall comparison
  bound, taking the tighter of the interval-tube and Grönwall results.
- Partial certification with a configurable minimum-cover-fraction refusal
  threshold (e.g., a default such as 90% of the requested domain).
- Per-mode certification for hybrid/multi-mode dynamics (e.g., distinct
  contact modes), each certified separately and unioned.
- Cover-membership gating: a runtime check that abstains, rather than
  silently trusting an out-of-domain score, when the current state leaves
  the certified cover.
- A refinement step (e.g., a CROWN-style linear-bound tightening) applied
  only to the worst surviving sub-regions after a cheaper first pass,
  taking the elementwise tighter of two independently sound bounds.
- A startup self-test of the outward-rounding arithmetic substrate that
  must pass before any certification is permitted to proceed.
- Binding the certificate to a hash of the exact deployed network weights,
  checked again at witness construction/load time, not only at
  certification time.
- Binding the certificate to a reference-model identity string covering
  every parameter (e.g., stiffness, time step, geometry) that affects the
  reference's behavior, refusing on any mismatch.
- A single-use certificate structure (nonce, epoch counter, keyed
  authentication tag) preventing forgery, replay, and use across a stale
  control epoch.

---

## 6. Suggested figures

1. **Lemma-chain flow diagram** — the enclosure lemma feeding the one-step
   discrepancy bound, feeding the tube-propagation lemma, feeding the
   sound-silence/miss-floor lemma, with the certified quantity and its
   soundness precondition labeled at each stage.
2. **Certificate binding structure** — a diagram of what a certificate
   binds (weight hash, reference identity/parameters, cover geometry,
   per-dimension ε) and how the authentication tag covers all of it, so
   that altering any one field is shown to invalidate the whole.
3. **K-step tube propagation** — the interval tube widening over
   `t = 0..K` against the operating-domain/cover boundary, showing the
   certified-horizon shrinkage when the tube would otherwise exit the
   cover before the declared horizon.
4. **Non-bypassable gate structure** — a single-entry-point diagram showing
   the gate as the only path to actuation, the certificate authority as a
   separate holder of the signing key, and the fail-closed behavior on any
   exception or binding failure.

---

## 7. Filing timing note

Per SPEC.md section 12, the recommended timing is: file the provisional
**after M5** — the first fully certified composition including the
predictive (K-step) witness, which per the project's milestone record is
already complete in this repository — and **before any public write-up**,
i.e., before the accompanying technical note (being drafted in parallel as
`TECHNICAL_NOTE.md`) is published or shared outside the project, if it will
be shared at all. Provisional filing before public disclosure preserves the
broadest set of downstream options; this outline does not evaluate whether
any particular disclosure has already occurred or what grace period, if
any, might apply — that is itself a question for the attorney, not this
document.

**Repeating the precondition stated at the top of this file:** this outline
is an engineering draft for attorney review, not legal advice, and it makes
no claim about patentability, validity, or freedom-to-operate. A
professional FTO search over G05B/G06N and the named runtime-assurance
dockets (Boeing/Collins, NASA, NVIDIA, TRI), together with a full attorney
review of the claim ideas above (especially Claim idea D, flagged in section
4 as architecture-general and closer to existing runtime-assurance and
uncertainty-triggered-handoff prior art), is a precondition to filing, not a
nice-to-have.
