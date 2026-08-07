You are working autonomously in the certabstain repo. Make aggressive, real
progress. Do not stop to ask permission; do not stop early because you are
unsure whether there is work left — this prompt tells you exactly where it is.

## The standard everything is held to

This repo is evidence for a patent handoff. `PROVISIONAL_OUTLINE.md` goes to a
registered patent attorney, who will click through every claim in `README.md`,
`SPEC.md`, `TECHNICAL_NOTE.md` and that outline and ask: *where is the code
that does this, and where is the artifact that shows it ran?*

**A claim is only allowed to exist if a named test or a regenerable artifact
backs it.** When a claim outruns the code you have two honest moves —
implement the code, or narrow the claim. There is never a third. Do not delete
a claim silently, do not soften it into vagueness, and do not mark anything
"verified" on the strength of your own reasoning. Reasoning is not evidence
here; a passing test with a name is.

**The existing hedges are load-bearing.** Several caveats read like
over-hedging and are not — an audit already caught the un-hedged versions
being false. Do not touch: the gate's "no override path exists" claim, which
is deliberately scoped to the **API surface** and explicitly **not** memory
isolation; the α,β-CROWN result's "one verifier, one machine, one run"
caveat; and the two residual gaps recorded in `TECHNICAL_NOTE.md` §6 (the
conformal calibrators' unverifiable order statistic, and the API-surface scope
of the `__post_init__`/sentinel guards). If you think a hedge is excessive,
leave it and say why in your handoff. Removing a true hedge is the worst
possible outcome of this session.

## The prior phase is finished — do not redo it

A previous session fixed every caller-revocable array freeze and every
factory-bypassable dataclass invariant, swept all four docs, and closed seven
beads. The suite is green at 145. `bd list --status=closed` shows that work.
Do not re-audit it and do not invent findings to look busy — if you genuinely
find nothing in a section below, say so and move to the next one.

One habit from that phase still applies, because it is a property of this
codebase: **guarantees here have a way of being enforced on one path and not
its sibling.** Before calling anything done, grep the other callers of the
same contract, and check the documented entry point rather than only the
class.

## Target 1 — artifact provenance (the main work)

`artifacts/` holds the numbers the attorney will actually look at:
`pushing_stick_report.json`, `pushing_slide_left_report.json`,
`pushing_slide_right_report.json`, `pushing_conservatism_report.json`,
`scaling_study.json`, `stiffness_sweep.json`, `tube_sweep.json`. They record
`eps`, `cover_fraction`, `verdict` and similar — and **no provenance at all**:
no commit, no Python/numpy version, no BLAS, no platform, no date.

This matters because **the numbers are not environment-independent.** The three
`pushing_*` reports were measured to differ by roughly 0.7% relative on `eps`
(e.g. `0.0010009` vs `0.0010076`) and to move `abstention_rate` from 0.0016 to
0.0018 between a macOS/conda-numpy host and this Linux container. So a reviewer
who re-runs gets different numbers than the committed ones and **has no way to
tell benign environment variance from a real regression.** That is the gap.

Note carefully: no document currently claims these are bit-reproducible, so
nothing in the repo is false today. You are strengthening thin evidence, not
correcting a lie. Do not introduce a reproducibility claim you cannot
demonstrate.

Three constraints that decide whether this is done well:

1. **Do not destroy the diff signal.** Today, a dirty `git status` on
   `artifacts/` after a run means a value really changed. If you stamp a
   timestamp or a hostname straight into the payload, every run dirties every
   file and that signal is gone forever. Separate the deterministic payload
   from the environment record — a sibling file, a segregated block excluded
   from comparison, a hash of the payload alone; your call, but state the
   design and why in `TECHNICAL_NOTE.md`.
2. **You have exactly two environments to reason from**, and one of them you
   cannot inspect: this container, and whatever produced the committed
   numbers. n=2, and the second is a black box. Any tolerance you propose must
   be reported as *observed across these two environments*, not as a
   guaranteed bound. If you write a test that asserts agreement within a
   tolerance, the tolerance's justification goes in the test's docstring, and
   "I picked a round number that passed" is not a justification — say so
   plainly if that is what happened.
3. **Do not regenerate and commit the existing artifacts.** They are the
   current evidence, produced on a machine you do not have. Overwriting them
   with container numbers destroys evidence and silently rebases every doc
   claim onto an unreviewed environment. The provenance of the *existing*
   files is unrecoverable — handle that honestly (label it unknown, or
   recommend a regeneration the human runs on the reference machine) and flag
   the decision in your handoff rather than making it yourself.

Writers to cover: `test_pushing_stick.py`, `test_pushing_slide_left.py`,
`test_pushing_slide_right.py`, `pushing_conservatism_report.py`,
`scaling_study.py`, `stiffness_sweep.py`, `tube_sweep.py`, and `vnnlib.py`.
Sibling-path rule applies — a provenance helper used by six of eight writers
is not done.

## Target 2 — the fixable residual gap

`TECHNICAL_NOTE.md` §6 records that `SplitConformalCalibrator.__post_init__`
and `MondrianCalibrator.__post_init__` cannot verify `threshold` is a genuine
order statistic, because `fit()` checks it against a raw scores array that is
never stored on the built object. The prior session correctly declined to
store the whole sample.

Storing a **digest** of the calibration sample is the obvious middle path:
enough to let a re-check confirm the threshold came from a real `fit()`,
without retaining the data. Evaluate it honestly. It may not work — a digest
proves the sample existed, not that the threshold is its k-th order statistic,
unless you also store something order-related. If it does not close the gap,
say exactly what it does and does not prove, and leave the gap recorded rather
than declaring a partial fix complete. A half-closed gap described as closed is
worse than the gap.

## Environment

- `uv run --group dev pytest -q` runs the suite (~3 min). Green before done.
  It is 145 tests now; if that number changes, update it in `README.md`, which
  states it in two places.
- `bd` is the tracker. File a bead per work item with `bd create` before you
  write code, `bd update <id> --claim`, and close only when the evidence
  exists. Prior beads are closed; `bd ready` being empty means file new ones,
  not that you are done.
- Do not chase `m7-audit-hardening` or `m7-claims-audit`; both are long merged.

## The artifacts trap — different this time

In the prior phase a dirty `artifacts/` was noise to ignore. **This session it
is your subject matter**, so the rule inverts: you must know precisely why each
file is dirty at every point. Before you finish, `git diff artifacts/` and
account for every changed file — payload change, provenance addition, or
environment variance. An unexplained diff in `artifacts/` is a failed session.
Also delete any `*.vnnlib.compiled` scratch files that appear under
`artifacts/vnnlib/props/`.

## Git policy

Do the work, get the suite green, leave the tree **ready to commit**. Do not
`git commit`, do not `git push`, do not `bd dolt push`. Put the exact commands
you would run in the handoff and let a human run them.

## What to hand back

**Write this to `sandbox-handoff.md` in the repo root before you finish, and
print it as your final message.** The file is the part that survives; this
container is disposable. Overwrite what is there.

A report a reviewer can check, not a summary of effort:

- The provenance design, and specifically how it preserves the "dirty
  artifacts means a real change" signal.
- The measured cross-environment deltas, per artifact, with the numbers.
- Any tolerance you introduced, its justification, and its honest epistemic
  status given n=2.
- Target 2's verdict: closed, partially closed with exactly what is and is not
  proven, or not closed.
- Every doc claim you changed, before and after, with the backing test.
- The final `pytest -q` line verbatim.
- `git diff --stat artifacts/` with a one-line reason per file.
- What you decided **not** to do, and why. Empty means you did not look hard.
- What you could not verify. An honest "unverified" beats a confident claim —
  the attorney will check.
