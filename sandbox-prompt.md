You are working autonomously in the certabstain repo. Make aggressive, real
progress. Do not stop to ask permission; do not stop early because you are
unsure whether there is work left — there is, and this prompt tells you how to
find it.

## The standard everything is held to

This repo is evidence for a patent handoff. `PROVISIONAL_OUTLINE.md` goes to a
registered patent attorney, who will click through every claim in `README.md`,
`SPEC.md`, `TECHNICAL_NOTE.md` and that outline and ask: *where is the code
that does this, and where is the artifact that shows it ran?* A claim that
cannot survive that click-through is a liability, not a feature.

So: **a claim is only allowed to exist if a named test or a regenerable
artifact backs it.** When you find a claim that outruns the code, you have two
honest moves — implement the code, or narrow the claim to what is true. You
never have a third. Do not delete the claim silently, do not soften it into
vagueness, and do not mark something "verified" on the strength of your own
reasoning. Reasoning is not evidence here; a passing test with a name is.

**The existing hedges are load-bearing.** Several caveats in these docs read
like over-hedging and are not — they are the difference between a true claim
and a false one, and an audit already caught the false version. Specifically:
the gate's "no override path exists" claim is deliberately scoped to the **API
surface** (no supported call sequence yields an uncertified emission) and
explicitly **not** memory isolation, because in CPython nothing stops
in-process code from touching a private attribute. Keep that distinction
intact everywhere it appears. Likewise, the α,β-CROWN cross-check result is
one verifier, one machine, one run — keep that caveat, it is not an
independent audit. If you think a hedge is excessive, leave it and write down
why in your handoff. Removing a true hedge is the single worst outcome of this
session.

## The bug pattern this codebase actually has

Five of the audit bugs fixed here were the same shape: **a guarantee enforced
on one path and not on its sibling.** `_ref_enclosure` validated its caller's
enclosure; `clearance_lower_bounds`, on the parallel path, did not. `ActionGate`
took a `cover=`; `build_monitor`, the documented one-call entry point, forgot
to pass it — so the class was safe and every real caller was not.

Two consequences, both mandatory:

1. Before you call any fix done, **grep every other caller of the same
   contract** and check whether the sibling path has the same hole. A fix that
   patches only the path you were looking at is not a fix.
2. **Test the documented entry point, not just the class.** A test that
   hand-builds the object bypasses exactly the path real users take, which is
   where the bugs have actually been.

## Start here — two reproduced, unfixed findings

Both were confirmed real by a prior audit and neither has been addressed.
Reproduce each one first and write the reproduction down as a failing test
before you change any source:

1. **`Interval.lo` / `.hi` are caller-revocable.** `interval.py` calls
   `setflags(write=False)` on the endpoint arrays, but the arrays own their
   data, so a caller can do `iv.lo.setflags(write=True)` and mutate the
   endpoints — including into an inverted `lo > hi` state that the constructor
   would have rejected. Note that a read-only *view* whose base is itself
   read-only cannot be re-enabled this way; that is the likely shape of the
   fix, but verify it empirically before committing to it.
2. **Frozen-dataclass constructors bypass `bind()` / `build()` invariants.**
   The `@dataclass(frozen=True)` types in `reference.py` can be constructed
   directly, skipping whatever the blessed factory enforces. Determine what
   each factory actually checks, then decide per type whether the direct
   constructor should validate or be made unreachable. State the reasoning.

When those two are done, keep going: sweep for the sibling-path pattern above
across `gate.py`, `witness2.py`, `discrepancy.py`, `tube.py`, `conformal.py`
and `nnbound.py`, and audit the claim-to-evidence mapping in the four documents
named at the top.

## Environment

- `uv run --group dev pytest -q` runs the suite. It takes about three minutes
  and must be green before you consider anything finished. The interpreter and
  wheels are provisioned by uv; there is no other Python here.
- `bd` is the issue tracker. **The tracker is currently empty** — that is why
  `bd ready` returns nothing, not because the work is done. Before you write
  code, file a bead per finding with `bd create`, claim it with
  `bd update <id> --claim`, and close it only when the evidence exists.
- `git log` note: branches `m7-audit-hardening` and `m7-claims-audit` are
  **already merged into main**. There is nothing outstanding on them. Do not
  go looking for work there.

## The artifacts trap — read this before you `git checkout` anything

Running the test suite **rewrites tracked files** under `artifacts/`
(`pushing_stick_report.json` and the two `pushing_slide_*` reports). Those
JSONs are bit-reproducible *within one environment* but **not across
environments** — the values differ between this container and the host by
roughly 0.7% relative on `eps`, from a different numpy/BLAS build. So:

- A dirty `artifacts/` after a test run **in here** is expected and is not a
  bug you found. Do not commit those diffs, and do not "fix" the code to chase
  the host's numbers.
- Equally, do not commit a change that makes them differ *for a real reason*
  without saying so loudly in your handoff.
- Also delete any `*.vnnlib.compiled` scratch files if they appear under
  `artifacts/vnnlib/props/`.

## Git policy

Stage nothing behind the user's back. Do the work, get the suite green, leave
the tree **ready to commit**, and report what you changed. Do not `git commit`,
do not `git push`, and do not run `bd dolt push`. If you believe a commit is
warranted, put the exact command in your handoff and let a human run it.

## What to hand back

**Write this to `sandbox-handoff.md` in the repo root before you finish, and
print it as your final message too.** The file is the part that survives: this
container is disposable and your terminal output may not be read. Overwrite
whatever is already in that file — the durable record of *findings* is the bead
database, not this report.

Not a summary of your effort — a report a reviewer can check:

- Each finding: the reproduction, the fix, the test that now covers it, and
  the sibling paths you grepped and cleared.
- Every doc claim you changed, with the before and after wording and the
  specific test or artifact that now backs it.
- The final `uv run --group dev pytest -q` line, verbatim, pass count included.
- Anything you concluded was **not** worth fixing, and why. This section being
  empty means you did not look hard enough.
- Anything you could not verify. Say so plainly. An honest "unverified" is
  worth more here than a confident claim, because the attorney will check.
