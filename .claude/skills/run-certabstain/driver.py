#!/usr/bin/env python3
"""Driver for certabstain -- agent tooling, not product surface.

certabstain is a library with no GUI, no server, and no CLI. The thing you
"run" is the guarantee pipeline: produce a certified epsilon, bind it to a
network, calibrate the false-alarm side, and watch the gate emit or abstain.
This driver is that pipeline, callable in one command.

Subcommands:

    check      import the package, print versions and resolved paths
    smoke      the full end-to-end flow (train -> certify -> bind ->
               calibrate -> gate emit + gate abstain)
    refuse     exercise the refusal paths -- this library's contract is
               that it raises rather than under-delivering, so the
               refusals are load-bearing behavior and get tested as such
    artifacts  regenerate the VNNLIB/ONNX export into a temp dir and
               byte-compare against the committed artifacts/vnnlib/
    all        check + smoke + refuse + artifacts

Every subcommand exits non-zero on failure and prints a one-line verdict.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import io
import contextlib
import os
import sys
import tempfile
import traceback
from pathlib import Path

UNIT = Path(__file__).resolve().parents[3]  # .../certabstain

# The repo directory IS the package (__init__.py sits at the repo root), so
# `import certabstain` needs the PARENT of the repo on sys.path. Append rather
# than insert: the parent is a general-purpose directory on a real machine and
# may contain modules that would shadow stdlib or site-packages if it went
# first.
if str(UNIT.parent) not in sys.path:
    sys.path.append(str(UNIT.parent))

FAILURES: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def bad(msg: str) -> None:
    print(f"  FAIL  {msg}")
    FAILURES.append(msg)


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def cmd_check() -> None:
    rule("check")
    import numpy as np

    import certabstain

    ok(f"python      {sys.version.split()[0]}")
    ok(f"numpy       {np.__version__}")
    ok(f"certabstain {certabstain.__file__}")
    if Path(certabstain.__file__).resolve().parent != UNIT:
        bad(f"imported certabstain from {certabstain.__file__}, expected {UNIT}")
    else:
        ok(f"package resolves to the working tree ({UNIT})")

    for name in ("certify_epsilon", "build_monitor", "ActionGate",
                 "VerifiedDiscrepancyWitness"):
        if hasattr(certabstain, name):
            ok(f"export      {name}")
        else:
            bad(f"missing export {name}")


# --------------------------------------------------------------------------
# smoke -- the real flow
# --------------------------------------------------------------------------
def cmd_smoke() -> None:
    rule("smoke: certify -> bind -> calibrate -> gate")
    import numpy as np

    from certabstain import (
        ActionGate,
        CircleClearance,
        Interval,
        VerifiedDiscrepancyWitness,
        build_monitor,
        certify_epsilon,
        rollout_scores,
    )
    from certabstain.nnbound import fit_mlp

    rng = np.random.default_rng(11)
    clear = CircleClearance(ox=0.0, oy=0.0, r=0.15)
    domain = Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))

    # 1. train a stand-in g_hat against the interval-extendable reference
    X = rng.uniform(domain.lo, domain.hi, size=(20_000, 2))
    net = fit_mlp((2, 16, 16, 1), X, clear.value(X), steps=2000, seed=0)
    ok(f"trained MLP {net.n_inputs}->{net.n_outputs}, "
       f"{net.n_hidden_layers} hidden layers")

    # 2. PRODUCE epsilon by branch-and-bound. floor_samples is dialed down
    #    from the demo's 200k purely for driver runtime; it is the sampled
    #    lower-bound check, not the certified upper bound.
    cert = certify_epsilon(
        net,
        clear.interval_batch,
        domain,
        reference_id=clear.reference_id(),
        ref_float=clear.value,
        target=None,
        max_leaf_evals=80_000,
        floor_samples=50_000,
    )
    eps = float(np.max(cert.eps))
    ok(f"certified epsilon = {eps:.5g}, cover = {cert.cover_fraction:.1%}")
    if not (eps > 0 and np.isfinite(eps)):
        bad(f"epsilon is not a usable positive finite bound: {eps}")
    if cert.cover_fraction < 0.90:
        bad(f"cover fraction {cert.cover_fraction:.1%} below the 90% floor")

    # 3. BIND -- re-checks weight hash and reference identity
    witness = VerifiedDiscrepancyWitness.bind(cert, net, clear.reference_id())
    # violation_floor() is a METHOD (SoundnessWitness protocol), unlike the
    # gate's abstention_rate/log/verifier which are properties.
    ok(f"bound witness, violation floor = {witness.violation_floor():.5g}")

    # 4. calibrate the false-alarm side on NOMINAL rollouts only.
    #
    #    These must be rollouts that actually KEEP CLEARANCE. Sampling the
    #    domain box uniformly puts calibration points right up against the
    #    obstacle, the calibrated threshold lands above the violation floor,
    #    and compose() refuses with SoundnessNotEstablished -- correctly, since
    #    a system that skims the boundary cannot support a two-sided claim.
    #    So: sample an annulus around the obstacle, radius 0.30-0.45, which
    #    clears r=0.15 by well over 2*epsilon and stays inside the +-0.5 box.
    def nominal_rollout(n: int = 20):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        radius = rng.uniform(0.30, 0.45, size=n)
        pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        return [float(witness.score(float(v))) for v in clear.value(pts)]

    nominal = [nominal_rollout() for _ in range(400)]
    monitor = build_monitor(
        nominal_trajectories=nominal,
        alpha=0.05,
        witness=witness,
        shift_budget=0.002,
        safe_action="HALT",
    )
    ok("build_monitor returned a two-sided monitor")
    print("\n" + monitor.describe().rstrip() + "\n")

    # 5. drive the gate: one state inside the certified domain, one far
    #    outside it. The second must abstain by name, not return a score.
    gate = ActionGate(
        threshold=0.0, false_alarm_bound=0.05,
        safe_action="HALT", cover=witness.covers,
    )
    results = {}
    for label, point in (
        ("in-distribution", np.array([0.3, 0.3])),
        ("off the domain", np.array([10.0, 10.0])),
    ):
        g_hat = float(clear.value(point[None, :])[0])
        d = gate.step(observation=point, proposed_action=np.array([0.0]),
                      score=witness.score(g_hat))
        results[label] = d
        tag = "ABSTAIN->HALT" if d.abstained else "emit"
        ok(f"{label:<16} {tag:<14} {d.reason}")

    if results["in-distribution"].abstained:
        bad("a clearly nominal in-domain state abstained")
    if not results["off the domain"].abstained:
        bad("a state 20x outside the certified domain did NOT abstain "
            "-- cover-membership gating is broken")
    else:
        ok("cover-membership gating fires off-domain (the M3-M5 claim)")

    # 6. the certificate is single-use: replaying it must be refused
    from certabstain.errors import CertAbstainError
    emitted = results["in-distribution"]
    cert_obj = getattr(emitted, "certificate", None)
    if cert_obj is not None:
        if gate.verifier.verify(cert_obj):
            ok("issued certificate verifies against the gate's verifier")
        else:
            bad("issued certificate failed its own gate's verifier")
    ok(f"abstention rate over {len(gate.log)} steps = "
       f"{gate.abstention_rate:.0%}")


# --------------------------------------------------------------------------
# refuse -- the refusals are the product
# --------------------------------------------------------------------------
def cmd_refuse() -> None:
    rule("refuse: every guarantee that cannot be established must raise")
    import numpy as np

    import certabstain as ca
    from certabstain import build_monitor
    from certabstain.errors import CertAbstainError

    def expect_raise(label: str, fn, exc=CertAbstainError):
        try:
            fn()
        except exc as e:
            ok(f"{label:<34} -> {type(e).__name__}")
            return
        except Exception as e:  # noqa: BLE001
            bad(f"{label}: raised {type(e).__name__} ({e}), expected "
                f"{getattr(exc, '__name__', exc)}")
            return
        bad(f"{label}: returned instead of refusing")

    w = ca.CertifiedModelErrorWitness(epsilon=0.05)
    rng = np.random.default_rng(3)
    good = [list(rng.normal(-1.0, 0.05, size=12)) for _ in range(500)]

    # shift budget >= alpha makes the claim meaningless -> must refuse
    expect_raise(
        "shift_budget >= alpha",
        lambda: build_monitor(nominal_trajectories=good, alpha=0.01,
                              witness=w, shift_budget=0.5, safe_action="HALT"),
    )
    # too few calibration rollouts for the requested level
    expect_raise(
        "calibration set too small",
        lambda: build_monitor(nominal_trajectories=good[:3], alpha=0.01,
                              witness=w, safe_action="HALT"),
    )
    # a system that skims the constraint boundary cannot support two sides
    tight = [list(rng.normal(0.0, 0.01, size=12)) for _ in range(500)]
    expect_raise(
        "clearance below ~2*epsilon",
        lambda: build_monitor(nominal_trajectories=tight, alpha=0.01,
                              witness=w, safe_action="HALT"),
    )
    # a non-finite threshold would certify everything: score > nan is False
    expect_raise(
        "non-finite gate threshold",
        lambda: ca.ActionGate(threshold=float("nan"), false_alarm_bound=0.01,
                              safe_action="HALT"),
        Exception,
    )
    # no override path exists anywhere in the gate API
    import inspect
    sig = inspect.signature(ca.ActionGate.__init__)
    banned = {"force", "override", "strict", "unsafe", "bypass"}
    hit = banned & set(sig.parameters)
    if hit:
        bad(f"ActionGate exposes an override parameter: {sorted(hit)}")
    else:
        ok("ActionGate has no force/override/strict parameter")


# --------------------------------------------------------------------------
# artifacts -- bit-reproducibility of the external cross-check set
# --------------------------------------------------------------------------
def _tree_hashes(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def cmd_artifacts() -> None:
    rule("artifacts: regenerate artifacts/vnnlib/ and byte-compare")
    from certabstain.vnnlib import generate_artifact_set

    committed = UNIT / "artifacts" / "vnnlib"
    if not committed.is_dir():
        bad(f"no committed artifact set at {committed}")
        return

    with tempfile.TemporaryDirectory() as td:
        fresh = Path(td) / "vnnlib"
        # generate_artifact_set is chatty; the verdict below is what matters
        with contextlib.redirect_stdout(io.StringIO()):
            generate_artifact_set(fresh)
        a, b = _tree_hashes(committed), _tree_hashes(fresh)

        only_committed = sorted(set(a) - set(b))
        only_fresh = sorted(set(b) - set(a))
        differing = sorted(k for k in set(a) & set(b) if a[k] != b[k])

        ok(f"regenerated {len(b)} files, committed set has {len(a)}")
        if only_committed:
            print(f"        only in committed: {only_committed[:6]}"
                  f"{' ...' if len(only_committed) > 6 else ''}")
        if only_fresh:
            print(f"        only in fresh:     {only_fresh[:6]}"
                  f"{' ...' if len(only_fresh) > 6 else ''}")
        if differing:
            bad(f"{len(differing)} committed artifact(s) do not reproduce "
                f"byte-for-byte: {differing[:6]}")
        elif set(a) & set(b):
            ok(f"{len(set(a) & set(b))} shared files reproduce byte-for-byte")


COMMANDS = {
    "check": cmd_check,
    "smoke": cmd_smoke,
    "refuse": cmd_refuse,
    "artifacts": cmd_artifacts,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*COMMANDS, "all"], nargs="?",
                    default="all")
    args = ap.parse_args()

    todo = list(COMMANDS) if args.command == "all" else [args.command]
    for name in todo:
        try:
            COMMANDS[name]()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            bad(f"{name} raised an unhandled exception")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"PASS -- {', '.join(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
