---
name: run-certabstain
description: Run, drive, smoke-test, or verify the certabstain library — produce a certified epsilon, calibrate a monitor, drive the action gate, check refusal paths, and confirm artifacts reproduce byte-for-byte. Use when asked to run certabstain, run the demo, reproduce artifacts, run the tests, or confirm a change to the certification/gate/bound code actually works.
---

# Running certabstain

certabstain is a **library** — no GUI, no server, no CLI. The thing you "run"
is the guarantee pipeline: produce a certified `epsilon` by branch-and-bound,
bind it to a network, calibrate the false-alarm side on nominal rollouts, and
watch the gate emit or abstain. `driver.py` is that pipeline in one command.

All paths below are relative to the repo root (`~/Downloads/certabstain`),
which is also the unit.

## Prerequisites

Nothing to install. The project-local venv at `.venv` already has everything
(numpy 2.5.1, pytest, matplotlib, scipy, onnx, onnxruntime, mpmath) on Python
3.12.0. Verified on darwin 25.5.0 (arm64).

**Use `./.venv/bin/python` and nothing else.** Not the system/miniconda python
— see Troubleshooting.

## Run (agent path)

```bash
./.venv/bin/python .claude/skills/run-certabstain/driver.py all
```

~2s. Exits 0 on pass, 1 on any failure, and prints a `PASS`/`FAILED` verdict
line. Works from **any** working directory (it puts the repo's parent on
`sys.path` itself), so no `PYTHONPATH`, no `VIRTUAL_ENV`, no `uv` wrapper.

Subcommands — run one when you only care about one layer:

| Command | What it does |
|---|---|
| `check` | imports the package, asserts it resolved to *this* working tree, prints versions |
| `smoke` | the full flow: train MLP → `certify_epsilon` → `VerifiedDiscrepancyWitness.bind` → `build_monitor` → `ActionGate.step` emitting, then abstaining off-domain |
| `refuse` | asserts the refusal paths still raise — `ShiftBudgetExceeded`, `InsufficientCalibrationData`, `SoundnessNotEstablished`, non-finite threshold, and no `force=`/`override=` on `ActionGate` |
| `artifacts` | regenerates `artifacts/vnnlib/` into a temp dir and byte-compares all 75 files against the committed set |

Expected `smoke` output (epsilon is deterministic at `seed=0`):

```
  ok    certified epsilon = 0.041107, cover = 100.0%
  ok    in-distribution  emit           certified
  ok    off the domain   ABSTAIN->HALT  left certified domain
  ok    cover-membership gating fires off-domain (the M3-M5 claim)
```

`refuse` is the one to run after touching `gate.py`, `soundness.py`, or
`errors.py` — this library's contract is that it *raises* rather than
under-delivering, so the refusals are load-bearing behavior, not error handling.

`artifacts` is the one to run after touching `vnnlib.py`, `discrepancy.py`,
`nnbound.py`, or `interval.py`. It is a genuine regression detector: all 75
files reproduce byte-for-byte, so any diff is a real behavior change.

## Direct invocation

Most changes here touch internals, not a user surface. Import and call
directly — same one-liner shape, from any cwd:

```bash
./.venv/bin/python -c "
import sys; sys.path.append('/Users/benjaminhess/Downloads')
import numpy as np
from certabstain import Interval, CircleClearance, certify_epsilon
from certabstain.nnbound import fit_mlp, ibp_bounds

clear  = CircleClearance(ox=0.0, oy=0.0, r=0.15)
domain = Interval(np.array([-0.5,-0.5]), np.array([0.5,0.5]))
X   = np.random.default_rng(11).uniform(domain.lo, domain.hi, size=(20_000,2))
net = fit_mlp((2,16,16,1), X, clear.value(X), steps=2000, seed=0)
print('IBP:', ibp_bounds(net, domain))
"
```

Verified output:
`IBP: Interval(lo=array([-3.38940512]), hi=array([5.34642649]))`
— note how loose plain IBP is over the whole box; that gap is what the
branch-and-bound in `discrepancy.py` exists to close.

## Test

```bash
VIRTUAL_ENV="$(pwd)/.venv" uv run --active --no-sync pytest -q
```

`127 passed in 141.29s`. Slow — the theorem-validation tests are Monte Carlo
plus adversarial corner sweeps. Run the driver first; it catches most breakage
in 2 seconds.

## Run (human path): demo and sweeps

Bare scripts do **not** self-bootstrap `sys.path` the way the driver does, so
they need `PYTHONPATH` set to the repo's *parent*:

```bash
PYTHONPATH=/Users/benjaminhess/Downloads ./.venv/bin/python demo.py
PYTHONPATH=/Users/benjaminhess/Downloads ./.venv/bin/python stiffness_sweep.py
```

`demo.py` prints the six-section narrative ending in the M3-M5 "silent-void row
closed" table. `stiffness_sweep.py` takes ~1.6s and **overwrites**
`artifacts/stiffness_sweep.{json,png}` — both regenerate byte-identically
(matplotlib PNG included), so a clean `git status` afterward is the expected
result and a diff means something changed.

Same pattern for `tube_sweep.py`, `scaling_study.py`,
`pushing_conservatism_report.py`.

## Gotchas

- **The repo directory *is* the package.** `__init__.py` sits at the repo root,
  so `import certabstain` needs the repo's **parent** (`~/Downloads`) on
  `sys.path`, not the repo itself. `driver.py` handles this; bare scripts don't.
- **That parent is `~/Downloads`** — a general-purpose folder with hundreds of
  unrelated files, including a stale `__pycache__/conftest.*.pyc`. `driver.py`
  therefore **appends** to `sys.path` instead of inserting, so nothing in there
  can shadow stdlib or site-packages. Preserve that if you edit it.
- **`violation_floor()` is a method; `abstention_rate` / `log` / `verifier` are
  properties.** The witness implements a `SoundnessWitness` protocol where the
  floor is a call; the gate exposes properties. Formatting `witness.violation_floor`
  fails with `TypeError: unsupported format string passed to method.__format__`.
- **A naive calibration set gets refused, and that's correct.** Sampling the
  domain box uniformly puts nominal points right against the obstacle; the
  calibrated threshold then lands above the violation floor and `build_monitor`
  raises `SoundnessNotEstablished`. Calibration rollouts must actually keep
  clearance — the driver samples an annulus at radius 0.30–0.45 around an
  `r=0.15` obstacle, clearing it by well over `2*epsilon`. If you see that
  refusal, check your calibration data before you touch `soundness.py`.
- **`floor_samples` changes the reported epsilon.** The driver uses 50k for
  speed and gets `0.041107`; `demo.py` uses 200k and gets `0.03971`. Neither is
  wrong — it's the sampled empirical floor, not the certified upper bound — but
  don't treat a mismatch between the two as a regression.
- **`timeout` does not exist on macOS.** `timeout 180 ...` fails with exit 127,
  which looks exactly like a crash in the script you were wrapping. Use
  `gtimeout` (coreutils) or just don't.
- **Committed artifacts are bit-reproducible**, all 75 vnnlib files and the
  sweep PNGs. This is a strong property worth not breaking — it's what makes
  `driver.py artifacts` a real check rather than a smoke test.
- **`artifacts/vnnlib/` has not been cross-verified with α,β-CROWN yet.** See
  `artifacts/vnnlib/RUN.md`. The driver checks that the export *reproduces*, not
  that the properties are *unsat*.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'certabstain'` running a bare script | `PYTHONPATH=/Users/benjaminhess/Downloads` — the parent, not the repo |
| `ImportError: Error importing numpy: you should not try to import numpy from its source directory` | You used the miniconda python (`/Users/benjaminhess/miniconda/bin/python3`). Use `./.venv/bin/python`. |
| `TypeError: unsupported format string passed to method.__format__` | You formatted `witness.violation_floor` without calling it |
| `SoundnessNotEstablished: calibrated threshold ... does not clear the violation floor` | Calibration rollouts skim the constraint boundary; sample a corridor with real clearance (see Gotchas) |
| `exit=127` from a wrapped command | `timeout` isn't installed on macOS |
| `driver.py artifacts` reports files that "do not reproduce byte-for-byte" | A real behavior change in the export path — diff the named files, don't just regenerate over them |
