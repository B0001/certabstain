# Re-verifying certabstain's M1 bounds with alpha-beta-CROWN

Every instance pairs a float64 ONNX model with a VNNLIB property asserting
the NEGATION of certabstain's certified output bounds (padded outward by two
ulp steps). Expected verdict on every instance: **unsat / safe** -- meaning
the reference verifier finds no point in the input box whose output escapes
our bounds. A single **sat** verdict is a counterexample to our soundness and
a release blocker; please report it with the instance name -- but read the
net_15 note below first, because one instance returns a spurious `sat` under
the verifier's default float32 settings.

**Status: this has now been run.** On 2026-08-03, alpha-beta-CROWN 0.7.0
(torch 2.11.0, CPU, `double_fp: true`) returned **unsat on all 24 instances**.
The same sweep at the float32 default returns unsat on 23 and a spurious `sat`
on net_15. Independently, evaluating net_15's f64 ONNX at 60,005 points in the
box via onnxruntime found no violation, with a tightest margin of `3.886e-15`.
This is a single run of a single verifier on one machine and does not make the
bounds independently audited -- reproduce it rather than taking this line for it.

**Run it in float64.** This is not optional. Write a config file:

    # certabstain.yaml
    general:
      device: cpu
      conv_mode: matrix
      double_fp: true      # <-- required; see the net_15 note below
    solver:
      batch_size: 512

    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
    cd alpha-beta-CROWN
    uv sync --python 3.11          # abcrown 0.7.0 pins torch==2.11.0
    # then per instance (--config is mandatory: abcrown resolves paths
    # relative to it and crashes with a TypeError if it is omitted):
    uv run python complete_verifier/abcrown.py \
        --config <path>/certabstain.yaml \
        --onnx_path <this dir>/models/net_XX_f64.onnx \
        --vnnlib_path <this dir>/props/net_XX.vnnlib \
        --timeout 300

Note that abcrown writes a `<name>.vnnlib.compiled` cache next to each
property file. Those are its scratch, not ours; delete them afterwards or a
reproducibility check on this directory will look dirty.

**net_15 returns a spurious `sat` at float32, and this is expected.**
abcrown defaults to `double_fp: false`. At float32 the PGD stage reports an
attack margin of exactly `0.00000000` on net_15 and stops with
`verified_status unsafe-pgd` -- while in the same breath printing
`Total number of violation: 0`. That is a tie at the boundary, not an escape.
The instance's true f64 margin is `3.886e-15`, about 1.5e8 times smaller than
the f32 forward error of these models (`5.704e-07`, recorded per instance in
`bounds.json`), so at float32 the margin is simply not representable. With
`double_fp: true` net_15 verifies `unsat` like the rest. A `sat` here is a
precision artifact of the verifier, not an unsound bound -- re-check in f64
before reporting one.

`instances.csv` lists all pairs in abcrown's CSV format (onnx, vnnlib,
timeout). `bounds.json` records the exact IBP and CROWN bounds we computed,
the network shapes, and the max forward discrepancy of the float32 courtesy
copies (`*_f32.onnx`) against the authoritative float64 models. Compare
bounds against float64. Instances marked `"activation": "tanh"` used the
experimental parallel-slope relaxation.

Nothing in this directory depends on certabstain code.
