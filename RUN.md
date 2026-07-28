# Re-verifying certabstain's M1 bounds with alpha-beta-CROWN

Every instance pairs a float64 ONNX model with a VNNLIB property asserting
the NEGATION of certabstain's certified output bounds (padded outward by two
ulp steps). Expected verdict on every instance: **unsat / safe** -- meaning
the reference verifier finds no point in the input box whose output escapes
our bounds. A single **sat** verdict is a counterexample to our soundness and
a release blocker; please report it with the instance name.

    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
    cd alpha-beta-CROWN
    # follow the repo's setup instructions (conda env), then per instance:
    python complete_verifier/abcrown.py \
        --onnx_path <this dir>/models/net_XX_f64.onnx \
        --vnnlib_path <this dir>/props/net_XX.vnnlib \
        --timeout 300

`instances.csv` lists all pairs in abcrown's CSV format (onnx, vnnlib,
timeout). `bounds.json` records the exact IBP and CROWN bounds we computed,
the network shapes, and the max forward discrepancy of the float32 courtesy
copies (`*_f32.onnx`) against the authoritative float64 models. Compare
bounds against float64. Instances marked `"activation": "tanh"` used the
experimental parallel-slope relaxation.

Nothing in this directory depends on certabstain code.
