"""ONNX + VNNLIB export for out-of-band cross-checking (M1, spec section 8).

The certified path never imports a verification stack; instead, every network
and every bound we emit can be exported to the standard formats and handed to
alpha-beta-CROWN (the VNN-COMP reference verifier) for independent
re-verification. The artifacts are self-contained: models, properties, an
instances CSV in abcrown format, our bounds in JSON, and a RUN file with the
exact commands. A third party can re-verify without running our code.

Property encoding: each .vnnlib asserts the input box and the NEGATION of
"our (padded) bounds hold" -- the disjunction that some output escapes the
padded bounds. A verifier result of "unsat"/"safe" on an instance therefore
means no contradiction with our bounds; "sat" would be a counterexample and a
release blocker. Bounds are padded outward by two ulp steps so that boundary
semantics (strict vs. inclusive) cannot manufacture a spurious counterexample.

The authoritative models are float64. Some toolchains prefer float32, so a
float32 copy is written alongside; the JSON records the max forward
discrepancy between the two on random probes, and bounds should be compared
against the float64 model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .interval import Interval, _down, _up
from .nnbound import MLP, crown_bounds, ibp_bounds

__all__ = [
    "export_onnx",
    "export_vnnlib",
    "generate_artifact_set",
    "onnx_forward_max_diff",
]


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #


def export_onnx(net: MLP, path: str | Path, *, dtype: str = "float64") -> Path:
    """Write the MLP as an ONNX graph: Gemm (+ Relu/Tanh) chain, batch dim 1."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    proto_dtype = TensorProto.DOUBLE if dtype == "float64" else TensorProto.FLOAT
    np_dtype = np.float64 if dtype == "float64" else np.float32

    nodes, initializers = [], []
    prev = "X"
    n_layers = len(net.weights)
    for k, (W, b) in enumerate(net.weights):
        wname, bname = f"W{k}", f"b{k}"
        initializers.append(
            numpy_helper.from_array(W.astype(np_dtype), name=wname)
        )
        initializers.append(
            numpy_helper.from_array(b.astype(np_dtype), name=bname)
        )
        gemm_out = "Y" if k == n_layers - 1 else f"z{k}"
        nodes.append(
            helper.make_node(
                "Gemm", [prev, wname, bname], [gemm_out],
                name=f"gemm{k}", alpha=1.0, beta=1.0, transB=1,
            )
        )
        if k < n_layers - 1:
            act = "Relu" if net.activation == "relu" else "Tanh"
            nodes.append(
                helper.make_node(act, [gemm_out], [f"a{k}"], name=f"act{k}")
            )
            prev = f"a{k}"

    graph = helper.make_graph(
        nodes,
        "certabstain_mlp",
        [helper.make_tensor_value_info("X", proto_dtype, [1, net.n_inputs])],
        [helper.make_tensor_value_info("Y", proto_dtype, [1, net.n_outputs])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def onnx_forward_max_diff(
    net: MLP, onnx_path: str | Path, n_probes: int = 100, seed: int = 0
) -> float:
    """Max |onnxruntime(model) - net.forward| over random probes.

    Validates that the exported graph is the same function we bounded. Runs
    with whatever dtype the model declares.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    in_dtype = np.float64 if "double" in sess.get_inputs()[0].type else np.float32
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_probes):
        x = rng.normal(size=(1, net.n_inputs)).astype(in_dtype)
        y_ort = sess.run(None, {"X": x})[0].astype(np.float64)
        y_ref = net.forward(x.astype(np.float64))
        worst = max(worst, float(np.max(np.abs(y_ort - y_ref))))
    return worst


# --------------------------------------------------------------------------- #
# VNNLIB
# --------------------------------------------------------------------------- #


def _numeral(v: Any) -> str:
    """Format one value as a legal VNNLIB decimal numeral.

    ``repr`` of a numpy scalar is not a numeral: under NumPy >= 2 it round-trips
    as ``np.float64(0.5)``, which no VNNLIB grammar accepts. Every literal in an
    exported instance goes through here, via ``float``, so the text stays a
    numeral whatever array scalar type reaches it. ``repr(float(...))`` is the
    shortest string that round-trips exactly, so no precision is lost against
    the padded bound.

    A non-finite value would format as ``inf``/``nan`` -- also not numerals, and
    an instance carrying one asserts nothing. Refused here rather than written
    out, since a file that parses but means nothing is worse than no file.
    """
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(
            f"cannot export {f!r} as a VNNLIB numeral: an instance bound must "
            "be finite, or the property it encodes is vacuous"
        )
    return repr(f)


def export_vnnlib(
    box: Interval,
    out_bounds: Interval,
    path: str | Path,
    *,
    pad_steps: int = 2,
) -> Path:
    """Write the negated-property VNNLIB file for one instance."""
    n_in = box.lo.shape[0]
    n_out = out_bounds.lo.shape[0]
    ub = _up(np.asarray(out_bounds.hi, dtype=np.float64), pad_steps)
    lb = _down(np.asarray(out_bounds.lo, dtype=np.float64), pad_steps)

    lines = [
        "; certabstain M1 cross-check instance",
        "; property below is the NEGATION of our claim: unsat/safe = no",
        "; contradiction with our certified bounds (padded outward by "
        f"{pad_steps} ulp steps); sat = counterexample, release blocker.",
        "",
    ]
    for i in range(n_in):
        lines.append(f"(declare-const X_{i} Real)")
    for i in range(n_out):
        lines.append(f"(declare-const Y_{i} Real)")
    lines.append("")
    for i in range(n_in):
        lines.append(f"(assert (>= X_{i} {_numeral(box.lo[i])}))")
        lines.append(f"(assert (<= X_{i} {_numeral(box.hi[i])}))")
    lines.append("")
    escapes = []
    for i in range(n_out):
        escapes.append(f"(and (>= Y_{i} {_numeral(ub[i])}))")
        escapes.append(f"(and (<= Y_{i} {_numeral(lb[i])}))")
    lines.append("(assert (or " + " ".join(escapes) + "))")
    lines.append("")

    path = Path(path)
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# The artifact set
# --------------------------------------------------------------------------- #

_RUN_MD = """\
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
    uv run python complete_verifier/abcrown.py \\
        --config <path>/certabstain.yaml \\
        --onnx_path <this dir>/models/net_XX_f64.onnx \\
        --vnnlib_path <this dir>/props/net_XX.vnnlib \\
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
"""


def generate_artifact_set(
    outdir: str | Path, n: int = 24, seed: int = 20260726
) -> dict:
    """Write >= n cross-check instances plus manifest. Returns the manifest."""
    outdir = Path(outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "props").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest: dict = {"instances": []}
    csv_rows = []

    for idx in range(n):
        activation = "tanh" if idx % 6 == 5 else "relu"
        n_in = int(rng.integers(2, 9))
        depth = int(rng.integers(1, 4))  # hidden layers
        sizes = (n_in, *(int(rng.integers(8, 65)) for _ in range(depth)), 
                 int(rng.integers(1, 5)))
        net = MLP.random(sizes, activation=activation, rng=rng)
        centre = rng.normal(size=n_in) * 0.5
        radius = 10.0 ** rng.uniform(-1.5, -0.3)
        box = Interval(centre - radius, centre + radius)

        bounds = crown_bounds(net, box, experimental=True)
        ibp = ibp_bounds(net, box, experimental=True)

        stem = f"net_{idx:02d}"
        p64 = export_onnx(net, outdir / "models" / f"{stem}_f64.onnx")
        p32 = export_onnx(
            net, outdir / "models" / f"{stem}_f32.onnx", dtype="float32"
        )
        prop = export_vnnlib(box, bounds, outdir / "props" / f"{stem}.vnnlib")
        csv_rows.append(f"models/{p64.name},props/{prop.name},300")

        manifest["instances"].append(
            {
                "name": stem,
                "activation": activation,
                "sizes": list(sizes),
                "box_lo": box.lo.tolist(),
                "box_hi": box.hi.tolist(),
                "crown_lo": bounds.lo.tolist(),
                "crown_hi": bounds.hi.tolist(),
                "ibp_lo": ibp.lo.tolist(),
                "ibp_hi": ibp.hi.tolist(),
                "f32_forward_max_diff": onnx_forward_max_diff(net, p32),
                "f64_forward_max_diff": onnx_forward_max_diff(net, p64),
            }
        )

    (outdir / "instances.csv").write_text("\n".join(csv_rows) + "\n")
    (outdir / "bounds.json").write_text(json.dumps(manifest, indent=2))
    (outdir / "RUN.md").write_text(_RUN_MD)
    return manifest
