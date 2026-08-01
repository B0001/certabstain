"""M1 acceptance tests for network bounds.

Spec criteria covered here:
  * 1,000 random MLPs (up to 4 affine layers, width <= 64, input dim <= 8),
    10^5 sampled inputs in total across their boxes: zero enclosure
    violations for IBP and for CROWN(final).
  * CROWN strictly tighter than IBP on >= 90% of boxes; median width ratio
    reported in the assertion message.
  * Jacobian enclosures contain analytic gradients (strictly) and central
    finite differences (with FD-error slack) at all samples.
  * ONNX export is the same function we bounded (onnxruntime probe).
  * VNNLIB artifacts are well-formed with outward-padded bounds.
The alpha-beta-CROWN run itself is out-of-band by design (spec section 3);
tests here validate the artifacts it consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from certabstain import (
    EnclosureError,
    NonFiniteEnclosure,
    Interval,
    MLP,
    crown_bounds,
    ibp_bounds,
    jacobian_bounds,
)

RNG = np.random.default_rng(20260727)


def _random_net_and_box(rng, activation=None):
    activation = activation or ("tanh" if rng.uniform() < 0.3 else "relu")
    n_in = int(rng.integers(1, 9))
    depth = int(rng.integers(1, 4))  # 1-3 hidden layers -> 2-4 affine layers
    sizes = (
        n_in,
        *(int(rng.integers(4, 65)) for _ in range(depth)),
        int(rng.integers(1, 5)),
    )
    net = MLP.random(sizes, activation=activation, rng=rng)
    centre = rng.normal(size=n_in) * 0.5
    radius = 10.0 ** rng.uniform(-2.0, -0.3)
    box = Interval(centre - radius, centre + radius)
    return net, box


def _sample_points(box: Interval, n: int, rng) -> np.ndarray:
    t = rng.uniform(size=(n, box.lo.shape[0]))
    t[rng.uniform(size=t.shape) < 0.05] = 0.0
    t[rng.uniform(size=t.shape) < 0.05] = 1.0
    pts = box.lo + t * (box.hi - box.lo)
    return np.clip(pts, box.lo, box.hi)


# ===================================================================== #
# Soundness fuzz + tightness metric (the headline acceptance run)
# ===================================================================== #


def test_thousand_net_soundness_and_tightness() -> None:
    n_nets, pts_per_net = 1000, 100  # 10^5 sampled inputs total
    rng = np.random.default_rng(31)
    ibp_viol = crown_viol = 0
    tighter = 0
    ratios = []
    worst = None

    for _ in range(n_nets):
        net, box = _random_net_and_box(rng)
        final, det = crown_bounds(
            net, box, experimental=True, return_details=True
        )
        ibp = det["ibp"]
        pure = det["crown"]

        pts = _sample_points(box, pts_per_net, rng)
        ys = net.forward(pts)
        ibp_ok = (ibp.lo[None, :] <= ys) & (ys <= ibp.hi[None, :])
        fin_ok = (final.lo[None, :] <= ys) & (ys <= final.hi[None, :])
        ibp_viol += int(ibp_ok.size - np.count_nonzero(ibp_ok))
        bad = int(fin_ok.size - np.count_nonzero(fin_ok))
        crown_viol += bad
        if bad and worst is None:
            i, j = np.argwhere(~fin_ok)[0]
            worst = (ys[i, j], final.lo[j], final.hi[j])

        w_pure = float(np.sum(pure.width()))
        w_ibp = float(np.sum(ibp.width()))
        if w_pure < w_ibp:
            tighter += 1
        if w_ibp > 0:
            ratios.append(w_pure / w_ibp)

    assert ibp_viol == 0, f"IBP containment violations: {ibp_viol}"
    assert crown_viol == 0, (
        f"CROWN containment violations: {crown_viol}; first: value "
        f"{worst[0]!r} escaped [{worst[1]!r}, {worst[2]!r}]"
    )
    frac = tighter / n_nets
    med = float(np.median(ratios))
    assert frac >= 0.90, (
        f"CROWN strictly tighter than IBP on only {frac:.1%} of boxes "
        f"(median width ratio {med:.3f}); spec requires >= 90%"
    )
    # Surface the metrics on success too (visible with pytest -s).
    print(
        f"\n[M1 metrics] CROWN tighter on {frac:.1%} of {n_nets} boxes; "
        f"median CROWN/IBP width ratio {med:.3f}; "
        f"violations ibp={ibp_viol} crown={crown_viol} over "
        f"{n_nets * pts_per_net} sampled inputs"
    )


def test_final_never_looser_than_ibp() -> None:
    rng = np.random.default_rng(37)
    for _ in range(100):
        net, box = _random_net_and_box(rng)
        final, det = crown_bounds(
            net, box, experimental=True, return_details=True
        )
        ibp = det["ibp"]
        assert np.all(final.lo >= ibp.lo) and np.all(final.hi <= ibp.hi)


def test_point_box_is_tight_and_contains_forward() -> None:
    rng = np.random.default_rng(41)
    for _ in range(100):
        net, box_ignored = _random_net_and_box(rng)
        x = rng.normal(size=net.n_inputs) * 0.5
        pbox = Interval.point(x)
        final = crown_bounds(net, pbox, experimental=True)
        y = net.forward(x)
        assert bool(np.all(final.contains(y))), "forward value escaped"
        slack = np.max(final.width() / (1.0 + np.abs(y)))
        assert slack < 1e-8, f"point-box enclosure suspiciously wide: {slack}"


# ===================================================================== #
# Jacobians
# ===================================================================== #


def _analytic_grad(net: MLP, x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    masks = []
    for k, (W, b) in enumerate(net.weights):
        zhat = z @ W.T + b
        if k < len(net.weights) - 1:
            if net.activation == "relu":
                masks.append((zhat > 0.0).astype(np.float64))
                z = np.maximum(zhat, 0.0)
            else:
                masks.append(1.0 - np.tanh(zhat) ** 2)
                z = np.tanh(zhat)
    J = net.weights[-1][0].copy()
    for k in range(len(masks) - 1, -1, -1):
        J = (J * masks[k][None, :]) @ net.weights[k][0]
    return J


def test_jacobian_contains_analytic_and_fd_gradients() -> None:
    rng = np.random.default_rng(43)
    for _ in range(150):
        net, _ = _random_net_and_box(rng)
        x0 = rng.normal(size=net.n_inputs) * 0.3
        box = Interval(x0 - 1e-3, x0 + 1e-3)
        J = jacobian_bounds(net, box, experimental=True)
        for _ in range(10):
            pt = x0 + rng.uniform(-1e-3, 1e-3, size=net.n_inputs)
            g = _analytic_grad(net, pt)
            assert bool(np.all(J.contains(g))), "analytic gradient escaped"
            # central finite differences, with slack for the O(h^2) FD error
            h = 1e-6
            fd = np.zeros_like(g)
            for j in range(net.n_inputs):
                e = np.zeros(net.n_inputs)
                e[j] = h
                fd[:, j] = (net.forward(pt + e) - net.forward(pt - e)) / (2 * h)
            pad = 1e-4 * (1.0 + np.abs(fd))
            inside = (J.lo - pad <= fd) & (fd <= J.hi + pad)
            assert bool(np.all(inside)), "finite-difference gradient escaped"


# ===================================================================== #
# Flags and refusals
# ===================================================================== #


def test_tanh_requires_experimental_flag() -> None:
    net = MLP.random((3, 8, 2), activation="tanh", rng=np.random.default_rng(1))
    box = Interval(-np.ones(3) * 0.1, np.ones(3) * 0.1)
    for fn in (ibp_bounds, crown_bounds, jacobian_bounds):
        with pytest.raises(ValueError, match="experimental"):
            fn(net, box)
        fn(net, box, experimental=True)  # acknowledged -> proceeds


def test_network_construction_refusals() -> None:
    with pytest.raises(NonFiniteEnclosure, match="non-finite"):
        MLP(((np.array([[np.nan]]), np.zeros(1)),))
    with pytest.raises(ValueError, match="input width"):
        MLP(
            (
                (np.zeros((4, 3)), np.zeros(4)),
                (np.zeros((2, 5)), np.zeros(2)),  # 5 != 4
            )
        )
    net = MLP.random((3, 8, 2), rng=np.random.default_rng(2))
    with pytest.raises(EnclosureError, match="shape"):
        ibp_bounds(net, Interval(np.zeros(4), np.ones(4)))


# ===================================================================== #
# ONNX / VNNLIB artifacts
# ===================================================================== #


def test_onnx_export_is_the_same_function(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from certabstain.vnnlib import export_onnx, onnx_forward_max_diff

    rng = np.random.default_rng(47)
    worst64 = worst32 = 0.0
    for i in range(5):
        net, _ = _random_net_and_box(rng)
        p64 = export_onnx(net, tmp_path / f"n{i}_f64.onnx")
        p32 = export_onnx(net, tmp_path / f"n{i}_f32.onnx", dtype="float32")
        worst64 = max(worst64, onnx_forward_max_diff(net, p64))
        worst32 = max(worst32, onnx_forward_max_diff(net, p32))
    assert worst64 < 1e-9, f"float64 ONNX diverges from forward: {worst64}"
    assert worst32 < 1e-3, f"float32 courtesy copy too far off: {worst32}"
    print(f"\n[M1 metrics] ONNX max forward diff: f64={worst64:.2e} f32={worst32:.2e}")


def test_vnnlib_property_is_wellformed_and_padded(tmp_path) -> None:
    pytest.importorskip("onnx")
    from certabstain.vnnlib import export_vnnlib

    net = MLP.random((3, 16, 2), rng=np.random.default_rng(53))
    box = Interval(-0.2 * np.ones(3), 0.2 * np.ones(3))
    bounds = crown_bounds(net, box)
    path = export_vnnlib(box, bounds, tmp_path / "p.vnnlib")
    text = path.read_text()

    assert text.count("declare-const X_") == 3
    assert text.count("declare-const Y_") == 2
    for i in range(3):
        assert f"(assert (>= X_{i} {box.lo[i]!r}))" in text
    # padded outward: the UB constants in the file exceed our hi strictly
    ub_line = [l for l in text.splitlines() if l.startswith("(assert (or")][0]
    for i in range(2):
        assert repr(float(bounds.hi[i])) not in ub_line or float(
            bounds.hi[i]
        ) == 0.0, "bounds must be padded, not copied verbatim"


def test_artifact_set_generation(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from certabstain.vnnlib import generate_artifact_set

    manifest = generate_artifact_set(tmp_path, n=6, seed=7)
    assert len(manifest["instances"]) == 6
    csv = (tmp_path / "instances.csv").read_text().strip().splitlines()
    assert len(csv) == 6
    for row in csv:
        onnx_rel, vnnlib_rel, timeout = row.split(",")
        assert (tmp_path / onnx_rel).exists()
        assert (tmp_path / vnnlib_rel).exists()
        assert timeout == "300"
    assert (tmp_path / "bounds.json").exists()
    assert (tmp_path / "RUN.md").exists()
    for inst in manifest["instances"]:
        assert inst["f64_forward_max_diff"] < 1e-9
        assert np.all(
            np.asarray(inst["crown_lo"]) >= np.asarray(inst["ibp_lo"]) - 1e-12
        )
