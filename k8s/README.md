# Kubernetes deployment

The three sweeps as batch Jobs, sharing one artifacts PVC.

```bash
docker build -t certabstain:0.1.0 .
kubectl apply -k k8s/overlays/dev     # namespace certabstain-dev
kubectl apply -k k8s/overlays/prod    # namespace certabstain-prod
```

| Job | Command | Output |
|---|---|---|
| `certabstain-scaling-study` | `-m certabstain.scaling_study` | `scaling_study.{json,png}` |
| `certabstain-tube-sweep` | `-m certabstain.tube_sweep` | `tube_sweep.{json,png}` |
| `certabstain-stiffness-sweep` | `-m certabstain.stiffness_sweep` | `stiffness_sweep.{json,png}` |

## Run them one at a time

The PVC is `ReadWriteOnce`. The three Jobs write disjoint filenames, so sharing
one claim is safe — but only sequentially. Two Jobs scheduled to *different*
nodes would leave one Pending forever, because an RWO volume attaches to one
node at a time.

## Image layout

This repo **is** the `certabstain` package (`__init__.py` at the root), and the
sweep scripts sit inside it importing `from certabstain import ...`. So the tree
is copied to `/app/certabstain` with `/app` on `PYTHONPATH`. That is why the
Jobs use `python -m certabstain.scaling_study` and not a file path — running the
file directly would not resolve those imports.

Two consequences worth knowing:

- The Dockerfile runs `chmod -R a+rX` after the COPY. Several sources in this
  repo are mode `0600` on disk and `COPY` preserves host modes, so without it
  the container dies with `PermissionError: /app/certabstain/__init__.py` as
  uid 10001 before running any of its own code.
- Sweeps write to `Path(__file__).parent / "artifacts"`, which resolves to
  `/app/certabstain/artifacts` — that exact path is the PVC mount point.

## Dependencies

Runtime is numpy plus matplotlib. matplotlib is from the dev group and is
genuinely optional (every sweep guards its import and degrades to JSON-only),
but it is included so the Jobs emit their plots. `MPLBACKEND=Agg` and
`MPLCONFIGDIR=/tmp/matplotlib` are set because there is no display and the root
filesystem is read-only. scipy/onnx/mpmath are test-only — no runtime import —
and are deliberately omitted.

## QoS note

Prod pods are **Burstable, not Guaranteed**: CPU request equals limit, but
memory request does not. That is deliberate — memory is sized per sweep
(`tube_sweep` carries interval state to `K_MAX` and asks for more than the other
two), and one shared patch cannot equalise memory without flattening that. The
Jobs are restartable (`backoffLimit: 3`), so the eviction risk is an acceptable
trade. For a run that must not be evicted, set memory request == limit on that
one Job.
