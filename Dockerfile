# syntax=docker/dockerfile:1
#
# Batch image for the certabstain sweeps. Runs as a Kubernetes Job, not a
# service -- each sweep is run-to-completion and writes JSON (+ PNG) artifacts.
#
#   docker build -t certabstain:0.1.0 .
#   docker run --rm -v "$PWD/artifacts:/app/certabstain/artifacts" \
#       certabstain:0.1.0 -m certabstain.scaling_study
#
# Layout note: this repo IS the `certabstain` package (__init__.py at the root),
# and the sweep scripts sit inside it importing `from certabstain import ...`.
# So the tree is copied to /app/certabstain and /app goes on the path -- that is
# what makes `python -m certabstain.scaling_study` resolve. Copying to /app
# directly would put the modules at top level and break every one of those
# imports.

FROM python:3.12-slim AS runtime

# pyproject declares numpy as the only runtime dependency and has no
# [build-system], so there is nothing to `pip install .` -- the package is used
# from the source tree. matplotlib is from the dev group and is genuinely
# optional: every sweep guards its import and degrades to JSON-only output.
# It is included because a Job that emits the plots is more useful than one that
# silently does not. scipy/onnx are test-only (no runtime import) and are omitted.
RUN pip install --no-cache-dir \
        "numpy>=2.2.6" \
        "matplotlib>=3.9"

WORKDIR /app

COPY . /app/certabstain/

# COPY preserves the host's file modes, and several sources in this repo are
# 0600 on disk. Root ignores that; uid 10001 does not, so without this the
# container dies with "PermissionError: /app/certabstain/__init__.py" before
# running a line of its own code. a+rX (capital X) adds the execute bit only to
# directories and files that already had one, so it does not mark modules
# executable.
RUN chmod -R a+rX /app/certabstain

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The sweeps are single-process and numpy-bound. Without this each BLAS call
# fans out to every core the node has, which under a CPU limit means constant
# throttling and thrash rather than speed.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# matplotlib needs a writable config dir and no display. Agg is the headless
# backend; MPLCONFIGDIR must point into the /tmp emptyDir because the root
# filesystem is read-only at runtime (otherwise it warns and falls back on
# every single import).
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Non-root. Kubernetes pins runAsUser/fsGroup to this same 10001 so the
# artifacts PVC is writable; keep the two in sync.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

# Sweeps write to Path(__file__).parent / "artifacts", which resolves to this
# exact path inside the image. This is where the PVC mounts.
RUN mkdir -p /app/certabstain/artifacts && chown 10001:10001 /app/certabstain/artifacts

USER 10001:10001

ENTRYPOINT ["python"]
# Overridden per Job. The scaling study is the default because it is the
# cheapest of the three and doubles as a smoke test.
CMD ["-m", "certabstain.scaling_study"]
