#!/usr/bin/env bash
# Drain the beads queue, one containerised worker per bead.
#
# Each bead gets a FRESH container and a fresh context window -- that is the
# point. A single -p session trying to do the whole phase runs out of context
# mid-task; one bead per session does not.
#
# Notes on the docker invocation, all of them learned the hard way:
#   * The mounted host .venv symlinks into /Users/.../miniconda and is dead in
#     the container, so uv is pointed at /tmp/venv instead.
#   * The uv volumes keep the interpreter + wheel downloads (~3 min cold) from
#     repeating on every single worker.
#   * Do NOT add a volume for /home/node/.claude. The config claude reads is
#     /home/node/.claude.json, which sits OUTSIDE that directory; persisting
#     only the directory leaves a stale .claude/backups/ next to a missing
#     config and every run after the first dies with "Claude configuration
#     file not found". Durable output is sandbox-handoffs/, not session state.

set -uo pipefail

PROMPT_FILE="sandbox-prompt.md"
HANDOFF_DIR="sandbox-handoffs"
LOCK_DIR=".sandbox.lock"
MAX_ATTEMPTS=2      # same bead comes back unclosed this many times -> stop
MAX_WORKERS=25      # backstop against a runaway queue

cd "$(dirname "$0")" || exit 1
[ -f "$PROMPT_FILE" ] || { echo "missing $PROMPT_FILE"; exit 1; }
mkdir -p "$HANDOFF_DIR"

# Two concurrent loops would re-dispatch each other's in-progress beads, so
# take an exclusive lock. mkdir is atomic; a stale dir after a hard kill is
# removed by hand, and the message says so.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another sandbox.sh appears to be running."
  echo "if it is not, remove the stale lock: rmdir $LOCK_DIR"
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

run_worker() {
  docker run -it --rm \
    -e CLAUDE_CODE_OAUTH_TOKEN="${CLAUDE_CODE_OAUTH_TOKEN}" \
    -e ANTHROPIC_BASE_URL="http://host.docker.internal:4000" \
    --add-host=host.docker.internal:host-gateway \
    -e ANTHROPIC_SMALL_FAST_MODEL="local-ollama-fast" \
    -v "$(pwd)":/workspace \
    -e UV_PROJECT_ENVIRONMENT=/tmp/venv \
    -e BEADS_ACTOR=sandbox \
    -v claude-uv-cache:/home/node/.cache/uv \
    -v claude-uv-python:/home/node/.local/share/uv/python \
    claude \
    -p "$1" --dangerously-skip-permissions
}

ids_by_status() {
  bd list --status="$1" --json 2>/dev/null \
    | python3 -c 'import json,sys
try: print(" ".join(i["id"] for i in json.load(sys.stdin)))
except Exception: pass'
}

next_ready() {
  bd ready --json 2>/dev/null \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)[0]["id"])
except Exception: pass'
}

# bd ready EXCLUDES in_progress beads. A worker that claims one and then dies,
# stalls, or hits a limit leaves it invisible to the queue forever -- so
# "nothing ready" is not the same as "nothing left". This is the fallback that
# makes the difference visible, and it is what gives MAX_ATTEMPTS something to
# count: a stale claim that keeps failing now comes back instead of vanishing.
# A bead labelled `human` is waiting on a decision no worker is allowed to
# make, so it never closes -- and sitting at the head of the stale list it
# starves every other stale claim until MAX_ATTEMPTS aborts the run. Skip it.
next_stale() {
  bd list --status=in_progress --json 2>/dev/null \
    | python3 -c 'import json,sys
try: print(next(i["id"] for i in json.load(sys.stdin) if "human" not in (i.get("labels") or [])))
except Exception: pass'
}

# Empty queue on the first pass means the phase has not been triaged yet, not
# that the work is done. Seed it: one worker that files beads and writes no
# code. This mirrors the split that worked by accident before -- a triage run
# whose output became the next run's queue.
if [ -z "$(next_ready)" ] && [ -z "$(next_stale)" ]; then
  echo "==> queue empty; running triage pass to file beads"
  run_worker "$(cat "$PROMPT_FILE")

---

# YOUR TASK THIS SESSION: triage only

Do NOT write or modify any code, test, or document this session. Your entire
job is to turn the objectives above into a work queue.

File one bead per discrete, independently-completable unit of work with
\`bd create\`, and use \`bd dep add\` where one genuinely blocks another. Each
bead's description must carry enough detail that a fresh session with no
memory of this one can execute it from \`bd show\` alone: what is wrong, how
you confirmed it, and what evidence would close it. Reproduce before you
file -- a bead asserting a problem you did not actually observe wastes a whole
worker session.

Leave every bead open and unclaimed. Then stop and report the list."
  echo "==> triage complete"
fi

last_id=""
attempts=0
workers=0

while :; do
  TASK_ID="$(next_ready)"

  if [ -z "$TASK_ID" ]; then
    TASK_ID="$(next_stale)"
    if [ -n "$TASK_ID" ]; then
      echo "==> nothing ready; re-dispatching stale claim: $TASK_ID"
    fi
  fi

  if [ -z "$TASK_ID" ]; then
    echo "==> queue drained after $workers worker(s)"
    break
  fi

  # Livelock guard, now load-bearing: with the stale-claim fallback above, a
  # bead that a worker keeps failing to finish WILL come back, so cap it.
  if [ "$TASK_ID" = "$last_id" ]; then
    attempts=$((attempts + 1))
  else
    attempts=1
    last_id="$TASK_ID"
  fi
  if [ "$attempts" -gt "$MAX_ATTEMPTS" ]; then
    echo "==> ABORT: $TASK_ID came back unfinished $MAX_ATTEMPTS times."
    echo "    Inspect with: bd show $TASK_ID"
    echo "    Its handoff (if any): $HANDOFF_DIR/$TASK_ID.md"
    exit 1
  fi

  workers=$((workers + 1))
  if [ "$workers" -gt "$MAX_WORKERS" ]; then
    echo "==> ABORT: hit MAX_WORKERS=$MAX_WORKERS. Queue is growing, not draining."
    exit 1
  fi

  echo "==> worker $workers: $TASK_ID (attempt $attempts)"

  run_worker "$(cat "$PROMPT_FILE")

---

# YOUR TASK THIS SESSION: bead $TASK_ID

Run \`bd show $TASK_ID\` first -- it is the specification, and it was written
by a previous session that reproduced the problem. Everything above is
standing context for this repo; the objectives section is background, not your
assignment. Do only this bead.

If it is already marked in_progress, a previous worker claimed it and did not
finish. Read its notes, do not assume its partial work is correct, and check
the working tree for what it left behind before continuing.

Claim it with \`bd update $TASK_ID --claim\` before you start.

Close it with \`bd close $TASK_ID\` ONLY when the evidence the bead asks for
exists and \`uv run --group dev pytest -q\` is green. If you cannot finish it,
leave it open, say why in \`bd update $TASK_ID --notes=...\`, and stop -- do not
close a bead to make the queue move. If the bead turns out to be wrong or
already fixed, close it with \`--reason\` explaining that, which is a real
outcome and not a failure.

If you discover work outside this bead's scope, file it as a new bead. Do not
do it now.

Write your handoff to \`$HANDOFF_DIR/$TASK_ID.md\`."

  status=$?
  [ "$status" -ne 0 ] && echo "==> worker exited $status"
done

stranded="$(ids_by_status in_progress)"
open_left="$(ids_by_status open)"

echo
echo "Handoffs:   $HANDOFF_DIR/"
echo "Open beads: $(printf '%s' "$open_left" | wc -w | tr -d ' ')"
echo "Nothing was committed or pushed. Review with: git status && git diff"

# "Queue drained" must never be reported while beads sit claimed-but-unclosed.
if [ -n "$stranded" ]; then
  echo
  echo "WARNING: bead(s) left in_progress -- claimed but never closed:"
  for id in $stranded; do echo "    $id"; done
  echo "These are NOT done. Inspect with: bd show <id>"
  exit 1
fi
