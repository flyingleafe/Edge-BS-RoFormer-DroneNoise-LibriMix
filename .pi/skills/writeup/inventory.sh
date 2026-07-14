#!/usr/bin/env bash
# Deterministic work catalogue for the /writeup workflow.
# Lists everything that happened since the last artifact of the given kind:
# commits, experiment configs, experiment docs (with excerpts), new
# reports/slides (with headings), code-change summary, untracked candidates.
#
# Usage: inventory.sh --kind slides|report [--since <ref>] [--out <file>]
set -euo pipefail

KIND="" SINCE="" OUT="/dev/stdout"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)  KIND="$2"; shift 2 ;;
    --since) SINCE="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ "$KIND" == "slides" || "$KIND" == "report" ]] || { echo "--kind slides|report required" >&2; exit 1; }

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
KIND_DIR="writing/slides"
[[ "$KIND" == "report" ]] && KIND_DIR="writing/reports"

# ---- boundary: last commit touching the newest artifact of this kind --------
LAST_ARTIFACT=""
if [[ -z "$SINCE" ]]; then
  LAST_ARTIFACT=$(find "$KIND_DIR" -maxdepth 1 -mindepth 1 -type d -name '20*' | sort | tail -1)
  [[ -n "$LAST_ARTIFACT" ]] || { echo "no previous artifact under $KIND_DIR; pass --since <ref>" >&2; exit 1; }
  SINCE=$(git log -1 --format='%H' -- "$LAST_ARTIFACT")
  [[ -n "$SINCE" ]] || { echo "no commits touch $LAST_ARTIFACT; pass --since <ref>" >&2; exit 1; }
fi

exec > "$OUT"

echo "# Work inventory since last $KIND"
echo
echo "- generated: $(date -Iseconds)"
echo "- boundary artifact: ${LAST_ARTIFACT:-"(explicit --since)"}"
echo "- boundary commit: $(git log -1 --format='%h %cs %s' "$SINCE")"
echo "- HEAD: $(git log -1 --format='%h %cs %s' HEAD)"
echo
echo "> Numbers in docs below may predate later fixes — always cross-check the"
echo "> newest report/doc before quoting. Sync results before analysis (Rule 5)."

echo
echo "## Commits (newest first)"
echo
echo '```'
git log --oneline --no-merges "$SINCE"..HEAD
echo '```'

echo
echo "## Experiment configs (conf/experiment/)"
echo
echo '```'
git diff --name-status "$SINCE"..HEAD -- conf/experiment/ | sed 's/^/  /'
echo '```'

echo
echo "## Docs (docs/) — excerpts for added files"
echo
git diff --name-status "$SINCE"..HEAD -- docs/ | while read -r status path _; do
  case "$status" in
    A) echo "### ADDED: $path"
       echo '```'
       git show "HEAD:$path" 2>/dev/null | head -30
       echo '```' ;;
    M) echo "- MODIFIED: $path" ;;
    D) echo "- DELETED: $path" ;;
    R*) echo "- RENAMED: $path -> $_" ;;
  esac
done

echo
echo "## Writing artifacts created/updated in the window"
echo
for wdir in writing/reports writing/slides; do
  git diff --name-only "$SINCE"..HEAD -- "$wdir" | cut -d/ -f1-3 | sort -u | while read -r adir; do
    [[ -d "$adir" ]] || continue
    main_typ=$(find "$adir" -maxdepth 1 -name '*.typ' | head -1)
    echo "### $adir"
    if [[ -n "$main_typ" ]]; then
      echo '```'
      grep -E '^=+ ' "$main_typ" | head -25 || true
      echo '```'
    fi
  done
done

echo
echo "## Code changes (summary)"
echo
echo '```'
git diff --stat "$SINCE"..HEAD -- src/ scripts/ tests/ | tail -3
git diff --name-only "$SINCE"..HEAD -- src/ scripts/ | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -15
echo '```'

echo
echo "## Untracked candidates (not yet committed)"
echo
echo '```'
git status --porcelain -- docs/ writing/ conf/ | grep '^??' | sed 's/^?? /  /' || echo "  (none)"
echo '```'

echo
echo "## Prep notes found (read these fully — often a ready-made narrative seed)"
echo
find "$KIND_DIR" -maxdepth 1 -name 'NEXT-*' -o -maxdepth 1 -name 'TODO*' 2>/dev/null | sed 's/^/- /' || true
