#!/usr/bin/env bash
# Re-exec under bash if running under /bin/sh (e.g. UCloud init)
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# ── Parse KEY=VALUE arguments from UCloud "Extra options" field ────────────
for arg in "$@"; do
  case "${arg}" in
    BRANCH=*)     BRANCH="${arg#BRANCH=}" ;;
    HF_TOKEN=*)   HF_TOKEN="${arg#HF_TOKEN=}" ;;
    GH_PAT=*)     GH_PAT="${arg#GH_PAT=}" ;;
  esac
done

REPO_URL="https://github.com/kwa6/dst-unified"
REPO_DIR="dst-unified"
BRANCH="${BRANCH:-mergetest}"

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo "=========================================="
echo "uCloud bootstrap starting  (branch: ${BRANCH})"
echo "=========================================="

# Clone or update repo
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning repository: $REPO_URL (branch: ${BRANCH})"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "Repository already exists — pulling latest"
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

cd "$REPO_DIR"

echo
echo "=========================================="
echo "Now in repo: $(pwd)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "=========================================="

# Export HF_TOKEN so pipeline.sh can log in and run LLM evals
export HF_TOKEN="${HF_TOKEN:-}"
export GH_PAT="${GH_PAT:-}"

# Starting pipeline
run_step bash scripts/setup_env.sh
run_step bash scripts/pipeline.sh

# ── Optional: push log back to GitHub ─────────────────────────────────────
if [ -n "${GH_PAT:-}" ]; then
  LOG_BRANCH="logs/ucloud-$(date +%Y%m%d-%H%M%S)"
  git config user.email "ucloud@noreply"
  git config user.name  "UCloud Runner"
  git checkout -b "$LOG_BRANCH"
  git add -A
  git commit -m "UCloud run results $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
  git push "https://kwa6:${GH_PAT}@github.com/kwa6/dst-unified.git" "$LOG_BRANCH"
  echo "Logs pushed to branch: $LOG_BRANCH"
fi

