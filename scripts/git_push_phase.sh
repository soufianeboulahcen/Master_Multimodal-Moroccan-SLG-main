#!/usr/bin/env bash
# git_push_phase.sh — standard git workflow for each completed phase/feature.
#
# Usage:
#   ./scripts/git_push_phase.sh <branch-name> "<commit-message>"
#
# Examples:
#   ./scripts/git_push_phase.sh phase-c-identity "add identity extraction module"
#   ./scripts/git_push_phase.sh controlnet-integration "add ControlNet OpenPose conditioning"
#   ./scripts/git_push_phase.sh inference-optimization "optimize GPU memory and inference speed"
#
# What it does:
#   1. Runs the test suite — aborts if any test fails
#   2. Creates the branch if it doesn't exist, or switches to it
#   3. Stages all modified/new tracked files (git add -u + untracked non-ignored)
#   4. Commits with the provided message
#   5. Pushes to origin
#   6. Prints a summary of what was pushed
#
# Rules enforced:
#   - Never commits if tests fail
#   - Never pushes to main directly
#   - Uses --force-with-lease (not --force) for rebased branches
#   - Aborts cleanly on any error (set -euo pipefail)

set -euo pipefail

# ── Args ─────────────────────────────────────────────────────────────────────
BRANCH="${1:-}"
MESSAGE="${2:-}"

if [[ -z "$BRANCH" || -z "$MESSAGE" ]]; then
    echo "Usage: $0 <branch-name> \"<commit-message>\""
    echo ""
    echo "Branch naming convention:"
    echo "  phase-b-dwpose          phase-c-identity"
    echo "  phase-d-video-diffusion controlnet-integration"
    echo "  instantid-support       inference-optimization"
    exit 1
fi

# ── Safety: never push directly to main ──────────────────────────────────────
if [[ "$BRANCH" == "main" ]]; then
    echo "ERROR: direct push to main is not allowed."
    echo "       Create a feature branch and open a PR."
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "═══════════════════════════════════════════════════════"
echo "  Git workflow: $BRANCH"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: run tests ─────────────────────────────────────────────────────────
echo ""
echo "▶ Step 1/5 — running test suite ..."
if python -m unittest mosl.render.test_render -v 2>&1 | tail -4; then
    echo "  tests passed"
else
    echo "ERROR: tests failed — aborting commit."
    echo "       Fix the failures before pushing."
    exit 1
fi

# ── Step 2: create or switch to branch ───────────────────────────────────────
echo ""
echo "▶ Step 2/5 — branch: $BRANCH"
CURRENT="$(git branch --show-current)"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    if [[ "$CURRENT" != "$BRANCH" ]]; then
        git checkout "$BRANCH"
        echo "  switched to existing branch '$BRANCH'"
    else
        echo "  already on '$BRANCH'"
    fi
else
    git checkout -b "$BRANCH"
    echo "  created new branch '$BRANCH'"
fi

# ── Step 3: stage files ───────────────────────────────────────────────────────
echo ""
echo "▶ Step 3/5 — staging files ..."
git add -A
STAGED="$(git diff --cached --name-only)"
if [[ -z "$STAGED" ]]; then
    echo "  nothing to commit — working tree is clean."
    echo "  (branch is already up to date)"
    exit 0
fi
echo "$STAGED" | sed 's/^/    /'

# ── Step 4: commit ────────────────────────────────────────────────────────────
echo ""
echo "▶ Step 4/5 — committing ..."
git commit -m "$MESSAGE

Co-authored-by: Ona <no-reply@ona.com>"
echo "  committed: $(git log --oneline -1)"

# ── Step 5: push ──────────────────────────────────────────────────────────────
echo ""
echo "▶ Step 5/5 — pushing to origin/$BRANCH ..."
# --force-with-lease is safe for rebased branches; refuses if remote has
# commits we haven't seen (protects against overwriting others' work).
if git push origin "$BRANCH" --force-with-lease 2>&1; then
    echo "  pushed successfully"
else
    echo "  first push (new branch) ..."
    git push --set-upstream origin "$BRANCH"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DONE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Branch : $BRANCH"
echo "  Commit : $(git log --oneline -1)"
echo "  Remote : $(git remote get-url origin)"
echo ""
echo "  Modified files:"
git diff HEAD~1 --name-only 2>/dev/null | sed 's/^/    /' || echo "    (first commit on branch)"
echo ""
echo "  Next step: open a PR at"
REPO_URL="$(git remote get-url origin | sed 's/\.git$//')"
echo "    $REPO_URL/pull/new/$BRANCH"
echo ""
