#!/bin/bash

# 1) Make sure you're on main and up to date
git checkout main
git fetch origin
git pull --ff-only

# 2) Find the root commit (your current initial commit)
ROOT=$(git rev-list --max-parents=0 HEAD)

# 3) Reset index to "just after root", but keep working tree as-is
#    (so all current files become staged as one big change)
git reset --soft "$ROOT"

# 4) Create the single commit you want
git commit --amend -m "Initial Commit"
# If you prefer a brand new commit instead of amending the root:
# git commit -m "Initial Commit"

# 5) Force push rewritten history to GitHub
git push --force-with-lease origin main
