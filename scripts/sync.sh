#!/bin/bash
# Keep github.com/topprismdata/duckbench-rir in sync with the local workspace.
set -u
cd /Users/guohongbin/1-Projects/duckbench-rir || exit 1
while :; do
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git -c user.name="topprismdata" -c user.email="topprismdata@users.noreply.github.com" \
        commit -q -m "auto-sync $(date -u +%Y-%m-%dT%H:%MZ)" || true
    git push -q origin main || true
  fi
  sleep 600
done
