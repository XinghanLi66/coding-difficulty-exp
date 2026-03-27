#!/bin/bash
# Periodically push dashboard_data.json to GitHub so Pages stays live.
# Run in background: nohup bash scripts/push_dashboard.sh > logs/push_dashboard.log 2>&1 &

PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
INTERVAL=300  # push every 5 minutes

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$PROJECT_DIR" || exit 1

while true; do
    # Refresh data
    $PYTHON scripts/export_data.py 2>/dev/null

    # Only push if dashboard_data.json actually changed
    if ! git diff --quiet dashboard_data.json; then
        git add dashboard_data.json
        git commit -m "dashboard: update $(date '+%H:%M:%S') — $(python3 -c "
import json
d=json.load(open('dashboard_data.json'))
print(f\"{d['completed']}/{d['total_runs']} done\")
" 2>/dev/null || echo 'in progress')"
        if git push origin main 2>&1; then
            log "Pushed dashboard_data.json to GitHub"
        else
            log "Push failed (will retry in ${INTERVAL}s)"
        fi
    else
        log "No change, skipping push"
    fi

    sleep "$INTERVAL"
done
