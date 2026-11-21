#!/usr/bin/env bash
# Paste into the Vast.ai template "On-start Script" field (works with interruptible/spot instances):
#
#   bash /workspace/video/scripts/onstart.sh
#
# After an interruption the instance restarts and this resumes exactly where it stopped
# (every finished shot, line and stage is cached). Progress log: /workspace/vp/logs/run.log
cd /workspace/video || exit 0
mkdir -p /workspace/vp/logs
if [[ -f /workspace/vp/logs/AUTO_RESUME_DISABLED ]]; then exit 0; fi
nohup bash start.sh resume >> /workspace/vp/logs/run.log 2>&1 &
