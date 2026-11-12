#!/usr/bin/env bash
# Demo helpers for the video. Run from the project folder.
#   bash demo/demo.sh clear      empty the dashboard (live captures stay in runtime_cleared/)
#   bash demo/demo.sh restore    load the pre-recorded run of the 5 demo boards into the dashboard
#   bash demo/demo.sh live-one   run the line live on one board (missing screw: done in 0.1 s)
#   bash demo/demo.sh live-all   clear everything and run the 5 demo boards live (~7 min with Gemma)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=(.venv/bin/python run.py --part cardboard_board --presence models/presence_cardboard.json)
case "${1:-}" in
  clear)    rm -rf runtime_cleared && { [ -d runtime ] && mv runtime runtime_cleared || true; } && mkdir -p runtime/captures && echo "dashboard is empty" ;;
  restore)  rm -rf runtime && cp -r demo/prerun_runtime runtime && echo "dashboard now shows the pre-recorded run" ;;
  live-one) "${RUN[@]}" --source demo/live ;;
  live-all) rm -rf runtime && "${RUN[@]}" --source demo/boards ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
