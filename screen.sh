#!/usr/bin/env sh
# Compact launcher for the keyword screening interface (same as: csv-screen …)
cd "$(dirname "$0")" || exit 1
if command -v csv-screen >/dev/null 2>&1; then
  if [ "$#" -eq 0 ]; then exec csv-screen -h; fi
  exec csv-screen "$@"
fi
if [ "$#" -eq 0 ]; then exec python -m two_stage_screen screen -h; fi
exec python -m two_stage_screen screen "$@"
