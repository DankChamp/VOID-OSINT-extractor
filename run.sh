#!/usr/bin/env bash
# video-intel launcher for POSIX shells.
#
#   ./run.sh                     set up, then show status
#   ./run.sh doctor              environment report for this device
#   ./run.sh setup               create the venv and install dependencies
#   ./run.sh test [args]         run the test suite (offline by default)
#   ./run.sh lint                ruff + mypy
#   ./run.sh init                create the project layout and state database
#   ./run.sh analyze URL [...]   run the CLI
#   ./run.sh web [--port 8765]   start the local web frontend
#   ./run.sh notebooklm export ID | import FILE --video ID
#   ./run.sh checkpoint          fold the WAL into state.db before copying data/
#   ./run.sh ollama | pull       start ollama / pull a model
#   ./run.sh help                this list
#
# This is a thin wrapper. All of the real work lives in run.py, which is plain Python and
# therefore behaves identically on Windows, macOS and Linux; this file exists so that
# "./run.sh" keeps working for people who expect it. On Windows, use `python run.py`.
#
# It deliberately avoids `set -o pipefail` (where an early-exiting `head` turns a
# working command into a broken one), `python3`, `curl`, and a hardcoded /tmp.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd -P)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  printf '%s\n' "video-intel: no python 3 interpreter found on PATH" >&2
  printf '%s\n' "install Python 3.11 or newer, then re-run this script" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/run.py" "$@"