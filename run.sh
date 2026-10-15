#!/usr/bin/env bash
# VOID OSINT extractor launcher for POSIX shells.
#
#   ./run.sh                     set up, then list the tools
#   ./run.sh doctor              per-tool environment report
#   ./run.sh setup               create the shared venv and install every tool
#   ./run.sh test [args]         run every tool's test suite
#   ./run.sh lint                ruff + mypy over every tool
#   ./run.sh <tool> <command>    run a tool's CLI, e.g.
#                               ./run.sh video-intel analyze "https://youtu.be/ID" -m quick
#   ./run.sh help                this list
#
# This is a thin wrapper. All of the real work lives in run.py, which is plain Python and
# therefore behaves identically on Windows, macOS and Linux; this file exists so that
# "./run.sh" keeps working for people who expect it. On Windows, use `python run.py`.
#
# It deliberately avoids `set -o pipefail` (where an early-exiting `head` turns a working
# command into a broken one), `python3`, `curl`, and a hardcoded /tmp.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd -P)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  printf '%s\n' "VOID OSINT extractor: no python 3 interpreter found on PATH" >&2
  printf '%s\n' "install Python 3.11 or newer, then re-run this script" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/run.py" "$@"