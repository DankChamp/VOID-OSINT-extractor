#!/usr/bin/env bash
# install.sh — Install reel-vault from local files so the ``OSINT`` command is available.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$REPO_ROOT/tools/reel-vault"

echo "==> reel-vault installer"
echo "==> Installing from: $TOOL_DIR"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "==> ERROR: python3 not found" >&2
    exit 1
fi

# Create a venv if one doesn't exist at the repo root
VENV="$REPO_ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "==> Creating virtual environment at $VENV"
    "$PY" -m venv "$VENV"
fi

# Upgrade pip and install the package in editable mode
echo "==> Installing dependencies (this may take a minute)"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --editable "$TOOL_DIR"

echo
echo "==> Installation complete."
echo "==> To use: activate the venv, then type:  OSINT"
echo "==>   source $VENV/bin/activate"
echo "==>   OSINT"
echo
echo "==> Or run directly without activating:"
echo "==>   $VENV/bin/OSINT"
