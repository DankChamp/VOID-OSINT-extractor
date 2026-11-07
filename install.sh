#!/usr/bin/env bash
# install.sh — Install reel-vault from local files so the ``OSINT`` command works.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$REPO_ROOT/tools/reel-vault"

echo "╔══════════════════════════════════════════╗"
echo "║  reel-vault installer                   ║"
echo "╚═════════════════════════════════════════╝"
echo
echo "Installing from: $TOOL_DIR"
echo

# Find Python
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: python3 not found. Please install Python 3.11+ first."
    exit 1
fi

# Create venv if needed
VENV="$REPO_ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtual environment at $VENV ..."
    "$PY" -m venv "$VENV"
fi

# Install dependencies
echo "Installing dependencies (this may take a minute)..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --editable "$TOOL_DIR"

echo
echo "╔══════════════════════════════════════════╗"
echo "║  Installation complete!                 ║"
echo "╚═════════════════════════════════════════╝"
echo
echo "To run the OSINT TUI:"
echo "  source $VENV/bin/activate"
echo "  OSINT"
echo
echo "Or run directly:"
echo "  $VENV/bin/OSINT"
echo
