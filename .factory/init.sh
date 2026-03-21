#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/roach/.openclaw/workspace/memory-decay"
VENV_DIR="$PROJECT_DIR/dashboard/.venv"

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --quiet dash plotly pandas pytest

# Create dashboard directory if not exists
mkdir -p "$PROJECT_DIR/dashboard"
mkdir -p "$PROJECT_DIR/tests"

# Create __init__.py for tests
touch "$PROJECT_DIR/tests/__init__.py"

# Create dashboard __init__.py
touch "$PROJECT_DIR/dashboard/__init__.py"

echo "Environment setup complete."
