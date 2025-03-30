#!/bin/bash
# Run the Bayesian Self-Adaptive LLM interactive demo

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_EXE="$SCRIPT_DIR/venv/bin/python"

# Set up the virtual environment if it doesn't exist
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "Virtual environment set up successfully."
    echo
fi

# Run the demo
echo "Starting Bayesian Self-Adaptive LLM demo..."
"$PYTHON_EXE" "$SCRIPT_DIR/demo.py" "$@"
