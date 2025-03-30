#!/bin/bash
# Run all tests for the Bayesian Self-Adaptive LLM project

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_EXE="$PROJECT_ROOT/venv/bin/python"

echo "Running tests with Python: $PYTHON_EXE"
echo "Project root: $PROJECT_ROOT"
echo

# Set up the virtual environment if it doesn't exist
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv "$PROJECT_ROOT/venv"
    source "$PROJECT_ROOT/venv/bin/activate"
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo "Virtual environment set up successfully."
    echo
fi

# Run tests
echo "===== Running Bayesian Controller Tests ====="
"$PYTHON_EXE" "$PROJECT_ROOT/scripts/test_bayesian_controller.py"
echo

echo "===== Running MambaFormer Tests ====="
"$PYTHON_EXE" "$PROJECT_ROOT/scripts/test_mambaformer.py"
echo

# Run a simple inference test
echo "===== Running Basic Inference Test ====="
"$PYTHON_EXE" "$PROJECT_ROOT/demo.py" --prompt "Test prompt for basic functionality"
echo

echo "All tests completed successfully!"
