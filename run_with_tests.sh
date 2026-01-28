#!/bin/bash
# Bash script to run tests before starting the Flask app (for Linux/macOS)
# Usage: bash run_with_tests.sh

echo "======================================================================"
echo "Activating virtual environment and running test suite..."
echo "======================================================================"

# Activate the virtual environment
source venv/bin/activate

# Run pytest
echo -e "\nRunning pytest..."
python -m pytest -q

# Check if tests passed
if [ $? -ne 0 ]; then
    echo -e "\nTests failed! Application startup aborted."
    exit 1
fi

echo ""
echo "======================================================================"
echo "All tests passed! Starting Flask application..."
echo "======================================================================"
echo ""

# Start the Flask app
python -m spam_detection
