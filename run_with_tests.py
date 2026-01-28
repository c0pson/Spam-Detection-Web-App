#!/usr/bin/env python
"""
Pre-startup test runner and app launcher.
Runs the full test suite before starting the Flask development server.
Exit with error code if tests fail.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run pytest and return exit code."""
    print("=" * 70)
    print("Running test suite before startup...")
    print("=" * 70)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=Path(__file__).parent)
    return result.returncode

def start_app():
    """Start the Flask development server."""
    print("\n" + "=" * 70)
    print("All tests passed! Starting Flask application...")
    print("=" * 70 + "\n")
    from spam_detection.main import create_app
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    test_exit_code = run_tests()
    if test_exit_code != 0:
        print("\n" + "=" * 70)
        print("Tests failed! Application startup aborted.")
        print("=" * 70)
        sys.exit(test_exit_code)
    start_app()
