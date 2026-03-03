#!/usr/bin/env python
"""Simple wrapper to run streamlit with proper error handling."""
import subprocess
import sys

cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
print(f"Running: {' '.join(cmd)}", file=sys.stderr, flush=True)
subprocess.run(cmd, check=False)
