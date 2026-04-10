#!/usr/bin/env python3
"""Standalone Phase 2 grader test"""
import subprocess

print("=== PHASE 2 GRADER TEST ===")
result = subprocess.run(["python", "benchmark.py"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print(f"Exit code: {result.returncode}")

