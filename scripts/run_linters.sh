#!/usr/bin/env bash
set -euo pipefail

echo "Running flake8..."
flake8 . || true

echo

echo "Running mypy..."
mypy . || true

echo

echo "Linters finished."