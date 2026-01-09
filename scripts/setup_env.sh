#!/usr/bin/env bash

# Simple bootstrap script to create .venv and install requirements.

set -euo pipefail

VENV_DIR=".venv"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is not available on PATH."
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r requirements.txt

echo "Virtual environment ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
