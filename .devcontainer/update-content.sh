#!/usr/bin/env bash
set -euo pipefail

# Ensure pixi is discoverable in non-interactive contexts
export PATH="${HOME}/.pixi/bin:/usr/local/bin:${PATH}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "[update-content] ERROR: 'pixi' not found on PATH."
  echo "Checked: \$HOME/.pixi/bin and /usr/local/bin. Did setup-pixi.sh run?"
  exit 127
fi

echo "[update-content] Re-syncing pixi environment ..."
echo "Installing pixi default environment..."
pixi install
echo "Installing qiskit environment..."
pixi install -e qiskit
echo "Installing squlearn environment..."
pixi install -e squlearn
