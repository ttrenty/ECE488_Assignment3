#!/usr/bin/env bash
set -euo pipefail

# Ensure our PATH for this setup run
export PATH="${HOME}/.pixi/bin:/usr/local/bin:${PATH}"

# --- Install pixi if missing ---
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
  echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> "$HOME/.bashrc"
fi

# Make 'pixi' discoverable for non-interactive scripts as well
if [ ! -x /usr/local/bin/pixi ]; then
  sudo ln -sf "${HOME}/.pixi/bin/pixi" /usr/local/bin/pixi
fi

# --- Install the assignment env once ---
echo "Installing pixi default environment..."
pixi install
echo "Installing qiskit environment..."
pixi install -e qiskit
echo "Installing squlearn environment..."
pixi install -e squlearn

# --- Register a VS Code terminal hook ---
# Remove any previous block to avoid duplicates
sudo sed -i '/# ASSIGN1_WELCOME_START/,/# ASSIGN1_WELCOME_END/d' /etc/bash.bashrc

# Add a guarded block:
# - Only runs for interactive terminals (-t 1)
# - Only when inside VS Code terminal (TERM_PROGRAM=vscode)
# - Uses WORKSPACE_FOLDER injected via remoteEnv to locate the script
sudo tee -a /etc/bash.bashrc >/dev/null <<'EOS'
# ASSIGN1_WELCOME_START
if [ -t 1 ] && [ "${TERM_PROGRAM:-}" = "vscode" ]; then
  if [ -n "${WORKSPACE_FOLDER:-}" ] && [ -f "${WORKSPACE_FOLDER}/.devcontainer/announce.sh" ]; then
    PATH="$HOME/.pixi/bin:/usr/local/bin:$PATH" \
      bash "${WORKSPACE_FOLDER}/.devcontainer/announce.sh" || true
  fi
fi
# ASSIGN1_WELCOME_END
EOS
