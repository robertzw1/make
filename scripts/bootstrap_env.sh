#!/usr/bin/env bash
# Bootstrap a virtualenv for this project even when `python3-venv` is
# missing from the host (e.g. a minimal Docker image).
#
#   scripts/bootstrap_env.sh [venv_dir]
#
# The script is idempotent: it will skip work that has already been done.
set -euo pipefail

VENV_DIR="${1:-.venv}"
PY_BIN="${PYTHON:-python3}"

if [[ -x "$VENV_DIR/bin/pip" ]]; then
    echo "[bootstrap] $VENV_DIR already has pip — nothing to do"
    exit 0
fi

echo "[bootstrap] python: $("$PY_BIN" -V)"

# --- 1. Create the venv ----------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[bootstrap] creating $VENV_DIR"
    if ! "$PY_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "[bootstrap] stdlib venv failed (likely missing python3-venv); retrying without pip"
        rm -rf "$VENV_DIR"
        "$PY_BIN" -m venv --without-pip "$VENV_DIR"
    fi
fi

# --- 2. Ensure pip is present ---------------------------------------------
if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
    PIP_PYZ="${PIP_PYZ:-/tmp/pip.pyz}"
    if [[ ! -f "$PIP_PYZ" ]]; then
        echo "[bootstrap] downloading pip.pyz from pypa.io"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -4 https://bootstrap.pypa.io/pip/pip.pyz -o "$PIP_PYZ"
        elif command -v wget >/dev/null 2>&1; then
            wget -q -4 https://bootstrap.pypa.io/pip/pip.pyz -O "$PIP_PYZ"
        else
            echo "[bootstrap] need curl or wget to fetch pip.pyz" >&2
            exit 1
        fi
    fi
    "$VENV_DIR/bin/python" "$PIP_PYZ" install --quiet --upgrade pip setuptools wheel
fi

"$VENV_DIR/bin/pip" --version
