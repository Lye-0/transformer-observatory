#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo 'uvが必要です。'; exit 1; }
command -v mise >/dev/null || { echo 'miseが必要です。'; exit 1; }
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_PYTHON_BIN_DIR="$PWD/.tools/bin"
export UV_CACHE_DIR="$PWD/.tools/uv-cache"
uv python install 3.13.15
uv sync --project backend --python 3.13.15 --locked
mise trust
mise install
mise exec -- pnpm --dir frontend install --frozen-lockfile
mise exec -- pnpm --dir frontend run build
echo '準備ができました。bash scripts/start.sh で起動できます。'
