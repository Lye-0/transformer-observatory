#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -x backend/.venv/bin/python || { echo '先にbash scripts/setup.shを実行してください。'; exit 1; }
test -f frontend/dist/index.html || { echo 'UIが未ビルドです。bash scripts/setup.shを実行してください。'; exit 1; }
exec backend/.venv/bin/python -m transformer_observatory "$@"
