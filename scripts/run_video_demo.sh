#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_video_demo.sh <video_uri_or_path>"
  exit 1
fi

python3 src/main.py --config config/config.yaml --mode video --input-uri "$1"
