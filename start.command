#!/bin/bash
set -e
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT="300"

./scripts/bootstrap/bootstrap.command "$@"
uv run --no-sync python launcher.py
