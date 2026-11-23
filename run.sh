#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8001}"
"$DIR/.venv/bin/python" -m uvicorn app:app --reload --port "$PORT" "$@"

