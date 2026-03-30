#!/usr/bin/env bash
# Start the Understanding LLMs by Building One interactive web tutorial
# Usage: ./start_tutorial.sh [port]  (default: 8000)
set -e

PORT="${1:-8000}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENT_DIR="$ROOT_DIR/walk-the-code"
SIBLING_WTC_DIR="$(cd "$ROOT_DIR/../walk-the-code" 2>/dev/null && pwd || true)"

SERVER_CMD=()
SERVER_SOURCE=""

if [[ -n "$SIBLING_WTC_DIR" && -f "$SIBLING_WTC_DIR/server.py" ]]; then
    SERVER_CMD=(python3 "$SIBLING_WTC_DIR/server.py")
    SERVER_SOURCE="local sibling repo: $SIBLING_WTC_DIR"
elif command -v wtc-serve &> /dev/null; then
    SERVER_CMD=(wtc-serve)
    SERVER_SOURCE="installed tool: $(command -v wtc-serve)"
else
    echo "Installing walk-the-code..."
    uv tool install "walk-the-code @ git+https://github.com/danilop/walk-the-code"
    SERVER_CMD=(wtc-serve)
    SERVER_SOURCE="newly installed tool: $(command -v wtc-serve)"
fi

echo "Starting Understanding LLMs by Building One at http://localhost:$PORT"
echo "Using walk-the-code from $SERVER_SOURCE"

# Open browser after a short delay (background)
(sleep 1 && python3 -m webbrowser "http://localhost:$PORT") &

# Start server (foreground — Ctrl+C to stop)
exec "${SERVER_CMD[@]}" --config "$CONTENT_DIR/config.json" "$PORT"
