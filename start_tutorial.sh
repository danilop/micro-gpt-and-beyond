#!/usr/bin/env bash
# Start the microGPT interactive web tutorial
# Usage: ./start_tutorial.sh [port]  (default: 8000)
set -e

PORT="${1:-8000}"
DIR="$(cd "$(dirname "$0")/walk-the-code" && pwd)"

# Install walk-the-code from GitHub if not available
if ! command -v wtc-serve &> /dev/null; then
    echo "Installing walk-the-code..."
    uv tool install "walk-the-code @ git+https://github.com/danilop/walk-the-code"
fi

echo "Starting microGPT tutorial at http://localhost:$PORT"

# Open browser after a short delay (background)
(sleep 1 && python3 -m webbrowser "http://localhost:$PORT") &

# Start server (foreground — Ctrl+C to stop)
exec wtc-serve --config "$DIR/config.json" "$PORT"
