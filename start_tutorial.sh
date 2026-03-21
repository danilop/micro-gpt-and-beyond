#!/usr/bin/env bash
# Start the microGPT interactive web tutorial
set -e

PORT=8000
DIR="$(cd "$(dirname "$0")/web_tutorial" && pwd)"

echo "Starting microGPT tutorial at http://localhost:$PORT"

# Open browser after a short delay (background)
(sleep 1 && python3 -m webbrowser "http://localhost:$PORT") &

# Start server (foreground — Ctrl+C to stop)
cd "$DIR"
exec python3 server.py
