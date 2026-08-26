#!/usr/bin/env bash
# Start the Ollama server in the background.
# Must be run before any experiment.

OLLAMA_BIN="$HOME/apps/ollama/bin/ollama"
OLLAMA_MODELS="$HOME/ollama-models"
LOG_FILE="/tmp/ollama.log"

if ! command -v "$OLLAMA_BIN" &>/dev/null; then
    echo "ERROR: Ollama binary not found at $OLLAMA_BIN"
    exit 1
fi

# Check if already running
if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo "Ollama server is already running."
    OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" list
    exit 0
fi

echo "Starting Ollama server..."
OLLAMA_MODELS="$OLLAMA_MODELS" nohup "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "  PID: $SERVER_PID"
echo "  Log: $LOG_FILE"

# Wait for it to become available (up to 15 seconds)
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
        echo "  Ready after ${i}s."
        OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" list
        exit 0
    fi
done

echo "WARNING: Server did not respond within 15 seconds. Check $LOG_FILE"
exit 1
