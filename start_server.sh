#!/bin/bash
# Start the FastAPI inference server

cd "$(dirname "$0")"

echo "🚀 Starting TabularANN Inference API Server..."
echo ""

# Set default port
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

# Optional: Set model directory (if you have an exported model)
# export MODEL_DIR=artifacts/models/champion
# export MODEL_VERSION=1

echo "Server will start on: http://${HOST}:${PORT}"
echo ""
echo "Endpoints:"
echo "  GET  /              - Root endpoint"
echo "  GET  /health        - Health check"
echo "  POST /load          - Load model (model_dir, version)"
echo "  POST /predict       - Make predictions"
echo "  GET  /model_info    - Get model information"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "Starting server..."

# Use uvicorn directly (faster than importing via python -m)
uvicorn src.serve.app:app --host "$HOST" --port "$PORT" --reload
