#!/bin/sh
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
RAG_CHROMA_DIR="${RAG_CHROMA_DIR:-chroma_db}"
PORT="${PORT:-8000}"

export OLLAMA_HOST

echo "[RAG] Iniciando Ollama interno en ${OLLAMA_HOST}..."
ollama serve &
OLLAMA_PID="$!"

echo "[RAG] Esperando Ollama en ${OLLAMA_BASE_URL}..."
attempt=0
until curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 120 ]; then
        echo "[RAG] ERROR: Ollama no respondio despues de 120 segundos."
        kill "$OLLAMA_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
echo "[RAG] Ollama interno listo en ${OLLAMA_BASE_URL}."

if ollama list | awk '{print $1}' | grep -qx "$OLLAMA_MODEL"; then
    echo "[RAG] Modelo ${OLLAMA_MODEL} disponible."
else
    echo "[RAG] Descargando modelo ${OLLAMA_MODEL}..."
    ollama pull "$OLLAMA_MODEL"
    echo "[RAG] Modelo ${OLLAMA_MODEL} disponible."
fi

if [ ! -d "$RAG_CHROMA_DIR" ] || [ -z "$(find "$RAG_CHROMA_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "[RAG] Ejecutando ingest para crear ${RAG_CHROMA_DIR}..."
    python -m app.services.rag.ingest
    echo "[RAG] Ingest ejecutado."
else
    echo "[RAG] ${RAG_CHROMA_DIR} ya existe; se omite ingest."
fi

export ASSISTANT_ENGINE="rag_ollama"
export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
export OLLAMA_MODEL="$OLLAMA_MODEL"
export RAG_CHROMA_DIR="$RAG_CHROMA_DIR"

echo "[RAG] Arrancando FastAPI en puerto ${PORT}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
