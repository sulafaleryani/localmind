#!/usr/bin/env bash
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a
MODEL="${1:-${OLLAMA_MODEL:-llama3.2}}"
command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
ollama pull "$MODEL"
