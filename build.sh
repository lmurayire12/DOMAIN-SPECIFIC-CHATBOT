#!/bin/bash
set -euo pipefail

echo "🐍 Python version: $(python --version)"
echo "📦 Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing Python dependencies..."
# Use no cache and retry once on transient network failure
if ! pip install --no-cache-dir -r requirements.txt; then
	echo "⚠️ pip install failed; retrying in 5 seconds..."
	sleep 5
	pip install --no-cache-dir -r requirements.txt
fi

echo "🤖 Pre-downloading FLAN-T5 model to cache..."
python -c "from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM; print('Downloading tokenizer...'); AutoTokenizer.from_pretrained('google/flan-t5-small'); print('Downloading model...'); TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', from_pt=True); print('✅ Model cached successfully!')"

echo "🧠 Pre-downloading sentence-transformers embedding model..."
python -c "from sentence_transformers import SentenceTransformer; print('Downloading all-MiniLM-L6-v2...'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('✅ Embedding model cached!')"

echo "✅ Build complete!"
