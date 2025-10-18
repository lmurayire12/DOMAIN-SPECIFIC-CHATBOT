#!/bin/bash

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🤖 Pre-downloading FLAN-T5 model to cache..."
python -c "from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM; print('Downloading tokenizer...'); AutoTokenizer.from_pretrained('google/flan-t5-small'); print('Downloading model...'); TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', from_pt=True); print('✅ Model cached successfully!')"

echo "🧠 Pre-downloading sentence-transformers embedding model..."
python -c "from sentence_transformers import SentenceTransformer; print('Downloading all-MiniLM-L6-v2...'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('✅ Embedding model cached!')"

echo "✅ Build complete!"
