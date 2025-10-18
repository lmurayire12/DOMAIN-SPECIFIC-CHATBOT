#!/bin/bash

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🤖 Pre-downloading FLAN-T5 model to cache..."
python -c "from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM; print('Downloading tokenizer...'); AutoTokenizer.from_pretrained('google/flan-t5-small'); print('Downloading model...'); TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', from_pt=True); print('✅ Model cached successfully!')"

echo "✅ Build complete!"
