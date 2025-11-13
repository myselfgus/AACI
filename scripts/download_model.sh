#!/bin/bash
# Script to download and setup Whisper Large 3 model

set -e

MODEL_NAME="${MODEL_NAME:-openai/whisper-large-v3}"
MODEL_CACHE="${MODEL_CACHE:-./models}"

echo "Downloading Whisper model: $MODEL_NAME"
echo "Cache directory: $MODEL_CACHE"

# Create cache directory
mkdir -p "$MODEL_CACHE"

# Download model using Python
python3 << EOF
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

model_name = os.getenv("MODEL_NAME", "openai/whisper-large-v3")
cache_dir = os.getenv("MODEL_CACHE", "./models")

print(f"Downloading processor...")
processor = WhisperProcessor.from_pretrained(
    model_name,
    cache_dir=cache_dir,
)

print(f"Downloading model...")
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=cache_dir,
)

print("Model downloaded successfully!")
print(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
EOF

echo "Setup complete!"
