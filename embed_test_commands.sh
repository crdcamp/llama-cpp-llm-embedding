#!/usr/bin/env bash
# INCOMPLETE SCRIPT
TARGET_MODEL="Qwen3-Embedding-8B-Q6_K.gguf"

MODELS_DIR="/models/"
EMBED_MODEL_DIR="models/embed_models/"
EMBED_DOCS="data/summary"/*

echo -e "DOCUMENTS TO EMBED:\n$(ls $EMBED_DOCS)\n"

for model in "$EMBED_MODEL_DIR"*; do
    if [[ "$model" == *"$TARGET_MODEL" ]]; then
        TARGET_MODEL_PATH="$model"
        echo -e "TARGET MODEL FOUND: $TARGET_MODEL_PATH\nStarting embedding process...\n"

        for doc in $EMBED_DOCS; do
            echo "Embedding: $doc"
            llama-embedding -m "$TARGET_MODEL_PATH" --text "$(cat "$doc")"
        done
        break

    else
        echo "MODEL NOT FOUND"
    fi
done
