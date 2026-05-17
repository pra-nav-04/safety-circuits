#!/usr/bin/env bash
# Run this once to create ~/.kaggle/kaggle.json from your API key.
# Usage: bash scripts/kaggle_setup_creds.sh

set -euo pipefail

echo "You need two things from kaggle.com → Settings → API:"
echo "  1. Your Kaggle username"
echo "  2. Your API key (the long string from 'Create New API Token')"
echo ""

read -rp "Kaggle username: " KAGGLE_USER
read -rsp "Kaggle API key: " KAGGLE_KEY
echo ""

mkdir -p ~/.kaggle
printf '{"username":"%s","key":"%s"}' "$KAGGLE_USER" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Saved to ~/.kaggle/kaggle.json"
echo ""
echo "Testing connection..."
kaggle datasets list --max-size 1 && echo "✓ Kaggle API working"
