#!/usr/bin/env bash
# DEPRECATED: Single-stage training is no longer supported.
# Use train_llama_twostage.sh instead for two-stage fine-tuning.
#
# Two-stage training approach:
# - Stage 1: Train on augmented data (LUAS or D0T)
# - Stage 2: Fine-tune on real data (MultiWOZ)
#
# Usage:
#   bash scripts/train_llama_twostage.sh luas
#   bash scripts/train_llama_twostage.sh d0t

echo "ERROR: Single-stage training is deprecated."
echo "Use two-stage training instead:"
echo
echo "  bash scripts/train_llama_twostage.sh luas    # LUAS → MultiWOZ"
echo "  bash scripts/train_llama_twostage.sh d0t     # D0T → MultiWOZ"
echo
exit 1
