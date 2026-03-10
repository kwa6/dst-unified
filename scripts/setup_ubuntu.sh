#!/usr/bin/env bash
set -e

echo "Installing system dependencies..."

sudo apt update
sudo apt install -y git-lfs

git lfs install

echo "System dependencies installed."