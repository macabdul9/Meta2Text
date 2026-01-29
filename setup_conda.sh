#!/bin/bash
# Setup script for creating conda environment 'archaia'

set -e

echo "Creating conda environment 'archaia'..."

# Create conda environment with Python 3.10
conda create -n archaia python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate archaia

# Install PyTorch (CPU version by default, adjust if GPU needed)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install pip packages
echo "Installing Python packages..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "  conda activate archaia"
