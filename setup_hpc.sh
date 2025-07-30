#!/bin/bash
# MOCNESS Extraction - HPC Setup Script

set -e  # Exit on any error

echo "=================================================="
echo "MOCNESS Extraction - HPC Setup"
echo "=================================================="

# Set up personal cache directories to avoid shared cache conflicts
USER_CACHE_DIR="/scratch/$USER/.mamba"
export MAMBA_PKGS_DIRS="$USER_CACHE_DIR/pkgs"
export CONDA_PKGS_DIRS="$USER_CACHE_DIR/pkgs"
mkdir -p "$USER_CACHE_DIR/pkgs"

echo "✅ Using personal cache directory: $USER_CACHE_DIR"

# Check if mamba or conda is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ Using mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✅ Using conda"
else
    echo "❌ Neither mamba nor conda found!"
    echo "Please install miniconda or mambaforge first."
    echo "On HPC systems, try: module load miniconda3"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Create or update environment
ENV_NAME="mocness-extraction"

echo "📦 Setting up environment with personal cache..."

if $CONDA_CMD env list | grep -q $ENV_NAME; then
    echo "📦 Environment '$ENV_NAME' already exists. Updating..."
    $CONDA_CMD env update -f environment.yml
else
    echo "📦 Creating new environment '$ENV_NAME'..."
    echo "   This may take 10-15 minutes on HPC systems..."
    $CONDA_CMD env create -f environment.yml
fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To use the environment:"
echo "  $CONDA_CMD activate $ENV_NAME"
echo ""
echo "To test the installation:"
echo "  python setup_check_mamba.py"
echo ""
echo "To create sample data:"
echo "  python create_samples.py"
echo ""
echo "To run extraction:"
echo "  python main.py"
echo ""
echo "For HPC batch jobs, see README_MAMBA.md for SLURM examples."
echo "=================================================="
