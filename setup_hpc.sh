#!/bin/bash
# MOCNESS Extraction - HPC Setup Script

set -e  # Exit on any error

echo "=================================================="
echo "MOCNESS Extraction - HPC Setup"
echo "=================================================="

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

if $CONDA_CMD env list | grep -q $ENV_NAME; then
    echo "📦 Environment '$ENV_NAME' already exists. Updating..."
    $CONDA_CMD env update -f environment.yml
else
    echo "📦 Creating new environment '$ENV_NAME'..."
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
