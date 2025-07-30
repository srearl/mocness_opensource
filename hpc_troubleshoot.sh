#!/bin/bash
# MOCNESS HPC Troubleshooting Script

echo "=================================================="
echo "MOCNESS HPC Troubleshooting"
echo "=================================================="

# Check current resources
echo "Current resource usage:"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
echo "Current directory: $(pwd)"
echo ""

# Check if we're in a job
if [ -n "$SLURM_JOB_ID" ]; then
    echo "✅ Running in SLURM job: $SLURM_JOB_ID"
    echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
    echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
else
    echo "⚠️  Not in a SLURM job - consider requesting interactive session:"
    echo "   salloc --mem=16G --cpus-per-task=4 --time=2:00:00"
fi
echo ""

# Set up cache directories
USER_CACHE="/scratch/$USER/.mamba"
echo "Setting up personal cache directories..."
echo "Cache directory: $USER_CACHE"

export MAMBA_PKGS_DIRS="$USER_CACHE/pkgs"
export CONDA_PKGS_DIRS="$USER_CACHE/pkgs"
mkdir -p "$USER_CACHE/pkgs"
mkdir -p "$USER_CACHE/envs"

echo "✅ Cache directories created"
echo ""

# Check mamba/conda
if command -v mamba &> /dev/null; then
    echo "✅ Mamba available: $(mamba --version)"
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    echo "✅ Conda available: $(conda --version)"
    CONDA_CMD="conda"
else
    echo "❌ Neither mamba nor conda found!"
    echo "Try: module load miniconda3"
    exit 1
fi
echo ""

# Check if environment exists
ENV_NAME="mocness-extraction"
if $CONDA_CMD env list | grep -q $ENV_NAME; then
    echo "✅ Environment '$ENV_NAME' already exists"
    echo "To activate: $CONDA_CMD activate $ENV_NAME"
else
    echo "ℹ️  Environment '$ENV_NAME' not found"
    echo ""
    echo "To create environment:"
    echo "1. Make sure you're in the project directory with environment.yml"
    echo "2. Run: $CONDA_CMD env create -f environment.yml"
    echo ""
    echo "If that fails, try step-by-step installation:"
    echo "   $CONDA_CMD create -n $ENV_NAME python=3.11"
    echo "   $CONDA_CMD activate $ENV_NAME"
    echo "   $CONDA_CMD install pytorch torchvision -c pytorch"
    echo "   $CONDA_CMD install transformers datasets -c conda-forge"
    echo "   pip install python-dotenv"
fi
echo ""

# Check project files
echo "Checking project files:"
if [ -f "environment.yml" ]; then
    echo "✅ environment.yml found"
else
    echo "❌ environment.yml not found - are you in the right directory?"
fi

if [ -f "main.py" ]; then
    echo "✅ main.py found"
else
    echo "❌ main.py not found - are you in the right directory?"
fi
echo ""

echo "Environment variables set:"
echo "MAMBA_PKGS_DIRS=$MAMBA_PKGS_DIRS"
echo "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
echo ""

echo "=================================================="
echo "Next steps:"
echo "1. If not in SLURM job: salloc --mem=16G --cpus-per-task=4 --time=2:00:00"
echo "2. Load conda: module load miniconda3 (or similar)"
echo "3. Create environment: $CONDA_CMD env create -f environment.yml"
echo "4. Activate: $CONDA_CMD activate $ENV_NAME"
echo "5. Test: python setup_check_mamba.py"
echo "=================================================="
