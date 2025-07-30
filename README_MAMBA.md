# MOCNESS Field Sheets Extraction - HPC/Mamba Version

An open source solution for extracting data from MOCNESS field sheet images using state-of-the-art document understanding models, optimized for HPC environments with Mamba package management.

## 🚀 Quick Start for HPC

### 1. Environment Setup
```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd mocness_opensource

# Create the conda environment
mamba env create -f environment.yml

# Activate the environment
source activate mocness-extraction

# Verify setup
python setup_check_mamba.py
```

### 2. Test with Sample Data
```bash
# Create sample MOCNESS images
python create_samples.py

# Run extraction
python main.py
```

### 3. Process Real Data
```bash
# Place your MOCNESS images in ./input directory
# Run full extraction
python main.py --input-dir ./input --output-dir ./output

# For faster processing, use single method:
python main.py --method trocr  # Fastest
```

## 📦 Package Management

### Mamba Commands
```bash
# Create environment
mamba env create -f environment.yml

# Activate environment
mamba activate mocness-extraction

# Update environment
mamba env update -f environment.yml

# List installed packages
mamba list

# Remove environment (if needed)
mamba env remove -n mocness-extraction
```

### Alternative: Conda Commands
If mamba is not available, use conda:
```bash
conda env create -f environment.yml
conda activate mocness-extraction
conda env update -f environment.yml
```

## 🖥️ HPC Considerations

### SLURM Job Script Example
```bash
#!/bin/bash
#SBATCH --job-name=mocness_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=compute

# Load conda/mamba module (adjust for your HPC)
module load miniconda3
# OR: module load mambaforge

# Activate environment
source activate mocness-extraction

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run extraction
python main.py --input-dir ./input --output-dir ./output
```

### GPU Usage (if available)
```bash
#!/bin/bash
#SBATCH --job-name=mocness_extraction_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load miniconda3
source activate mocness-extraction

# Run with GPU acceleration
python main.py --input-dir ./input --output-dir ./output
```

## 📁 Project Structure

```
mocness_opensource/
├── environment.yml          # Mamba/Conda environment file
├── setup_check_mamba.py     # Setup verification for HPC
├── main.py                  # Main extraction script (unchanged)
├── config.py                # Configuration (unchanged)
├── utils.py                 # Utilities (unchanged)
├── create_samples.py        # Sample generator (unchanged)
├── .vscode/
│   ├── tasks.json           # Original uv tasks
│   └── tasks_mamba.json     # Mamba-specific tasks
├── input/                   # Input images
└── output/                  # Results
```

## 🔧 Dependencies

The `environment.yml` includes:
- **Python** 3.8+
- **PyTorch** with CUDA support
- **Transformers** for document understanding models
- **OpenCV** for image processing
- **Pillow** for image handling
- **All required ML libraries**

## ⚡ Performance on HPC

### Resource Requirements
- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8 CPU cores, 16GB RAM
- **With GPU**: 1 GPU, 4 CPU cores, 8GB RAM

### Processing Speed
- **CPU (4 cores)**: ~20-40 seconds per image
- **CPU (8 cores)**: ~15-30 seconds per image
- **GPU**: ~5-15 seconds per image

## 🛠️ Troubleshooting HPC Issues

### Common HPC Problems
1. **Module not found**: Check available modules with `module avail`
2. **Environment activation fails**: Try `conda activate` instead of `source activate`
3. **CUDA not detected**: Check GPU allocation with `nvidia-smi`
4. **Memory issues**: Increase `--mem` in SLURM script or use `--device cpu`

### Package Installation Issues
```bash
# If mamba is slow or fails
conda env create -f environment.yml

# If pip packages fail
source activate mocness-extraction
pip install transformers datasets python-dotenv huggingface-hub

# Clear cache if needed
mamba clean --all
```

### File Permissions
```bash
# Make scripts executable
chmod +x setup_check_mamba.py
chmod +x main.py
chmod +x create_samples.py
```

## 📊 Batch Processing

### Process Multiple Directories
```bash
#!/bin/bash
# Process multiple cruise directories
for cruise_dir in /path/to/cruises/*/; do
    python main.py --input-dir "$cruise_dir/mocness_images" --output-dir "$cruise_dir/extracted_data"
done
```

### Parallel Processing
```bash
# Using GNU parallel (if available)
parallel python main.py --input-dir {} --output-dir {//}/output ::: cruise_*/input
```

## 🔄 Migration from UV

### What Changed
- ✅ **Environment management**: `environment.yml` instead of `pyproject.toml`
- ✅ **Setup script**: `setup_check_mamba.py` with conda/mamba detection
- ✅ **VS Code tasks**: `tasks_mamba.json` for mamba commands
- ✅ **Documentation**: HPC-specific instructions

### What Stayed the Same
- ✅ **Core extraction code**: `main.py` unchanged
- ✅ **Configuration**: `config.py` unchanged  
- ✅ **Utilities**: `utils.py` unchanged
- ✅ **All functionality**: Same three extraction methods
- ✅ **Output formats**: JSON, CSV, reports

## 🎯 Next Steps

1. **Setup**: Run `python setup_check_mamba.py`
2. **Test**: Create samples with `python create_samples.py`
3. **Extract**: Run `python main.py`
4. **Deploy**: Adapt SLURM script for your HPC
5. **Scale**: Process your MOCNESS image collections

## 📞 HPC Support

For HPC-specific issues:
1. Check your cluster's documentation for conda/mamba setup
2. Contact your HPC support team for module loading
3. Test with small batches before large-scale processing
4. Monitor resource usage with `htop`, `nvidia-smi`

The core extraction functionality remains identical - only the environment management has changed!
