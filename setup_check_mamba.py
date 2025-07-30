#!/usr/bin/env python3
"""
Setup and verification script for MOCNESS extraction project.
HPC/Mamba version.
"""

import sys
import os
import subprocess
from pathlib import Path
import torch
from transformers import pipeline

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_conda_mamba():
    """Check if conda/mamba is available."""
    mamba_available = False
    conda_available = False
    
    try:
        subprocess.run(["mamba", "--version"], capture_output=True, check=True)
        mamba_available = True
        print("✅ Mamba available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        conda_available = True
        if not mamba_available:
            print("✅ Conda available (mamba recommended for faster installs)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    if not mamba_available and not conda_available:
        print("❌ Neither mamba nor conda found")
        print("   Please install miniconda/mambaforge first")
        return False
    
    return True

def check_environment():
    """Check if we're in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"✅ Current environment: {conda_env}")
    
    # Check if environment.yml exists
    env_file = Path("environment.yml")
    if env_file.exists():
        print("✅ environment.yml found")
    else:
        print("❌ environment.yml not found")
        return False
    
    return True

def check_torch():
    """Check PyTorch installation."""
    try:
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ️  CUDA not available - will use CPU")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_transformers():
    """Check Transformers installation."""
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        # Test basic functionality
        print("Testing basic transformers functionality...")
        classifier = pipeline("sentiment-analysis")
        result = classifier("This is a test")
        print("✅ Transformers working correctly")
        return True
    except ImportError:
        print("❌ Transformers not installed")
        return False
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def check_directories():
    """Check for required directories."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Read environment variables
        try:
            with open(env_file) as f:
                content = f.read()
                if "INPUT_DIR" in content:
                    print("✅ INPUT_DIR configured in .env")
                else:
                    print("⚠️  INPUT_DIR not found in .env")
                    
                if "OUTPUT_DIR" in content:
                    print("✅ OUTPUT_DIR configured in .env")
                else:
                    print("⚠️  OUTPUT_DIR not found in .env")
        except Exception as e:
            print(f"⚠️  Could not read .env file: {e}")
    else:
        print("⚠️  .env file not found")
        print("   You can create one with:")
        print("   INPUT_DIR=./input")
        print("   OUTPUT_DIR=./output")

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        content = """# MOCNESS Extraction Environment Variables
INPUT_DIR=./input
OUTPUT_DIR=./output

# Optional: If you want to use specific model cache directory
# HF_HOME=./models_cache
"""
        with open(env_file, 'w') as f:
            f.write(content)
        print("✅ Created sample .env file")

def create_directories():
    """Create input and output directories."""
    input_dir = Path("./input")
    output_dir = Path("./output")
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    print("✅ Created input and output directories")
    
    # Create a README in input directory
    readme_content = """# Input Directory

Place your MOCNESS field sheet images here.

Expected file naming pattern:
- tow_001_form.png (form sheets)
- tow_001_notes.png (notes sheets)
- tow_002_form.png
- etc.

The extraction script will automatically find and process all files matching this pattern.
"""
    
    readme_file = input_dir / "README.md"
    if not readme_file.exists():
        with open(readme_file, 'w') as f:
            f.write(readme_content)

def print_mamba_instructions():
    """Print instructions for setting up the environment."""
    print("\n" + "=" * 60)
    print("MAMBA/CONDA SETUP INSTRUCTIONS")
    print("=" * 60)
    print("To set up the environment for the first time:")
    print("")
    print("1. Create environment from file:")
    print("   mamba env create -f environment.yml")
    print("   # OR with conda:")
    print("   conda env create -f environment.yml")
    print("")
    print("2. Activate the environment:")
    print("   mamba activate mocness-extraction")
    print("   # OR with conda:")
    print("   conda activate mocness-extraction")
    print("")
    print("3. Run this setup check again:")
    print("   python setup_check.py")
    print("")
    print("To update existing environment:")
    print("   mamba env update -f environment.yml")
    print("=" * 60)

def main():
    """Run all checks."""
    print("MOCNESS Extraction - Setup Verification (Mamba/HPC Version)")
    print("=" * 60)
    
    # Check basic requirements
    basic_checks = [
        check_python_version,
        check_conda_mamba,
        check_environment,
    ]
    
    basic_passed = True
    for check in basic_checks:
        if not check():
            basic_passed = False
    
    if not basic_passed:
        print_mamba_instructions()
        return
    
    # Check package installations
    package_checks = [
        check_torch,
        check_transformers,
    ]
    
    packages_passed = True
    for check in package_checks:
        if not check():
            packages_passed = False
    
    print("\nSetup:")
    check_directories()
    create_sample_env()
    create_directories()
    
    print("\n" + "=" * 60)
    if packages_passed:
        print("✅ All checks passed! Ready to run extraction.")
        print("\nNext steps:")
        print("1. Place your MOCNESS images in the ./input directory")
        print("2. Run: python main.py --input-dir ./input --output-dir ./output")
    else:
        print("❌ Some package checks failed.")
        print("\nIf you haven't created the environment yet:")
        print("   mamba env create -f environment.yml")
        print("   mamba activate mocness-extraction")
        print("\nTo update existing environment:")
        print("   mamba env update -f environment.yml")
    
    print("\nFor help:")
    print("python main.py --help")

if __name__ == "__main__":
    main()
