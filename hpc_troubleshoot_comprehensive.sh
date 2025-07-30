#!/bin/bash

echo "=== MOCNESS HPC Comprehensive Troubleshooting Script ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo

# Check if mamba environment exists
echo "=== Checking Mamba Environment ==="
if mamba env list | grep -q "mocness-extraction"; then
    echo "✅ mocness-extraction environment found"
else
    echo "❌ mocness-extraction environment not found"
    echo "Creating environment from environment.yml..."
    mamba env create -f environment.yml
fi

echo
echo "=== Activating Environment ==="
eval "$(mamba shell.bash hook)"
mamba activate mocness-extraction
echo "Current environment: $CONDA_DEFAULT_ENV"

echo
echo "=== Installing Missing Dependencies ==="

# Install system dependencies
echo "Installing system dependencies..."
mamba install -y tesseract protobuf sentencepiece

# Check if tesseract is in PATH
echo "Checking tesseract PATH..."
which tesseract || echo "⚠️ tesseract not found in PATH"

# Install tesseract-ocr system package if needed
echo "Installing system tesseract-ocr package..."
mamba install -y tesseract-ocr || echo "Trying alternative tesseract installation..."

# Add conda-forge channel for better tesseract support
echo "Adding conda-forge channel for tesseract..."
mamba install -c conda-forge -y tesseract pytesseract

# Install additional Python packages for LayoutLMv3
echo "Installing LayoutLMv3 dependencies..."
pip install layoutparser[layoutmodels,tesseract] || echo "LayoutParser installation failed" 

# Try to install detectron2 (may fail on some systems)
echo "Attempting to install detectron2..."
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html || echo "Detectron2 installation failed - may not be critical"

# Install additional packages for better tokenizer support
echo "Installing additional tokenizer dependencies..."
pip install tokenizers sentencepiece protobuf

# Set tesseract path if needed
echo "Setting up tesseract path..."
python -c "
import pytesseract
import os
# Try to find tesseract in common conda locations
possible_paths = [
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'tesseract'),
    '/usr/bin/tesseract',
    '/usr/local/bin/tesseract'
]
for path in possible_paths:
    if os.path.exists(path):
        print(f'Found tesseract at: {path}')
        pytesseract.pytesseract.tesseract_cmd = path
        break
else:
    print('Could not find tesseract binary')
"

echo
echo "=== Verification ==="
echo "Testing tesseract installation..."
which tesseract && tesseract --version || echo "❌ tesseract binary not found"

echo
echo "Testing tesseract accessibility from Python..."
python -c "
import subprocess
try:
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
    print(f'✅ Tesseract accessible from Python: {result.stdout.split()[1]}')
except Exception as e:
    print(f'❌ Tesseract not accessible from Python: {e}')
"

echo
echo "Testing pytesseract Python package..."
python -c "
try:
    import pytesseract
    print(f'✅ pytesseract imported successfully')
    # Test if pytesseract can find tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f'✅ pytesseract can access tesseract: {version}')
    except Exception as e:
        print(f'❌ pytesseract cannot access tesseract: {e}')
except Exception as e:
    print(f'❌ pytesseract import failed: {e}')
"

echo
echo "Testing protobuf installation..."
python -c "import google.protobuf; print(f'Protobuf version: {google.protobuf.__version__}')"

echo
echo "Testing sentencepiece installation..."
python -c "import sentencepiece as spm; print(f'SentencePiece version: {spm.__version__}')"

echo
echo "=== Testing MOCNESS Models ==="
echo "Testing TrOCR (should work)..."
python -c "
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    print('✅ TrOCR loaded successfully')
except Exception as e:
    print(f'❌ TrOCR error: {e}')
"

echo
echo "Testing LayoutLMv3..."
python -c "
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('microsoft/layoutlmv3-base')
    print('✅ LayoutLMv3 loaded successfully')
except Exception as e:
    print(f'❌ LayoutLMv3 error: {e}')
"

echo
echo "Testing Donut..."
python -c "
try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    print('✅ Donut loaded successfully')
except Exception as e:
    print(f'❌ Donut error: {e}')
"

echo
echo "=== Debugging LayoutLMv3 Issues ==="
echo "Checking LayoutLMv3 processor output format..."
python -c "
try:
    from transformers import LayoutLMv3Processor
    from PIL import Image
    import numpy as np
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='white')
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')
    
    # Test encoding
    encoding = processor(test_image, 'test question', return_tensors='pt')
    print(f'✅ Processor output keys: {list(encoding.keys())}')
    if hasattr(encoding, 'input_ids'):
        print(f'✅ input_ids shape: {encoding.input_ids.shape}')
    else:
        print(f'❌ No input_ids attribute found')
        print(f'Available attributes: {dir(encoding)}')
except Exception as e:
    print(f'❌ LayoutLMv3 processor test error: {e}')
"

echo
echo "=== Next Steps ==="
echo "If all models loaded successfully, run:"
echo "python main.py --input-dir ./input --output-dir ./output"
echo
echo "To test individual models:"
echo "python main.py --input-dir ./input --output-dir ./output --method trocr"
echo "python main.py --input-dir ./input --output-dir ./output --method layoutlm"
echo "python main.py --input-dir ./input --output-dir ./output --method donut"
echo
echo "=== Troubleshooting Complete ==="
