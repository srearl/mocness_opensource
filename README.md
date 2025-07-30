# MOCNESS Field Sheets Extraction - Open Source Implementation

An open source solution for extracting data from MOCNESS (Multiple Opening/Closing Net and Environmental Sensing System) field sheet images using state-of-the-art document understanding models.

## 🚀 Features

- **Multiple Extraction Methods**: 
  - **TrOCR** for optical character recognition
  - **LayoutLMv3** for document understanding and Q&A
  - **Donut** for end-to-end document parsing
- **Automated Processing**: Batch process multiple field sheets
- **Multiple Output Formats**: JSON, CSV, and summary reports
- **GPU Acceleration**: Automatic GPU detection and usage
- **Error Handling**: Robust error handling and logging
- **Zero API Costs**: Completely open source, no external API dependencies

## 🔧 Installation

### Prerequisites
- Python 3.8+
- uv (recommended) or pip

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Run setup verification:
   ```bash
   uv run python setup_check.py
   ```

## 📁 Project Structure

```
mocness_opensource/
├── main.py              # Main extraction script
├── config.py            # Configuration and settings
├── utils.py             # Utility functions
├── setup_check.py       # Setup verification script
├── create_samples.py    # Create sample images for testing
├── input/               # Place your MOCNESS images here
├── output/              # Extracted data will be saved here
└── .env                 # Environment variables (optional)
```

## 🖼️ Input Requirements

Place your MOCNESS field sheet images in the `input/` directory with the following naming pattern:
- `tow_001_form.png` - Form sheets
- `tow_001_notes.png` - Notes sheets
- `tow_002_form.png`
- etc.

## 🏃‍♂️ Usage

### Basic Usage
```bash
# Extract from all images using all methods
uv run python main.py

# Specify custom directories
uv run python main.py --input-dir /path/to/images --output-dir /path/to/output

# Use specific extraction method
uv run python main.py --method trocr
uv run python main.py --method layoutlm
uv run python main.py --method donut
```

### Using VS Code Tasks
Open the project in VS Code and use the predefined tasks:
- **Setup Check**: Verify installation
- **Run MOCNESS Extraction**: Full extraction
- **Run Extraction (TrOCR only)**: Fast OCR-only extraction

### Environment Variables
Create a `.env` file (optional):
```env
INPUT_DIR=./input
OUTPUT_DIR=./output
HF_HOME=./models_cache  # Optional: Custom model cache directory
```

## 🧪 Testing with Samples

Create sample images for testing:
```bash
uv run python create_samples.py
```

This creates realistic sample MOCNESS field sheets in the `input/` directory.

## 📊 Output

The extraction produces:
- **Individual JSON files**: One per image with detailed extraction results
- **Combined JSON**: All results in a single file
- **CSV file**: Tabular format for easy analysis
- **Summary report**: Processing statistics and error details
- **Log file**: Detailed processing logs

### Output Structure
```json
{
  "file_path": "input/tow_001_form.png",
  "timestamp": "2025-07-29T...",
  "extraction_methods": {
    "trocr": {
      "raw_text": "Extracted text content..."
    },
    "layoutlm": {
      "What is the station number?": "ST-001",
      "What is the tow number?": "TOW-001",
      ...
    },
    "donut": {
      "extracted_text": "Structured document content..."
    }
  }
}
```

## 🔧 Models Used

- **microsoft/layoutlmv3-base**: Document understanding and question answering
- **microsoft/trocr-base-printed**: Optical character recognition
- **naver-clova-ix/donut-base-finetuned-cord-v2**: End-to-end document understanding

## ⚡ Performance

### Hardware Requirements
- **Minimum**: 4GB RAM, CPU-only processing
- **Recommended**: 8GB+ RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage**: ~3GB for all models

### Processing Speed
- **CPU**: ~30-60 seconds per image
- **GPU**: ~5-15 seconds per image

## 🛠️ Configuration

Customize extraction questions in `config.py`:
```python
MOCNESS_QUESTIONS = [
    "What is the station number?",
    "What is the tow number?",
    # Add your custom questions...
]
```

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection
3. **No images found**: Verify file naming pattern

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
python main.py --device cpu
```

## 📈 Comparison with OpenAI API

| Feature | Open Source | OpenAI API |
|---------|-------------|------------|
| Cost | Free | ~$0.01-0.02 per image |
| Privacy | Complete | Data sent to OpenAI |
| Customization | Full control | Limited |
| Dependencies | Local models | Internet + API key |
| Speed | Fast (with GPU) | Variable |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Microsoft LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm)
- [Microsoft TrOCR](https://github.com/microsoft/TrOCR)
- [Naver Clova Donut](https://github.com/clovaai/donut)

## 📞 Support

- Create an [issue](https://github.com/your-repo/mocness-opensource/issues) for bug reports
- Check existing [discussions](https://github.com/your-repo/mocness-opensource/discussions) for questions
- See [setup_check.py](setup_check.py) for verification steps
