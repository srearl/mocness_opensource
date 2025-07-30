"""
Configuration settings for MOCNESS extraction.
"""

import os
from pathlib import Path

# Default model configurations
MODELS = {
    "layoutlm": {
        "model_name": "microsoft/layoutlmv3-base",
        "description": "LayoutLMv3 for document understanding and Q&A"
    },
    "trocr": {
        "model_name": "microsoft/trocr-base-printed",
        "description": "TrOCR for optical character recognition"
    },
    "donut": {
        "model_name": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "description": "Donut for end-to-end document understanding"
    }
}

# Default extraction questions for MOCNESS forms
MOCNESS_QUESTIONS = [
    "What is the station number?",
    "What is the tow number?", 
    "What is the date?",
    "What is the start time?",
    "What is the end time?",
    "What is the latitude?",
    "What is the longitude?",
    "What is the depth?",
    "What is the maximum depth?",
    "What is the volume filtered?",
    "What is the flowmeter reading?",
    "What is the flowmeter start reading?",
    "What is the flowmeter end reading?",
    "What are the net mesh sizes?",
    "What is the vessel name?",
    "What is the cruise name?",
    "Who is the chief scientist?",
    "Who collected this sample?",
    "What is the weather condition?",
    "What is the sea state?",
    "What are the comments or notes?",
    "What equipment was used?",
    "What is the net type?",
    "What is the sampling method?"
]

# File patterns
FILE_PATTERNS = {
    "form": "tow_*_form.png",
    "notes": "tow_*_notes.png",
    "all": ["tow_*_form.png", "tow_*_notes.png"]
}

# Output formats
OUTPUT_FORMATS = ["json", "csv", "txt"]

# Model download cache directory
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "transformers"

# Environment variables
INPUT_DIR = os.getenv("INPUT_DIR", "./input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
