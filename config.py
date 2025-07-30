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
    "What is the cruise number?",
    "What is the tow number?", 
    "What is the date?",
    "What is the location?",
    "What time did the tow start?",
    "What time did the tow end?",
    "What is the GMT start time?",
    "What is the GMT end time?",
    "What is the start latitude?",
    "What is the start longitude?",
    "What is the end latitude?",
    "What is the end longitude?",
    "What is the wind speed?",
    "What is the wind direction?",
    "What is the sea state?",
    "What is the net size?",
    "What is the net mesh?",
    "What is the net condition?",
    "What are the general comments?",
    "What equipment was used?",
    "What is the maximum depth reached?",
    "What flowmeter readings were recorded?",
    "What volumes were filtered?",
    "What biological observations were made?"
]

# Specific questions for form vs notes pages
FORM_QUESTIONS = [
    "What is the cruise number?",
    "What is the tow number?", 
    "What is the date?",
    "What is the location?",
    "What is the local start time?",
    "What is the local end time?",
    "What is the GMT start time?",
    "What is the GMT end time?",
    "What is the start latitude?",
    "What is the start longitude?", 
    "What is the end latitude?",
    "What is the end longitude?",
    "What is the wind speed?",
    "What is the wind direction?",
    "What is the sea state?",
    "What is the net size?",
    "What is the net mesh size?",
    "What is the net condition percentage?"
]

NOTES_QUESTIONS = [
    "What biological specimens were observed?",
    "What are the comments for each net?",
    "What preservation methods were used?",
    "What special observations were made?",
    "What samples were collected for DNA?",
    "What jellies or other organisms were noted?",
    "What splitting ratios were recorded?",
    "What biomass observations were made?"
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
