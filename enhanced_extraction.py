#!/usr/bin/env python3
"""
Enhanced MOCNESS Field Sheets Extraction
========================================

This script provides enhanced image preprocessing and targeted field extraction
for MOCNESS field sheet forms.

Key improvements:
1. Advanced image preprocessing (contrast, sharpening, deskewing)
2. Region-based extraction for specific form fields
3. Multiple preprocessing techniques
4. Better OCR configuration
"""

import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def enhance_image_for_ocr(image_path: str, output_dir: str = "temp") -> list:
    """
    Apply multiple image enhancement techniques for better OCR.
    Returns list of enhanced image paths.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    enhanced_images = []
    base_name = Path(image_path).stem
    
    # 1. Grayscale with contrast adjustment
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast_enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    
    # 2. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. Morphological operations to clean text
    kernel = np.ones((1,1), np.uint8)
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Save enhanced images
    enhanced_paths = []
    for i, (img, name) in enumerate([
        (contrast_enhanced, "contrast"),
        (adaptive_thresh, "adaptive"),
        (morphed, "morphed"),
        (denoised, "denoised")
    ]):
        path = f"{output_dir}/{base_name}_{name}.png"
        cv2.imwrite(path, img)
        enhanced_paths.append(path)
        enhanced_images.append(img)
    
    return enhanced_paths, enhanced_images

def extract_text_with_multiple_methods(image_path: str) -> dict:
    """
    Extract text using multiple methods and image enhancements.
    """
    results = {}
    
    # Original image OCR
    try:
        original_text = pytesseract.image_to_string(Image.open(image_path))
        results['original_ocr'] = original_text.strip()
    except Exception as e:
        results['original_ocr'] = f"Error: {e}"
    
    # Enhanced images OCR
    try:
        enhanced_paths, enhanced_images = enhance_image_for_ocr(image_path)
        
        for i, (path, img_array) in enumerate(zip(enhanced_paths, enhanced_images)):
            try:
                # Convert numpy array to PIL Image for tesseract
                if len(img_array.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(img_array, mode='L')
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                
                # Extract text with different configurations
                config_basic = '--oem 3 --psm 6'
                config_detailed = '--oem 3 --psm 11'
                
                text_basic = pytesseract.image_to_string(pil_img, config=config_basic)
                text_detailed = pytesseract.image_to_string(pil_img, config=config_detailed)
                
                method_name = path.split('_')[-1].replace('.png', '')
                results[f'{method_name}_basic'] = text_basic.strip()
                results[f'{method_name}_detailed'] = text_detailed.strip()
                
            except Exception as e:
                results[f'enhanced_{i}'] = f"Error: {e}"
                
    except Exception as e:
        results['enhanced_error'] = f"Error in enhancement: {e}"
    
    # TrOCR for comparison
    try:
        results['trocr'] = extract_with_trocr(image_path)
    except Exception as e:
        results['trocr'] = f"TrOCR Error: {e}"
    
    return results

def extract_with_trocr(image_path: str) -> str:
    """Extract text using TrOCR model."""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Resize if too large
    max_size = 2048
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

def analyze_form_structure(image_path: str) -> dict:
    """
    Analyze the structure of the form to identify field regions.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detect lines
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
    
    # Find contours to identify form sections
    contours, _ = cv2.findContours(
        cv2.bitwise_or(horizontal_lines, vertical_lines),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Identify form regions
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter small artifacts
            regions.append({
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': w * h
            })
    
    return {
        'total_regions': len(regions),
        'regions': sorted(regions, key=lambda r: r['area'], reverse=True)[:10]  # Top 10 largest
    }

def main():
    """Main function to test enhanced extraction."""
    setup_logging()
    
    # Test files
    form_path = "/home/srearl/localRepos/mocness_opensource/justone/tow_1_form.png"
    notes_path = "/home/srearl/localRepos/mocness_opensource/justone/tow_1_notes.png"
    
    for file_path in [form_path, notes_path]:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            continue
            
        logging.info(f"Processing: {file_path}")
        
        # Analyze structure
        logging.info("Analyzing form structure...")
        structure = analyze_form_structure(file_path)
        logging.info(f"Found {structure['total_regions']} regions")
        
        # Extract text with multiple methods
        logging.info("Extracting text with multiple methods...")
        results = extract_text_with_multiple_methods(file_path)
        
        # Save results
        output_file = f"enhanced_extraction_{Path(file_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'file_path': file_path,
                'structure_analysis': structure,
                'text_extraction': results
            }, f, indent=2)
        
        logging.info(f"Results saved to: {output_file}")
        
        # Print summary
        print(f"\n=== {Path(file_path).name} ===")
        print(f"Regions detected: {structure['total_regions']}")
        for method, text in results.items():
            if text and len(text.strip()) > 0:
                print(f"{method}: {text[:100]}..." if len(text) > 100 else f"{method}: {text}")

if __name__ == "__main__":
    main()
