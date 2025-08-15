def main():
    """
MOCNESS Field Sheets Extraction - Open Source Implementation
===========================================================

This script extracts data from MOCNESS field sheet images using open source models:
- LayoutLMv3 for document understanding and form extraction
- TrOCR for optical character recognition
- Donut for end-to-end document understanding

Author: GitHub Copilot
Date: July 29, 2025
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime

import torch
from PIL import Image
import cv2
import numpy as np
from transformers import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3Processor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    DonutProcessor,
    VisionEncoderDecoderModel as DonutModel
)
from datasets import Dataset
from dotenv import load_dotenv

# Configure tesseract path if in conda environment
try:
    import pytesseract
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        tesseract_path = os.path.join(conda_prefix, 'bin', 'tesseract')
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Set tesseract path to: {tesseract_path}")
except ImportError:
    pass

from config import MOCNESS_QUESTIONS, MODELS, FORM_QUESTIONS, NOTES_QUESTIONS
from utils import (
    check_gpu_availability, 
    save_results_as_csv, 
    create_summary_report, 
    validate_image_files,
    save_results_as_csv_by_tow,
    create_summary_report_by_tow
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mocness_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MOCNESSExtractor:
    """Open source document extraction for MOCNESS field sheets."""
    
    def __init__(self, device: str = None):
        """Initialize the extractor with models."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Log GPU info if available
        gpu_info = check_gpu_availability()
        if gpu_info["available"]:
            logger.info(f"GPU: {gpu_info['device_name']} ({gpu_info['memory_gb']} GB)")
        
        # Initialize models
        self.layoutlm_processor = None
        self.layoutlm_model = None
        self.trocr_processor = None
        self.trocr_model = None
        self.donut_processor = None
        self.donut_model = None
        
    def load_layoutlm(self):
        """Load LayoutLMv3 model for document understanding."""
        try:
            logger.info("Loading LayoutLMv3 model...")
            self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
                MODELS["layoutlm"]["model_name"]
            )
            self.layoutlm_model = LayoutLMv3ForQuestionAnswering.from_pretrained(
                MODELS["layoutlm"]["model_name"]
            ).to(self.device)
            logger.info("LayoutLMv3 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LayoutLMv3: {e}")
            raise
            
    def load_trocr(self):
        """Load TrOCR model for OCR."""
        try:
            logger.info("Loading TrOCR model...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                MODELS["trocr"]["model_name"]
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                MODELS["trocr"]["model_name"]
            ).to(self.device)
            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TrOCR: {e}")
            raise
            
    def load_donut(self):
        """Load Donut model for document understanding."""
        try:
            logger.info("Loading Donut model...")
            self.donut_processor = DonutProcessor.from_pretrained(
                MODELS["donut"]["model_name"]
            )
            self.donut_model = DonutModel.from_pretrained(
                MODELS["donut"]["model_name"]
            ).to(self.device)
            logger.info("Donut model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Donut: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (TrOCR works better with reasonable sizes)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Apply advanced image enhancement for better OCR."""
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        
        # Apply contrast enhancement
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(denoised, mode='L')
        
        return enhanced_pil
    
    def extract_with_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract OCR as fallback."""
        try:
            # First try with enhanced image
            enhanced_image = self.enhance_image_for_ocr(image)
            
            # Use multiple OCR configurations
            configs = [
                r'--oem 3 --psm 6',  # Basic configuration
                r'--oem 3 --psm 11', # Detailed sparse text
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,°()[]{}:;-_/@#$%&+=?!'
            ]
            
            results = []
            for config in configs:
                try:
                    text = pytesseract.image_to_string(enhanced_image, config=config)
                    if text.strip():
                        results.append(text.strip())
                except Exception as e:
                    logger.warning(f"OCR config failed: {config}, error: {e}")
                    continue
            
            # Return the longest result (usually more complete)
            if results:
                return max(results, key=len)
            else:
                return ""
            
        except Exception as e:
            logger.error(f"Error with Tesseract OCR: {e}")
            return ""
    
    def extract_with_trocr(self, image: Image.Image) -> str:
        """Extract text using TrOCR."""
        if not self.trocr_processor or not self.trocr_model:
            self.load_trocr()
            
        try:
            # Preprocess image
            image = self.preprocess_image(image)
            
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Use better generation parameters
            generated_ids = self.trocr_model.generate(
                pixel_values,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            generated_text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"Error with TrOCR extraction: {e}")
            return ""
    
    def extract_with_layoutlm(self, image: Image.Image, questions: List[str]) -> Dict[str, str]:
        """Extract structured data using LayoutLMv3."""
        if not self.layoutlm_processor or not self.layoutlm_model:
            self.load_layoutlm()
            
        results = {}
        
        for question in questions:
            try:
                encoding = self.layoutlm_processor(
                    image, question, return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                with torch.no_grad():
                    outputs = self.layoutlm_model(**encoding)
                    
                answer_start_index = outputs.start_logits.argmax()
                answer_end_index = outputs.end_logits.argmax()
                
                predict_answer_tokens = encoding['input_ids'].squeeze()[
                    answer_start_index:answer_end_index + 1
                ]
                answer = self.layoutlm_processor.tokenizer.decode(
                    predict_answer_tokens, skip_special_tokens=True
                )
                
                results[question] = answer
                
            except Exception as e:
                logger.error(f"Error with LayoutLM question '{question}': {e}")
                results[question] = ""
                
        return results
    
    def extract_with_donut(self, image: Image.Image) -> Dict:
        """Extract structured data using Donut."""
        if not self.donut_processor or not self.donut_model:
            self.load_donut()
            
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare image with error checking
            try:
                pixel_values = self.donut_processor(image, return_tensors="pt").pixel_values
            except Exception as e:
                logger.error(f"Donut image processing failed: {e}")
                return {}
            
            # Check if pixel_values is None or empty
            if pixel_values is None:
                logger.error("Donut processor returned None pixel_values")
                return {}
                
            pixel_values = pixel_values.to(self.device)
            
            # Try different approaches for decoder_input_ids
            try:
                # Method 1: Use decoder start token if available
                if hasattr(self.donut_model.config, 'decoder_start_token_id'):
                    decoder_input_ids = torch.tensor([[self.donut_model.config.decoder_start_token_id]])
                else:
                    # Method 2: Use tokenizer bos token
                    decoder_input_ids = torch.tensor([[self.donut_processor.tokenizer.bos_token_id]])
            except:
                # Method 3: Fallback to a simple approach
                decoder_input_ids = torch.tensor([[0]])  # Start with 0
                
            decoder_input_ids = decoder_input_ids.to(self.device)
            
            # Use very simple generation parameters to avoid issues
            try:
                outputs = self.donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=128,  # Very short to avoid issues
                    num_beams=1,
                    do_sample=False,
                    early_stopping=True,
                )
                
                # Simple decoding
                if outputs is not None and len(outputs) > 0:
                    sequence = self.donut_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return {"extracted_text": sequence.strip()}
                else:
                    logger.error("Donut generation returned empty results")
                    return {}
                    
            except Exception as e:
                logger.error(f"Donut generation failed: {e}")
                # Try even simpler approach
                try:
                    with torch.no_grad():
                        outputs = self.donut_model(pixel_values=pixel_values)
                    return {"extracted_text": "Donut processed (simplified output)"}
                except Exception as e2:
                    logger.error(f"Donut simplified processing also failed: {e2}")
                    return {}
            
        except Exception as e:
            logger.error(f"Error with Donut extraction: {e}")
            return {}
    
    def process_mocness_form(self, image_path: str, method: str = "all") -> Dict:
        """Process a MOCNESS form image and extract structured data."""
        logger.info(f"Processing form: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Determine if this is a form or notes page based on filename
            is_notes_page = "notes" in str(image_path).lower()
            
            # Choose appropriate questions
            if is_notes_page:
                questions = NOTES_QUESTIONS
                logger.info("Processing as notes page")
            else:
                questions = FORM_QUESTIONS  
                logger.info("Processing as form page")
            
            # Extract with multiple methods
            results = {
                "file_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "extraction_methods": {}
            }
            
            # Method 0: Tesseract OCR as baseline
            if method in ["tesseract", "all"]:
                logger.info("Extracting with Tesseract OCR...")
                tesseract_text = self.extract_with_tesseract(image)
                results["extraction_methods"]["tesseract"] = {
                    "raw_text": tesseract_text
                }
            
            # Method 1: TrOCR for basic text extraction
            if method in ["trocr", "all"]:
                logger.info("Extracting with TrOCR...")
                trocr_text = self.extract_with_trocr(image)
                results["extraction_methods"]["trocr"] = {
                    "raw_text": trocr_text
                }
            
            # Method 2: LayoutLMv3 for structured Q&A
            if method in ["layoutlm", "all"]:
                logger.info("Extracting with LayoutLMv3...")
                layoutlm_results = self.extract_with_layoutlm(image, questions)
                results["extraction_methods"]["layoutlm"] = layoutlm_results
            
            # Method 3: Donut for document understanding
            if method in ["donut", "all"]:
                logger.info("Extracting with Donut...")
                donut_results = self.extract_with_donut(image)
                results["extraction_methods"]["donut"] = donut_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "file_path": image_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_hybrid_approach(self, image_path: str) -> Dict:
        """Use a hybrid approach combining all methods intelligently."""
        logger.info(f"Using hybrid extraction for: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            is_notes_page = "notes" in str(image_path).lower()
            
            # Start with OCR to get raw text
            tesseract_text = self.extract_with_tesseract(image)
            trocr_text = self.extract_with_trocr(image)
            
            # Combine OCR results for better coverage
            combined_text = f"Tesseract: {tesseract_text}\n\nTrOCR: {trocr_text}"
            
            # Use appropriate questions
            questions = NOTES_QUESTIONS if is_notes_page else FORM_QUESTIONS
            
            # Get structured answers
            layoutlm_results = self.extract_with_layoutlm(image, questions)
            
            # Try to extract key information from raw text
            extracted_info = self.parse_raw_text(combined_text, is_notes_page)
            
            return {
                "file_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "page_type": "notes" if is_notes_page else "form",
                "raw_ocr": {
                    "tesseract": tesseract_text,
                    "trocr": trocr_text
                },
                "structured_qa": layoutlm_results,
                "parsed_info": extracted_info,
                "combined_extraction": self.combine_extractions(
                    layoutlm_results, extracted_info, combined_text
                )
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid extraction for {image_path}: {e}")
            return {"file_path": image_path, "error": str(e)}
    
    def parse_raw_text(self, text: str, is_notes: bool) -> Dict:
        """Parse useful information from raw OCR text."""
        import re
        
        info = {}
        text_upper = text.upper()
        
        if not is_notes:
            # Form page parsing
            # Look for cruise number (various patterns)
            cruise_patterns = [
                r'CRUISE[:\s]*(\d+)',
                r'(\d{4})[:\s]*CRUISE',
                r'(\d{4})[OF]*\s*(?:LOCATION|LOC)',  # Pattern like "2407 Location"
            ]
            for pattern in cruise_patterns:
                cruise_match = re.search(pattern, text_upper)
                if cruise_match:
                    info['cruise'] = cruise_match.group(1)
                    break
            
            # Look for tow number
            tow_patterns = [
                r'TOW[:\s#]*(\d+)',
                r'TOW#\s*(\w+)',
                r'MOC[:\s]*(\d+)'
            ]
            for pattern in tow_patterns:
                tow_match = re.search(pattern, text_upper)
                if tow_match:
                    info['tow'] = tow_match.group(1)
                    break
            
            # Look for location (improved patterns)
            location_patterns = [
                r'LOCATION[:\s]*([A-Z\s]+?)(?:TOW|DATE|\d|$)',
                r'(?:GUAYMAS|BASIN|GULF)[A-Z\s]*',
                r'BASIN[:\s]*([A-Z\s]+)',
            ]
            for pattern in location_patterns:
                location_match = re.search(pattern, text_upper)
                if location_match:
                    location = location_match.group(1) if len(location_match.groups()) > 0 else location_match.group(0)
                    info['location'] = location.strip()
                    break
            
            # Look for date patterns (enhanced)
            date_patterns = [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(\d{1,2}\s+\w+\s+\d{4})',
                r'DATE[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text)
                if date_match:
                    info['date'] = date_match.group(1)
                    break
            
            # Look for coordinates
            lat_match = re.search(r"(\d+°\d+\.\d+'[NS])", text)
            if lat_match:
                info['latitude'] = lat_match.group(1)
                
            lon_match = re.search(r"(\d+°\d+\.\d+'[EW])", text)
            if lon_match:
                info['longitude'] = lon_match.group(1)
                
            # Look for times (4-digit patterns)
            time_matches = re.findall(r'(\d{4})', text)
            if len(time_matches) >= 2:
                info['times'] = time_matches[:4]  # Start/end local and GMT
            
            # Look for net information
            net_match = re.search(r'NET[:\s]*(\d+)', text_upper)
            if net_match:
                info['net_size'] = net_match.group(1)
                
            # Look for mesh size
            mesh_patterns = [
                r'(\d+\s*[μµ]M)',
                r'MESH[:\s]*(\d+)',
                r'(\d+)\s*MICRON'
            ]
            for pattern in mesh_patterns:
                mesh_match = re.search(pattern, text_upper)
                if mesh_match:
                    info['net_mesh'] = mesh_match.group(1)
                    break
                
        else:
            # Notes page parsing (enhanced)
            # Look for net observations with better patterns
            net_patterns = [
                r'N(\d+)[:\s-]*([^N\n]+)',  # N1 - observation
                r'(\d+)[:\s-]*([A-Z][^N\n]+)',  # Just number - observation
            ]
            
            info['net_observations'] = {}
            for pattern in net_patterns:
                net_matches = re.findall(pattern, text_upper)
                for net_num, observation in net_matches:
                    net_key = f'N{net_num}' if net_num.isdigit() else net_num
                    info['net_observations'][net_key] = observation.strip()
            
            # Look for specific biological observations
            bio_observations = []
            bio_patterns = [
                r'MASSIVE\s+([A-Z]+)',  # MASSIVE ATOLLA
                r'JELLIES?\s+([A-Z\s]+)',  # jellies separate bottle
                r'([A-Z]+PODS?)\s*',  # myctopods, copepods
                r'(SALP[A-Z]*)',  # salps
                r'(SHRIMP[A-Z]*)',  # shrimp
                r'DNA\s+([A-Z\s]+)',  # DNA related
            ]
            
            for pattern in bio_patterns:
                matches = re.findall(pattern, text_upper)
                bio_observations.extend(matches)
            
            if bio_observations:
                info['biological_observations'] = list(set(bio_observations))  # Remove duplicates
            
            # Look for preservation methods
            preservation_terms = ['FORMALIN', 'ETHANOL', 'FROZEN', 'DNA']
            found_preservation = []
            for term in preservation_terms:
                if term in text_upper:
                    found_preservation.append(term.lower())
            if found_preservation:
                info['preservation_methods'] = found_preservation
        
        return info
    
    def combine_extractions(self, layoutlm_results: Dict, parsed_info: Dict, raw_text: str) -> Dict:
        """Intelligently combine results from different extraction methods."""
        combined = {}
        
        # Prefer parsed_info for key fields, fallback to layoutlm
        key_fields = ['cruise', 'tow', 'location', 'date', 'latitude', 'longitude']
        
        for field in key_fields:
            if field in parsed_info and parsed_info[field]:
                combined[field] = parsed_info[field]
            else:
                # Look for related answers in layoutlm results
                for question, answer in layoutlm_results.items():
                    if field.lower() in question.lower() and answer.strip():
                        combined[field] = answer
                        break
        
        # Add all unique information
        combined.update(parsed_info)
        
        # Add confidence score based on how much data we extracted
        total_fields = len(combined)
        combined['extraction_confidence'] = min(1.0, total_fields / 10.0)
        
        return combined
    
    def process_directory(self, input_dir: str, output_dir: str, method: str = "all") -> None:
        """Process all MOCNESS images in a directory and combine form/notes by tow."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        try:
            file_info = validate_image_files(input_dir)
            logger.info(f"Found {file_info['total_mocness_files']} MOCNESS files")
            logger.info(f"Form files: {len(file_info['form_files'])}")
            logger.info(f"Notes files: {len(file_info['notes_files'])}")
        except Exception as e:
            logger.error(f"Error validating input directory: {e}")
            return
        
        # Find and group files by tow number
        form_files = list(input_path.glob("tow_*_form.png"))
        notes_files = list(input_path.glob("tow_*_notes.png"))
        
        # Group files by tow number
        tow_groups = {}
        
        # Process form files
        for form_file in form_files:
            # Extract tow number from filename (e.g., "tow_001_form.png" -> "001")
            tow_num = form_file.stem.split('_')[1]
            if tow_num not in tow_groups:
                tow_groups[tow_num] = {}
            tow_groups[tow_num]['form'] = form_file
        
        # Process notes files
        for notes_file in notes_files:
            # Extract tow number from filename (e.g., "tow_001_notes.png" -> "001")
            tow_num = notes_file.stem.split('_')[1]
            if tow_num not in tow_groups:
                tow_groups[tow_num] = {}
            tow_groups[tow_num]['notes'] = notes_file
        
        logger.info(f"Processing {len(tow_groups)} tow groups using method: {method}")
        
        all_results = []
        
        for tow_num, files in tow_groups.items():
            logger.info(f"Processing tow {tow_num}")
            
            # Initialize combined result for this tow
            combined_result = {
                "tow_number": tow_num,
                "timestamp": datetime.now().isoformat(),
                "extraction_method": method,
                "form_data": {},
                "notes_data": {}
            }
            
            # Process form file if it exists
            if 'form' in files:
                logger.info(f"Processing form: {files['form']}")
                form_data = self.process_mocness_form(str(files['form']), method)
                combined_result["form_data"] = form_data
            else:
                logger.warning(f"No form file found for tow {tow_num}")
            
            # Process notes file if it exists
            if 'notes' in files:
                logger.info(f"Processing notes: {files['notes']}")
                notes_data = self.process_mocness_form(str(files['notes']), method)
                combined_result["notes_data"] = notes_data
            else:
                logger.warning(f"No notes file found for tow {tow_num}")
            
            # Save individual tow result
            tow_output_file = output_path / f"tow_{tow_num}_complete.json"
            with open(tow_output_file, 'w') as f:
                json.dump(combined_result, f, indent=2)
            
            logger.info(f"Saved combined results for tow {tow_num} to {tow_output_file}")
            all_results.append(combined_result)
        
        # Save all tows combined results
        all_tows_file = output_path / "all_tows_extracted.json"
        with open(all_tows_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save as CSV (flattened structure)
        csv_file = output_path / "all_extractions.csv"
        save_results_as_csv_by_tow(all_results, str(csv_file))
        
        # Create summary report
        create_summary_report_by_tow(all_results, str(output_path))
        
        logger.info(f"Processing complete. Results saved to {output_path}")


def main():
    """Main function to run the extraction."""
    parser = argparse.ArgumentParser(description="Extract data from MOCNESS field sheets")
    parser.add_argument("--input-dir", 
                       default=os.getenv("INPUT_DIR", "./input"),
                       help="Input directory with images")
    parser.add_argument("--output-dir", 
                       default=os.getenv("OUTPUT_DIR", "./output"),
                       help="Output directory for results")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--method", default="all", 
                       choices=["trocr", "layoutlm", "donut", "all"],
                       help="Extraction method to use")
    
    args = parser.parse_args()
    
    logger.info("Starting MOCNESS extraction...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Method: {args.method}")
    
    # Initialize extractor
    extractor = MOCNESSExtractor(device=args.device)
    
    # Process directory
    extractor.process_directory(args.input_dir, args.output_dir, args.method)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
