"""
Utility functions for MOCNESS extraction.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if GPU is available and return device info."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "available": True,
            "device_count": device_count,
            "device_name": device_name,
            "memory_gb": round(memory, 2)
        }
    else:
        return {"available": False}


def estimate_model_requirements():
    """Estimate memory requirements for different models."""
    requirements = {
        "layoutlm": {
            "model_size_gb": 1.2,
            "min_ram_gb": 4,
            "recommended_ram_gb": 8
        },
        "trocr": {
            "model_size_gb": 0.8,
            "min_ram_gb": 2,
            "recommended_ram_gb": 4
        },
        "donut": {
            "model_size_gb": 1.5,
            "min_ram_gb": 4,
            "recommended_ram_gb": 8
        }
    }
    return requirements


def save_results_as_csv(results: List[Dict], output_file: str):
    """Save extraction results as CSV file."""
    if not results:
        logger.warning("No results to save")
        return
    
    # Flatten the nested structure for CSV
    flattened_results = []
    
    for result in results:
        flat_result = {
            "file_path": result.get("file_path", ""),
            "timestamp": result.get("timestamp", ""),
            "error": result.get("error", "")
        }
        
        # Add TrOCR results
        trocr_data = result.get("extraction_methods", {}).get("trocr", {})
        flat_result["trocr_text"] = trocr_data.get("raw_text", "")
        
        # Add LayoutLM results
        layoutlm_data = result.get("extraction_methods", {}).get("layoutlm", {})
        for question, answer in layoutlm_data.items():
            # Clean question for column name
            col_name = f"layoutlm_{question.lower().replace(' ', '_').replace('?', '')}"
            flat_result[col_name] = answer
        
        # Add Donut results
        donut_data = result.get("extraction_methods", {}).get("donut", {})
        flat_result["donut_text"] = donut_data.get("extracted_text", "")
        
        flattened_results.append(flat_result)
    
    # Write CSV
    if flattened_results:
        fieldnames = flattened_results[0].keys()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_results)
        
        logger.info(f"Results saved as CSV: {output_file}")


def create_summary_report(results: List[Dict], output_dir: str):
    """Create a summary report of the extraction process."""
    output_path = Path(output_dir)
    
    # Calculate statistics
    total_files = len(results)
    successful_extractions = len([r for r in results if "error" not in r])
    failed_extractions = total_files - successful_extractions
    
    # Count extraction methods used
    method_counts = {"trocr": 0, "layoutlm": 0, "donut": 0}
    
    for result in results:
        if "extraction_methods" in result:
            for method in method_counts.keys():
                if method in result["extraction_methods"]:
                    method_counts[method] += 1
    
    # Create summary
    summary = {
        "processing_summary": {
            "total_files": total_files,
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "success_rate": round(successful_extractions / total_files * 100, 2) if total_files > 0 else 0
        },
        "extraction_methods": method_counts,
        "files_processed": [r.get("file_path", "") for r in results],
        "errors": [{"file": r.get("file_path", ""), "error": r.get("error", "")} 
                  for r in results if "error" in r]
    }
    
    # Save summary
    summary_file = output_path / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_files}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {summary['processing_summary']['success_rate']}%")
    print("\nExtraction methods used:")
    for method, count in method_counts.items():
        print(f"  {method.upper()}: {count} files")
    
    if summary["errors"]:
        print(f"\nErrors encountered:")
        for error in summary["errors"]:
            print(f"  {error['file']}: {error['error']}")
    print("="*60)


def validate_image_files(input_dir: str) -> Dict[str, List[str]]:
    """Validate that input directory contains expected image files."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    form_files = list(input_path.glob("tow_*_form.png"))
    notes_files = list(input_path.glob("tow_*_notes.png"))
    other_files = [f for f in input_path.glob("*.png") 
                   if f not in form_files and f not in notes_files]
    
    return {
        "form_files": [str(f) for f in form_files],
        "notes_files": [str(f) for f in notes_files],
        "other_png_files": [str(f) for f in other_files],
        "total_mocness_files": len(form_files) + len(notes_files)
    }


def test_model_loading():
    """Test if models can be loaded successfully."""
    logger.info("Testing model loading...")
    
    try:
        # Test a simple pipeline to check if transformers is working
        classifier = pipeline("sentiment-analysis")
        result = classifier("This is a test")
        logger.info("Basic transformers functionality: OK")
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False
