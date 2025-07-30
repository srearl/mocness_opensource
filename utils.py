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


def save_results_as_csv_by_tow(results: List[Dict], output_file: str):
    """Save tow-based extraction results as CSV file."""
    if not results:
        logger.warning("No results to save")
        return
    
    # Flatten the nested structure for CSV
    flattened_results = []
    
    for tow_result in results:
        tow_num = tow_result.get("tow_number", "")
        timestamp = tow_result.get("timestamp", "")
        
        # Create base row for this tow
        base_row = {
            "tow_number": tow_num,
            "timestamp": timestamp
        }
        
        # Process form data
        form_data = tow_result.get("form_data", {})
        if form_data.get("extraction_methods"):
            form_row = base_row.copy()
            form_row["data_type"] = "form"
            form_row["file_path"] = form_data.get("file_path", "")
            
            # Add TrOCR results
            trocr_data = form_data.get("extraction_methods", {}).get("trocr", {})
            form_row["trocr_text"] = trocr_data.get("raw_text", "")
            
            # Add LayoutLM results
            layoutlm_data = form_data.get("extraction_methods", {}).get("layoutlm", {})
            for question, answer in layoutlm_data.items():
                col_name = f"layoutlm_{question.lower().replace(' ', '_').replace('?', '').replace(',', '').replace('(', '').replace(')', '')[:50]}"
                form_row[col_name] = answer
            
            # Add Donut results
            donut_data = form_data.get("extraction_methods", {}).get("donut", {})
            form_row["donut_text"] = donut_data.get("extracted_text", "")
            
            flattened_results.append(form_row)
        
        # Process notes data
        notes_data = tow_result.get("notes_data", {})
        if notes_data.get("extraction_methods"):
            notes_row = base_row.copy()
            notes_row["data_type"] = "notes"
            notes_row["file_path"] = notes_data.get("file_path", "")
            
            # Add TrOCR results
            trocr_data = notes_data.get("extraction_methods", {}).get("trocr", {})
            notes_row["trocr_text"] = trocr_data.get("raw_text", "")
            
            # Add LayoutLM results
            layoutlm_data = notes_data.get("extraction_methods", {}).get("layoutlm", {})
            for question, answer in layoutlm_data.items():
                col_name = f"layoutlm_{question.lower().replace(' ', '_').replace('?', '').replace(',', '').replace('(', '').replace(')', '')[:50]}"
                notes_row[col_name] = answer
            
            # Add Donut results
            donut_data = notes_data.get("extraction_methods", {}).get("donut", {})
            notes_row["donut_text"] = donut_data.get("extracted_text", "")
            
            flattened_results.append(notes_row)
    
    # Write CSV
    if flattened_results:
        # Get all possible fieldnames
        all_fieldnames = set()
        for row in flattened_results:
            all_fieldnames.update(row.keys())
        
        fieldnames = sorted(list(all_fieldnames))
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_results)
        
        logger.info(f"Results saved as CSV: {output_file}")


def create_summary_report_by_tow(results: List[Dict], output_dir: str):
    """Create a summary report for tow-based results."""
    summary = {
        "timestamp": results[0]["timestamp"] if results else "",
        "total_tows": len(results),
        "processing_summary": {
            "tows_with_form": 0,
            "tows_with_notes": 0,
            "tows_with_both": 0,
            "total_files_processed": 0
        },
        "extraction_methods": {
            "trocr": {"success": 0, "errors": 0},
            "layoutlm": {"success": 0, "errors": 0},
            "donut": {"success": 0, "errors": 0}
        },
        "errors": []
    }
    
    for tow_result in results:
        tow_num = tow_result.get("tow_number", "")
        has_form = bool(tow_result.get("form_data", {}).get("extraction_methods"))
        has_notes = bool(tow_result.get("notes_data", {}).get("extraction_methods"))
        
        if has_form:
            summary["processing_summary"]["tows_with_form"] += 1
            summary["processing_summary"]["total_files_processed"] += 1
            
            # Check form extraction methods
            form_methods = tow_result["form_data"].get("extraction_methods", {})
            for method in ["trocr", "layoutlm", "donut"]:
                if method in form_methods:
                    if form_methods[method]:  # Has content
                        summary["extraction_methods"][method]["success"] += 1
                    else:
                        summary["extraction_methods"][method]["errors"] += 1
        
        if has_notes:
            summary["processing_summary"]["tows_with_notes"] += 1
            summary["processing_summary"]["total_files_processed"] += 1
            
            # Check notes extraction methods
            notes_methods = tow_result["notes_data"].get("extraction_methods", {})
            for method in ["trocr", "layoutlm", "donut"]:
                if method in notes_methods:
                    if notes_methods[method]:  # Has content
                        summary["extraction_methods"][method]["success"] += 1
                    else:
                        summary["extraction_methods"][method]["errors"] += 1
        
        if has_form and has_notes:
            summary["processing_summary"]["tows_with_both"] += 1
        
        # Collect errors
        for data_type in ["form_data", "notes_data"]:
            data = tow_result.get(data_type, {})
            if data.get("error"):
                summary["errors"].append({
                    "tow": tow_num,
                    "type": data_type.replace("_data", ""),
                    "error": data["error"]
                })
    
    # Save summary report
    summary_file = Path(output_dir) / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved: {summary_file}")
    
    # Print summary to console
    print("="*60)
    print("MOCNESS EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total tows processed: {summary['total_tows']}")
    print(f"Total files processed: {summary['processing_summary']['total_files_processed']}")
    print(f"Tows with forms: {summary['processing_summary']['tows_with_form']}")
    print(f"Tows with notes: {summary['processing_summary']['tows_with_notes']}")
    print(f"Tows with both form and notes: {summary['processing_summary']['tows_with_both']}")
    
    print("\nExtraction method performance:")
    for method, stats in summary["extraction_methods"].items():
        total = stats["success"] + stats["errors"]
        if total > 0:
            success_rate = (stats["success"] / total) * 100
            print(f"  {method.upper()}: {stats['success']}/{total} ({success_rate:.1f}% success)")
    
    if summary["errors"]:
        print(f"\nErrors encountered:")
        for error in summary["errors"]:
            print(f"  Tow {error['tow']} ({error['type']}): {error['error']}")
    print("="*60)
