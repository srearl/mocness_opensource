#!/usr/bin/env python3
"""
Demo script to test MOCNESS extraction with sample images.
Creates sample images and runs extraction to verify setup.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

def create_sample_form_image(output_path: str):
    """Create a sample MOCNESS form image for testing."""
    # Create a white background image
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a default font, fall back to basic if not available
        font_large = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
        font_medium = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw form title
    draw.text((50, 30), "MOCNESS FIELD SHEET", fill='black', font=font_large)
    
    # Draw form fields
    y_pos = 100
    line_height = 40
    
    fields = [
        ("Station:", "ST-001"),
        ("Tow Number:", "TOW-001"),
        ("Date:", "2025-07-29"),
        ("Start Time:", "14:30:00"),
        ("End Time:", "15:45:00"),
        ("Latitude:", "42.3601° N"),
        ("Longitude:", "71.0589° W"),
        ("Max Depth:", "500 m"),
        ("Volume Filtered:", "1250 m³"),
        ("Flowmeter Start:", "12345"),
        ("Flowmeter End:", "13595"),
        ("Net Mesh Size:", "200 μm"),
        ("Vessel:", "R/V Atlantis"),
        ("Chief Scientist:", "Dr. Smith"),
    ]
    
    for field, value in fields:
        draw.text((50, y_pos), field, fill='black', font=font_medium)
        draw.text((300, y_pos), value, fill='blue', font=font_medium)
        y_pos += line_height
    
    # Draw notes section
    y_pos += 50
    draw.text((50, y_pos), "Notes:", fill='black', font=font_medium)
    y_pos += 30
    notes = [
        "Clear weather conditions",
        "Calm seas, light wind",
        "All nets deployed successfully",
        "Good sample collection"
    ]
    
    for note in notes:
        draw.text((70, y_pos), f"• {note}", fill='black', font=font_small)
        y_pos += 25
    
    # Draw a simple table for sample data
    y_pos += 50
    draw.text((50, y_pos), "Net Samples:", fill='black', font=font_medium)
    y_pos += 30
    
    # Table header
    draw.rectangle([50, y_pos, width-50, y_pos + 25], outline='black', width=1)
    draw.text((60, y_pos + 5), "Net", fill='black', font=font_small)
    draw.text((150, y_pos + 5), "Depth (m)", fill='black', font=font_small)
    draw.text((250, y_pos + 5), "Volume", fill='black', font=font_small)
    
    # Table rows
    sample_data = [
        ("Net 1", "0-50", "125 m³"),
        ("Net 2", "50-100", "115 m³"),
        ("Net 3", "100-200", "200 m³"),
    ]
    
    y_pos += 25
    for net, depth, volume in sample_data:
        draw.rectangle([50, y_pos, width-50, y_pos + 25], outline='black', width=1)
        draw.text((60, y_pos + 5), net, fill='black', font=font_small)
        draw.text((150, y_pos + 5), depth, fill='black', font=font_small)
        draw.text((250, y_pos + 5), volume, fill='black', font=font_small)
        y_pos += 25
    
    # Save the image
    img.save(output_path)
    print(f"Created sample form: {output_path}")

def create_sample_notes_image(output_path: str):
    """Create a sample MOCNESS notes image for testing."""
    # Create a white background image
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
        font_medium = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    draw.text((50, 30), "MOCNESS FIELD NOTES", fill='black', font=font_large)
    draw.text((50, 60), "Tow 001 - Additional Notes", fill='black', font=font_medium)
    
    # Draw handwritten-style notes
    y_pos = 120
    line_height = 30
    
    notes = [
        "Weather Conditions:",
        "- Wind: 5-10 knots from SW", 
        "- Seas: 1-2 feet",
        "- Visibility: Excellent",
        "",
        "Equipment Notes:",
        "- MOCNESS functioning normally",
        "- All nets closed properly",
        "- Flowmeter readings consistent",
        "",
        "Biological Observations:",
        "- High zooplankton density at 50-100m",
        "- Large copepods visible",
        "- Some gelatinous organisms noted",
        "",
        "Sample Processing:",
        "- All samples preserved in formalin",
        "- Samples labeled and stored",
        "- Digital photos taken",
        "",
        "Issues/Comments:",
        "- None reported",
        "- Successful tow completion",
    ]
    
    for note in notes:
        if note.startswith("-"):
            draw.text((70, y_pos), note, fill='blue', font=font_small)
        elif note == "":
            pass  # Skip blank lines but maintain spacing
        else:
            draw.text((50, y_pos), note, fill='black', font=font_medium)
        y_pos += line_height
    
    # Save the image
    img.save(output_path)
    print(f"Created sample notes: {output_path}")

def create_sample_images():
    """Create sample MOCNESS images for testing."""
    # Create input directory
    input_dir = Path("./input")
    input_dir.mkdir(exist_ok=True)
    
    # Create sample images
    create_sample_form_image(str(input_dir / "tow_001_form.png"))
    create_sample_notes_image(str(input_dir / "tow_001_notes.png"))
    
    # Create a second set
    create_sample_form_image(str(input_dir / "tow_002_form.png"))
    create_sample_notes_image(str(input_dir / "tow_002_notes.png"))
    
    print(f"\nSample images created in {input_dir}")
    print("You can now run the extraction with:")
    print("python main.py --input-dir ./input --output-dir ./output")

if __name__ == "__main__":
    create_sample_images()
