"""
Helper script to generate synthetic medical report images (PNG, JPG)
and a synthetic scanned PDF (image-only, zero digital text stream).
Uses Pillow and pypdf to produce realistic scanned diagnostic reports.
No real patient data is used.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypdf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_reports"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def generate_sample_images_and_scanned_pdf():
    # 1. Create a high-resolution report image using Pillow
    width, height = 1200, 1600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Use default font or truetype if available
    try:
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_subtitle = ImageFont.truetype("arial.ttf", 22)
        font_body_bold = ImageFont.truetype("arialbd.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_subtitle = font_title
        font_body_bold = font_title
        font_body = font_title

    # Header
    draw.text((width // 2, 60), "HEALTHLENS CLINICAL LABORATORY", fill="#1A365D", font=font_title, anchor="mt")
    draw.text((width // 2, 110), "Diagnostic Pathology Services | Synthetic Scanned Test Report", fill="#4A5568", font=font_subtitle, anchor="mt")
    draw.line([(60, 160), (width - 60, 160)], fill="#CBD5E0", width=2)

    # Demographics Box
    draw.rectangle([(60, 180), (width - 60, 320)], fill="#F7FAFC", outline="#CBD5E0", width=2)
    draw.text((80, 200), "Patient Name : Michael Chang", fill="#2D3748", font=font_body)
    draw.text((650, 200), "Age / Gender : 45 Y / Male", fill="#2D3748", font=font_body)
    draw.text((80, 240), "Patient ID   : HL-2024-8802", fill="#2D3748", font=font_body)
    draw.text((650, 240), "Referred By  : Dr. Sarah Bennett, MD", fill="#2D3748", font=font_body)
    draw.text((80, 280), "Collection Date : 12-Mar-2024 09:00 AM", fill="#2D3748", font=font_body)
    draw.text((650, 280), "Reporting Date  : 12-Mar-2024 03:00 PM", fill="#2D3748", font=font_body)

    # Section Header
    draw.text((60, 350), "DEPARTMENT OF HEMATOLOGY: COMPLETE BLOOD COUNT (CBC)", fill="#2B6CB0", font=font_body_bold)
    draw.line([(60, 390), (width - 60, 390)], fill="#2B6CB0", width=2)

    # Table Columns Header
    draw.rectangle([(60, 400), (width - 60, 445)], fill="#EDF2F7")
    draw.text((80, 412), "Investigation / Test", fill="#1A202C", font=font_body_bold)
    draw.text((500, 412), "Observed Value", fill="#1A202C", font=font_body_bold)
    draw.text((750, 412), "Unit", fill="#1A202C", font=font_body_bold)
    draw.text((920, 412), "Reference Range", fill="#1A202C", font=font_body_bold)

    # Table Rows
    rows = [
        ("Hemoglobin (Hb)", "13.8", "g/dL", "13.0 - 17.0"),
        ("RBC Count", "4.60", "million/mcL", "4.50 - 5.90"),
        ("Packed Cell Volume (PCV)", "42.1", "%", "40.0 - 50.0"),
        ("Mean Corpuscular Volume (MCV)", "88.5", "fL", "80.0 - 100.0"),
        ("MCH", "29.4", "pg", "27.0 - 33.0"),
        ("MCHC", "33.2", "g/dL", "31.5 - 35.0"),
        ("RDW", "13.1", "%", "11.5 - 15.0"),
        ("Total Leukocyte Count (WBC)", "7800", "cells/mcL", "4000 - 11000"),
        ("Platelet Count", "240000", "cells/mcL", "150000 - 450000"),
        ("Neutrophils", "64", "%", "40 - 70"),
        ("Lymphocytes", "28", "%", "20 - 40"),
        ("Monocytes", "5", "%", "2 - 8"),
        ("Eosinophils", "3", "%", "1 - 6"),
    ]

    y = 460
    for name, val, unit, ref in rows:
        draw.text((80, y), name, fill="#2D3748", font=font_body)
        draw.text((530, y), val, fill="#2D3748", font=font_body)
        draw.text((750, y), unit, fill="#2D3748", font=font_body)
        draw.text((920, y), ref, fill="#2D3748", font=font_body)
        draw.line([(60, y + 36), (width - 60, y + 36)], fill="#E2E8F0", width=1)
        y += 48

    # Clinical Notes
    y += 40
    draw.text((60, y), "CLINICAL IMPRESSIONS & LAB NOTES:", fill="#2B6CB0", font=font_body_bold)
    y += 35
    draw.text((60, y), "- All hematological parameters are within normal biological limits for adult male.", fill="#4A5568", font=font_body)
    y += 30
    draw.text((60, y), "- Synthetic test document generated for HealthLens capstone testing.", fill="#4A5568", font=font_body)

    # Save PNG
    png_path = SAMPLE_DIR / "sample_medical_report_image.png"
    img.save(png_path)
    print(f"Saved synthetic report PNG at: {png_path}")

    # Save JPG
    jpg_path = SAMPLE_DIR / "sample_medical_report_image.jpg"
    img.save(jpg_path, quality=95)
    print(f"Saved synthetic report JPG at: {jpg_path}")

    # Save Scanned PDF (PDF containing the image with NO text stream)
    pdf_path = SAMPLE_DIR / "sample_scanned_report.pdf"
    img.save(pdf_path, "PDF", resolution=100.0)
    print(f"Saved synthetic scanned PDF at: {pdf_path}")


if __name__ == "__main__":
    generate_sample_images_and_scanned_pdf()
