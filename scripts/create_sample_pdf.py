"""
Helper script to generate a synthetic, 2-page digital PDF medical report.
Uses ReportLab to produce a standard digital PDF containing searchable text streams.
No real patient data is used.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Destination path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PDF = PROJECT_ROOT / "data" / "sample_reports" / "sample_medical_report.pdf"


def generate_sample_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#1A365D"),
    )
    sub_title_style = ParagraphStyle(
        "SubHeader",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#4A5568"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
    )
    cell_style = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
    )

    story = []

    # ================= PAGE 1: DEMOGRAPHICS + COMPLETE BLOOD COUNT =================
    story.append(Paragraph("HEALTHLENS DIAGNOSTIC LABORATORY", title_style))
    story.append(Paragraph("Accredited Medical Testing & Pathology Services | Synthetic Test Report", sub_title_style))
    story.append(Spacer(1, 10))

    # Demographics Table
    demo_data = [
        [Paragraph("<b>Patient Name:</b> Sarah Jenkins", cell_style), Paragraph("<b>Age / Gender:</b> 35 Y / Female", cell_style)],
        [Paragraph("<b>Patient ID:</b> HL-2024-5501", cell_style), Paragraph("<b>Referred By:</b> Dr. David Miller, MD", cell_style)],
        [Paragraph("<b>Sample ID:</b> SMP-77102", cell_style), Paragraph("<b>Collection Date:</b> 10-Feb-2024 08:15 AM", cell_style)],
        [Paragraph("<b>Sample Type:</b> Whole Blood (EDTA)", cell_style), Paragraph("<b>Reporting Date:</b> 10-Feb-2024 01:30 PM", cell_style)],
    ]
    t_demo = Table(demo_data, colWidths=[260, 260])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 12))

    story.append(Paragraph("DEPARTMENT OF HEMATOLOGY: COMPLETE BLOOD COUNT (CBC)", section_style))
    story.append(Spacer(1, 6))

    cbc_data = [
        [Paragraph("Investigation / Test", cell_bold), Paragraph("Observed Value", cell_bold), Paragraph("Unit", cell_bold), Paragraph("Biological Reference Range", cell_bold)],
        [Paragraph("Hemoglobin (Hb)", cell_style), Paragraph("11.5", cell_style), Paragraph("g/dL", cell_style), Paragraph("12.0 - 15.5", cell_style)],
        [Paragraph("RBC Count", cell_style), Paragraph("3.95", cell_style), Paragraph("million/mcL", cell_style), Paragraph("3.80 - 5.10", cell_style)],
        [Paragraph("Packed Cell Volume (PCV)", cell_style), Paragraph("35.2", cell_style), Paragraph("%", cell_style), Paragraph("36.0 - 46.0", cell_style)],
        [Paragraph("Mean Corpuscular Volume (MCV)", cell_style), Paragraph("89.1", cell_style), Paragraph("fL", cell_style), Paragraph("80.0 - 100.0", cell_style)],
        [Paragraph("MCH", cell_style), Paragraph("29.1", cell_style), Paragraph("pg", cell_style), Paragraph("27.0 - 33.0", cell_style)],
        [Paragraph("MCHC", cell_style), Paragraph("32.6", cell_style), Paragraph("g/dL", cell_style), Paragraph("31.5 - 35.0", cell_style)],
        [Paragraph("RDW", cell_style), Paragraph("13.4", cell_style), Paragraph("%", cell_style), Paragraph("11.5 - 15.0", cell_style)],
        [Paragraph("Total Leukocyte Count (WBC)", cell_style), Paragraph("11200", cell_style), Paragraph("cells/mcL", cell_style), Paragraph("4000 - 11000", cell_style)],
        [Paragraph("Platelet Count", cell_style), Paragraph("260000", cell_style), Paragraph("cells/mcL", cell_style), Paragraph("150000 - 450000", cell_style)],
        [Paragraph("Neutrophils", cell_style), Paragraph("72", cell_style), Paragraph("%", cell_style), Paragraph("40 - 70", cell_style)],
        [Paragraph("Lymphocytes", cell_style), Paragraph("22", cell_style), Paragraph("%", cell_style), Paragraph("20 - 40", cell_style)],
        [Paragraph("Monocytes", cell_style), Paragraph("4", cell_style), Paragraph("%", cell_style), Paragraph("2 - 8", cell_style)],
        [Paragraph("Eosinophils", cell_style), Paragraph("2", cell_style), Paragraph("%", cell_style), Paragraph("1 - 6", cell_style)],
    ]
    t_cbc = Table(cbc_data, colWidths=[200, 100, 90, 130])
    t_cbc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cbc)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Page 1 of 2 -- HealthLens Complete Blood Count Segment", sub_title_style))

    # ================= PAGE 2: LIPID PROFILE + CLINICAL NOTES =================
    story.append(PageBreak())

    story.append(Paragraph("HEALTHLENS DIAGNOSTIC LABORATORY", title_style))
    story.append(Paragraph("Accredited Medical Testing & Pathology Services | Synthetic Test Report", sub_title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("DEPARTMENT OF BIOCHEMISTRY: LIPID PROFILE", section_style))
    story.append(Spacer(1, 6))

    lipid_data = [
        [Paragraph("Investigation / Test", cell_bold), Paragraph("Observed Value", cell_bold), Paragraph("Unit", cell_bold), Paragraph("Biological Reference Range", cell_bold)],
        [Paragraph("Total Cholesterol", cell_style), Paragraph("215", cell_style), Paragraph("mg/dL", cell_style), Paragraph("< 200 Desirable", cell_style)],
        [Paragraph("Triglycerides", cell_style), Paragraph("165", cell_style), Paragraph("mg/dL", cell_style), Paragraph("< 150 Normal", cell_style)],
        [Paragraph("HDL Cholesterol", cell_style), Paragraph("48", cell_style), Paragraph("mg/dL", cell_style), Paragraph("> 50 Optimal", cell_style)],
        [Paragraph("LDL Cholesterol", cell_style), Paragraph("134", cell_style), Paragraph("mg/dL", cell_style), Paragraph("< 100 Optimal", cell_style)],
        [Paragraph("VLDL Cholesterol", cell_style), Paragraph("33", cell_style), Paragraph("mg/dL", cell_style), Paragraph("5 - 30", cell_style)],
        [Paragraph("Cholesterol / HDL Ratio", cell_style), Paragraph("4.48", cell_style), Paragraph("Ratio", cell_style), Paragraph("3.3 - 4.4", cell_style)],
    ]
    t_lipid = Table(lipid_data, colWidths=[200, 100, 90, 130])
    t_lipid.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_lipid)
    story.append(Spacer(1, 15))

    story.append(Paragraph("CLINICAL IMPRESSIONS & LAB NOTES", section_style))
    story.append(Spacer(1, 4))
    notes = [
        "- Mild borderline anemia indicated by slightly decreased Hemoglobin and Hematocrit.",
        "- Mild neutrophilia / elevated total WBC count noted; recommend correlation for mild reactive inflammation.",
        "- Borderline elevation in Total Cholesterol and LDL. Lifestyle modification and dietary counseling suggested.",
        "- NOTE: This is a synthetic computer-generated report created exclusively for HealthLens capstone research.",
    ]
    for n in notes:
        story.append(Paragraph(n, cell_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 15))
    story.append(Paragraph("Page 2 of 2 -- End of Diagnostic Laboratory Report", sub_title_style))

    doc.build(story)
    print(f"Sample PDF created successfully at: {output_path}")


if __name__ == "__main__":
    generate_sample_pdf(OUTPUT_PDF)
