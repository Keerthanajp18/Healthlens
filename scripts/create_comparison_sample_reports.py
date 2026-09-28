"""
Helper script to generate three synthetic validated reports for the same patient
across three different dates (January 2024, April 2024, and July 2024) to test
Stage 6 comparison and Stage 7 longitudinal trend analysis.
"""

from pathlib import Path
import sys
import json

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


def generate_comparison_sample_reports():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # REPORT 1: 15-Jan-2024 (Baseline)
    # ----------------------------------------------------
    report_1 = {
        "metadata": {
            "source_file": "sarah_jenkins_cbc_lipid_2024_01.pdf",
            "report_date": "15-Jan-2024",
            "total_tests": 7,
            "status_summary": {
                "NORMAL": 4,
                "LOW": 1,
                "HIGH": 2,
                "UNKNOWN": 0
            }
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 10.2,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jan-2024",
                "status": "LOW",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
                "reference_type": "RANGE"
            },
            {
                "test_name": "RBC Count",
                "value": 3.80,
                "unit": "million/mcL",
                "reference_range": "3.80 - 5.10",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jan-2024",
                "status": "NORMAL",
                "lower_bound": 3.80,
                "upper_bound": 5.10,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Leukocyte Count (WBC)",
                "value": 11800,
                "unit": "cells/mcL",
                "reference_range": "4000 - 11000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jan-2024",
                "status": "HIGH",
                "lower_bound": 4000.0,
                "upper_bound": 11000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Platelet Count",
                "value": 250000,
                "unit": "cells/mcL",
                "reference_range": "150000 - 450000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jan-2024",
                "status": "NORMAL",
                "lower_bound": 150000.0,
                "upper_bound": 450000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Cholesterol",
                "value": 230,
                "unit": "mg/dL",
                "reference_range": "< 200 Desirable",
                "category": "Lipid Profile",
                "report_date": "15-Jan-2024",
                "status": "HIGH",
                "lower_bound": None,
                "upper_bound": 200.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "Triglycerides",
                "value": 175,
                "unit": "mg/dL",
                "reference_range": "< 150 Normal",
                "category": "Lipid Profile",
                "report_date": "15-Jan-2024",
                "status": "HIGH",
                "lower_bound": None,
                "upper_bound": 150.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "Fasting Blood Sugar",
                "value": 92,
                "unit": "mg/dL",
                "reference_range": "70 - 99",
                "category": "Biochemistry",
                "report_date": "15-Jan-2024",
                "status": "NORMAL",
                "lower_bound": 70.0,
                "upper_bound": 99.0,
                "reference_type": "RANGE"
            }
        ]
    }

    # ----------------------------------------------------
    # REPORT 2: 20-Apr-2024 (3-Month Follow-up)
    # ----------------------------------------------------
    report_2 = {
        "metadata": {
            "source_file": "sarah_jenkins_cbc_lipid_2024_04.pdf",
            "report_date": "20-Apr-2024",
            "total_tests": 7,
            "status_summary": {
                "NORMAL": 5,
                "LOW": 1,
                "HIGH": 1,
                "UNKNOWN": 0
            }
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 11.5,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "category": "Complete Blood Count (CBC)",
                "report_date": "20-Apr-2024",
                "status": "LOW",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
                "reference_type": "RANGE"
            },
            {
                "test_name": "RBC Count",
                "value": 3.95,
                "unit": "million/mcL",
                "reference_range": "3.80 - 5.10",
                "category": "Complete Blood Count (CBC)",
                "report_date": "20-Apr-2024",
                "status": "NORMAL",
                "lower_bound": 3.80,
                "upper_bound": 5.10,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Leukocyte Count (WBC)",
                "value": 9400,
                "unit": "cells/mcL",
                "reference_range": "4000 - 11000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "20-Apr-2024",
                "status": "NORMAL",
                "lower_bound": 4000.0,
                "upper_bound": 11000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Platelet Count",
                "value": 250000,
                "unit": "cells/mcL",
                "reference_range": "150000 - 450000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "20-Apr-2024",
                "status": "NORMAL",
                "lower_bound": 150000.0,
                "upper_bound": 450000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Cholesterol",
                "value": 210,
                "unit": "mg/dL",
                "reference_range": "< 200 Desirable",
                "category": "Lipid Profile",
                "report_date": "20-Apr-2024",
                "status": "HIGH",
                "lower_bound": None,
                "upper_bound": 200.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "Triglycerides",
                "value": 145,
                "unit": "mg/dL",
                "reference_range": "< 150 Normal",
                "category": "Lipid Profile",
                "report_date": "20-Apr-2024",
                "status": "NORMAL",
                "lower_bound": None,
                "upper_bound": 150.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "HbA1c (Glycated Hemoglobin)",
                "value": 5.4,
                "unit": "%",
                "reference_range": "< 5.7",
                "category": "Biochemistry",
                "report_date": "20-Apr-2024",
                "status": "NORMAL",
                "lower_bound": None,
                "upper_bound": 5.7,
                "reference_type": "LESS_THAN"
            }
        ]
    }

    # ----------------------------------------------------
    # REPORT 3: 15-Jul-2024 (6-Month Follow-up)
    # ----------------------------------------------------
    report_3 = {
        "metadata": {
            "source_file": "sarah_jenkins_cbc_lipid_2024_07.pdf",
            "report_date": "15-Jul-2024",
            "total_tests": 7,
            "status_summary": {
                "NORMAL": 7,
                "LOW": 0,
                "HIGH": 0,
                "UNKNOWN": 0
            }
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 12.8,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
                "reference_type": "RANGE"
            },
            {
                "test_name": "RBC Count",
                "value": 4.20,
                "unit": "million/mcL",
                "reference_range": "3.80 - 5.10",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": 3.80,
                "upper_bound": 5.10,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Leukocyte Count (WBC)",
                "value": 8100,
                "unit": "cells/mcL",
                "reference_range": "4000 - 11000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": 4000.0,
                "upper_bound": 11000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Platelet Count",
                "value": 255000,
                "unit": "cells/mcL",
                "reference_range": "150000 - 450000",
                "category": "Complete Blood Count (CBC)",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": 150000.0,
                "upper_bound": 450000.0,
                "reference_type": "RANGE"
            },
            {
                "test_name": "Total Cholesterol",
                "value": 195,
                "unit": "mg/dL",
                "reference_range": "< 200 Desirable",
                "category": "Lipid Profile",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": None,
                "upper_bound": 200.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "Triglycerides",
                "value": 135,
                "unit": "mg/dL",
                "reference_range": "< 150 Normal",
                "category": "Lipid Profile",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": None,
                "upper_bound": 150.0,
                "reference_type": "LESS_THAN"
            },
            {
                "test_name": "HbA1c (Glycated Hemoglobin)",
                "value": 5.3,
                "unit": "%",
                "reference_range": "< 5.7",
                "category": "Biochemistry",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": None,
                "upper_bound": 5.7,
                "reference_type": "LESS_THAN"
            }
        ]
    }

    path_1 = PROCESSED_DATA_DIR / "sample_patient_report_2024_01_validated.json"
    path_2 = PROCESSED_DATA_DIR / "sample_patient_report_2024_04_validated.json"
    path_3 = PROCESSED_DATA_DIR / "sample_patient_report_2024_07_validated.json"

    with open(path_1, "w", encoding="utf-8") as f:
        json.dump(report_1, f, indent=2)

    with open(path_2, "w", encoding="utf-8") as f:
        json.dump(report_2, f, indent=2)

    with open(path_3, "w", encoding="utf-8") as f:
        json.dump(report_3, f, indent=2)

    print(f"Created: {path_1}")
    print(f"Created: {path_2}")
    print(f"Created: {path_3}")


if __name__ == "__main__":
    generate_comparison_sample_reports()
