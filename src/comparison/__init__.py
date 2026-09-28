"""
Comparison & Trend Analysis Submodule
Responsible for:
- Aligning and comparing test values across multiple reports for the same patient (Stage 6)
- Calculating absolute and percentage changes over time
- Handling unit verification and missing test scenarios
- Visualizing longitudinal health trends with charts (Stage 7)
"""

from src.comparison.report_comparator import (
    MedicalReportComparator,
    compare_medical_reports,
    parse_date,
)
from src.comparison.trend_analyzer import (
    MedicalTrendAnalyzer,
    analyze_medical_trends,
)
from src.comparison.visualizer import (
    MedicalTrendVisualizer,
    visualize_medical_trends,
)

__all__ = [
    "MedicalReportComparator",
    "compare_medical_reports",
    "parse_date",
    "MedicalTrendAnalyzer",
    "analyze_medical_trends",
    "MedicalTrendVisualizer",
    "visualize_medical_trends",
]

