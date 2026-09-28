"""
Validation Submodule
Responsible for:
- Deterministic validation of extracted test values against biological reference ranges
- Categorizing test findings into 'NORMAL', 'LOW', 'HIGH', or 'UNKNOWN'
- Ground-truth evaluation based exclusively on laboratory-reported ranges
"""

from src.validation.validator import MedicalTestValidator, validate_medical_tests

__all__ = ["MedicalTestValidator", "validate_medical_tests"]
