"""
Q-Diagnose Lab Report Extractor & Reference Review
==================================================
Parses supported clinical lab report files, extracts recognizable biomarkers,
and compares values with general reference intervals. Disease-specific patient
predictions are unavailable until validated models are configured.
"""

import io
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Full dictionary of supported biomarkers (Section B)
BIOMARKER_DEFINITIONS = {
    # 1. Glycemic & Metabolic
    "fasting_glucose": {
        "label": "Fasting Blood Glucose",
        "unit": "mg/dL",
        "normal_min": 70.0,
        "normal_max": 99.0,
        "borderline_max": 125.0,
        "category": "Metabolic",
        "synonyms": ["fasting blood sugar", "fasting blood glucose", "fasting glucose", "fbs", "fbg", "blood sugar", "glucose fasting", "glucose", "b sugar"]
    },
    "postprandial_glucose": {
        "label": "Post-Prandial Blood Glucose (PPBS)",
        "unit": "mg/dL",
        "normal_min": 70.0,
        "normal_max": 139.0,
        "borderline_max": 199.0,
        "category": "Metabolic",
        "synonyms": ["postprandial blood glucose", "post prandial blood glucose", "post prandial glucose", "postprandial glucose", "ppbs", "pp glucose", "blood sugar pp", "2hr postprandial"]
    },
    "hba1c": {
        "label": "Glycated Hemoglobin (HbA1c)",
        "unit": "%",
        "normal_min": 4.0,
        "normal_max": 5.6,
        "borderline_max": 6.4,
        "category": "Metabolic",
        "synonyms": ["hba1c", "a1c", "glycated hemoglobin", "glycohemoglobin", "hemoglobin a1c"]
    },
    "insulin": {
        "label": "Fasting Serum Insulin",
        "unit": "uIU/mL",
        "normal_min": 2.6,
        "normal_max": 24.9,
        "borderline_max": 35.0,
        "category": "Metabolic",
        "synonyms": ["insulin", "fasting insulin", "serum insulin", "fasting serum insulin"]
    },
    "c_peptide": {
        "label": "C-Peptide",
        "unit": "ng/mL",
        "normal_min": 1.1,
        "normal_max": 4.4,
        "borderline_max": 5.0,
        "category": "Metabolic",
        "synonyms": ["c-peptide", "c peptide", "cpeptide"]
    },

    # 2. Cardiovascular & Lipid Profile
    "total_cholesterol": {
        "label": "Total Serum Cholesterol",
        "unit": "mg/dL",
        "normal_min": 125.0,
        "normal_max": 199.0,
        "borderline_max": 239.0,
        "category": "Cardiovascular",
        "synonyms": ["total serum cholesterol", "total cholesterol", "serum cholesterol", "cholesterol", "tc", "chol"]
    },
    "ldl_cholesterol": {
        "label": "LDL Cholesterol (Bad)",
        "unit": "mg/dL",
        "normal_min": 50.0,
        "normal_max": 99.0,
        "borderline_max": 159.0,
        "category": "Cardiovascular",
        "synonyms": ["ldl", "ldl cholesterol", "ldl-c", "low density lipoprotein"]
    },
    "hdl_cholesterol": {
        "label": "HDL Cholesterol (Good)",
        "unit": "mg/dL",
        "normal_min": 40.0,
        "normal_max": 80.0,
        "borderline_max": 90.0,
        "category": "Cardiovascular",
        "synonyms": ["hdl", "hdl cholesterol", "hdl-c", "high density lipoprotein"]
    },
    "triglycerides": {
        "label": "Serum Triglycerides",
        "unit": "mg/dL",
        "normal_min": 50.0,
        "normal_max": 149.0,
        "borderline_max": 199.0,
        "category": "Cardiovascular",
        "synonyms": ["triglycerides", "tg", "trigs", "serum triglycerides"]
    },
    "vldl": {
        "label": "Very Low-Density Lipoprotein (VLDL)",
        "unit": "mg/dL",
        "normal_min": 5.0,
        "normal_max": 30.0,
        "borderline_max": 40.0,
        "category": "Cardiovascular",
        "synonyms": ["vldl", "vldl cholesterol", "vldl-c"]
    },
    "systolic_bp": {
        "label": "Systolic Blood Pressure",
        "unit": "mmHg",
        "normal_min": 90.0,
        "normal_max": 120.0,
        "borderline_max": 139.0,
        "category": "Cardiovascular",
        "synonyms": ["systolic blood pressure", "blood pressure systolic", "systolic bp", "systolic", "sbp", "bp sys"]
    },
    "diastolic_bp": {
        "label": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "normal_min": 60.0,
        "normal_max": 80.0,
        "borderline_max": 89.0,
        "category": "Cardiovascular",
        "synonyms": ["diastolic blood pressure", "blood pressure diastolic", "diastolic bp", "diastolic", "dbp", "bp dia"]
    },
    "troponin_i": {
        "label": "Cardiac Troponin I",
        "unit": "ng/mL",
        "normal_min": 0.0,
        "normal_max": 0.04,
        "borderline_max": 0.10,
        "category": "Cardiovascular",
        "synonyms": ["cardiac troponin i", "cardiac troponin", "troponin i", "troponin-i", "troponin", "ctni"]
    },
    "bnp": {
        "label": "B-Type Natriuretic Peptide",
        "unit": "pg/mL",
        "normal_min": 0.0,
        "normal_max": 100.0,
        "borderline_max": 300.0,
        "category": "Cardiovascular",
        "synonyms": ["bnp", "brain natriuretic peptide", "nt-probnp"]
    },
    "hs_crp": {
        "label": "High-Sensitivity CRP",
        "unit": "mg/L",
        "normal_min": 0.0,
        "normal_max": 1.0,
        "borderline_max": 3.0,
        "category": "Cardiovascular",
        "synonyms": ["high-sensitivity crp", "high sensitivity crp", "hs-crp", "hscrp", "cardiac crp"]
    },

    # 3. Renal / Kidney Function
    "serum_creatinine": {
        "label": "Serum Creatinine",
        "unit": "mg/dL",
        "normal_min": 0.6,
        "normal_max": 1.2,
        "borderline_max": 1.5,
        "category": "Renal",
        "synonyms": ["creatinine", "serum creatinine", "creat", "s. creatinine", "cr"]
    },
    "bun": {
        "label": "Blood Urea Nitrogen (BUN)",
        "unit": "mg/dL",
        "normal_min": 7.0,
        "normal_max": 20.0,
        "borderline_max": 28.0,
        "category": "Renal",
        "synonyms": ["bun", "urea nitrogen", "blood urea nitrogen", "blood urea", "urea"]
    },
    "egfr": {
        "label": "Estimated GFR (eGFR)",
        "unit": "mL/min/1.73m²",
        "normal_min": 90.0,
        "normal_max": 130.0,
        "borderline_max": 89.0,
        "category": "Renal",
        "synonyms": ["egfr", "gfr", "estimated gfr", "glomerular filtration rate"]
    },
    "uric_acid": {
        "label": "Serum Uric Acid",
        "unit": "mg/dL",
        "normal_min": 3.5,
        "normal_max": 7.2,
        "borderline_max": 8.5,
        "category": "Renal",
        "synonyms": ["uric acid", "serum uric acid", "urate"]
    },

    # 4. Hepatic / Liver Function
    "alt": {
        "label": "Alanine Aminotransferase (ALT / SGPT)",
        "unit": "U/L",
        "normal_min": 7.0,
        "normal_max": 56.0,
        "borderline_max": 80.0,
        "category": "Hepatic",
        "synonyms": ["alt", "sgpt", "alanine aminotransferase", "alanine transaminase"]
    },
    "ast": {
        "label": "Aspartate Aminotransferase (AST / SGOT)",
        "unit": "U/L",
        "normal_min": 10.0,
        "normal_max": 40.0,
        "borderline_max": 70.0,
        "category": "Hepatic",
        "synonyms": ["ast", "sgot", "aspartate aminotransferase", "aspartate transaminase"]
    },
    "alp": {
        "label": "Alkaline Phosphatase (ALP)",
        "unit": "U/L",
        "normal_min": 44.0,
        "normal_max": 147.0,
        "borderline_max": 180.0,
        "category": "Hepatic",
        "synonyms": ["alp", "alkaline phosphatase", "alk phos"]
    },
    "total_bilirubin": {
        "label": "Total Bilirubin",
        "unit": "mg/dL",
        "normal_min": 0.2,
        "normal_max": 1.2,
        "borderline_max": 1.8,
        "category": "Hepatic",
        "synonyms": ["total bilirubin", "bilirubin total", "tbili", "t. bilirubin", "bilirubin"]
    },
    "direct_bilirubin": {
        "label": "Direct Bilirubin (Conjugated)",
        "unit": "mg/dL",
        "normal_min": 0.0,
        "normal_max": 0.3,
        "borderline_max": 0.5,
        "category": "Hepatic",
        "synonyms": ["direct bilirubin", "conjugated bilirubin", "dbili"]
    },
    "albumin": {
        "label": "Serum Albumin",
        "unit": "g/dL",
        "normal_min": 3.5,
        "normal_max": 5.5,
        "borderline_max": 3.4,
        "category": "Hepatic",
        "synonyms": ["albumin", "serum albumin", "alb"]
    },
    "total_protein": {
        "label": "Total Serum Protein",
        "unit": "g/dL",
        "normal_min": 6.0,
        "normal_max": 8.3,
        "borderline_max": 8.8,
        "category": "Hepatic",
        "synonyms": ["total protein", "serum protein", "tp"]
    },

    # 5. Complete Blood Count (CBC) & Hematology
    "hemoglobin": {
        "label": "Hemoglobin (Hb)",
        "unit": "g/dL",
        "normal_min": 12.0,
        "normal_max": 17.5,
        "borderline_max": 11.9,
        "category": "Hematology",
        "synonyms": ["hemoglobin", "haemoglobin", "hb", "hgb"]
    },
    "wbc_count": {
        "label": "White Blood Cell Count (WBC)",
        "unit": "x10^3/uL",
        "normal_min": 4.0,
        "normal_max": 11.0,
        "borderline_max": 13.5,
        "category": "Hematology",
        "synonyms": ["wbc", "white blood cell count", "total leukocyte count", "tlc", "leukocytes", "white count"]
    },
    "rbc_count": {
        "label": "Red Blood Cell Count (RBC)",
        "unit": "x10^6/uL",
        "normal_min": 4.0,
        "normal_max": 5.9,
        "borderline_max": 6.2,
        "category": "Hematology",
        "synonyms": ["rbc", "red blood cell count", "erythrocyte count"]
    },
    "platelet_count": {
        "label": "Platelet Count",
        "unit": "x10^3/uL",
        "normal_min": 150.0,
        "normal_max": 450.0,
        "borderline_max": 149.0,
        "category": "Hematology",
        "synonyms": ["platelet count", "platelets", "plt", "thrombocytes"]
    },
    "hematocrit": {
        "label": "Hematocrit (PCV)",
        "unit": "%",
        "normal_min": 36.0,
        "normal_max": 50.0,
        "borderline_max": 54.0,
        "category": "Hematology",
        "synonyms": ["hematocrit", "haematocrit", "pcv", "packed cell volume", "hct"]
    },
    "mcv": {
        "label": "Mean Corpuscular Volume (MCV)",
        "unit": "fL",
        "normal_min": 80.0,
        "normal_max": 100.0,
        "borderline_max": 104.0,
        "category": "Hematology",
        "synonyms": ["mcv", "mean corpuscular volume"]
    },
    "mch": {
        "label": "Mean Corpuscular Hemoglobin (MCH)",
        "unit": "pg",
        "normal_min": 27.0,
        "normal_max": 33.0,
        "borderline_max": 35.0,
        "category": "Hematology",
        "synonyms": ["mch", "mean corpuscular hemoglobin"]
    },
    "mchc": {
        "label": "Mean Corpuscular Hemoglobin Conc. (MCHC)",
        "unit": "g/dL",
        "normal_min": 32.0,
        "normal_max": 36.0,
        "borderline_max": 37.0,
        "category": "Hematology",
        "synonyms": ["mchc", "mean corpuscular hemoglobin concentration"]
    },
    "ferritin": {
        "label": "Serum Ferritin",
        "unit": "ng/mL",
        "normal_min": 20.0,
        "normal_max": 300.0,
        "borderline_max": 19.0,
        "category": "Hematology",
        "synonyms": ["ferritin", "serum ferritin"]
    },

    # 6. Thyroid Profile
    "tsh": {
        "label": "Thyroid Stimulating Hormone (TSH)",
        "unit": "uIU/mL",
        "normal_min": 0.45,
        "normal_max": 4.5,
        "borderline_max": 7.0,
        "category": "Thyroid",
        "synonyms": ["tsh", "thyroid stimulating hormone", "thyrotropin"]
    },
    "free_t3": {
        "label": "Free Triiodothyronine (FT3)",
        "unit": "pg/mL",
        "normal_min": 2.0,
        "normal_max": 4.4,
        "borderline_max": 5.2,
        "category": "Thyroid",
        "synonyms": ["free t3", "ft3", "triiodothyronine free"]
    },
    "free_t4": {
        "label": "Free Thyroxine (FT4)",
        "unit": "ng/dL",
        "normal_min": 0.8,
        "normal_max": 1.8,
        "borderline_max": 2.2,
        "category": "Thyroid",
        "synonyms": ["free t4", "ft4", "thyroxine free"]
    },
    "total_t3": {
        "label": "Total T3",
        "unit": "ng/dL",
        "normal_min": 80.0,
        "normal_max": 200.0,
        "borderline_max": 230.0,
        "category": "Thyroid",
        "synonyms": ["total t3", "t3 total", "serum t3"]
    },
    "total_t4": {
        "label": "Total T4",
        "unit": "ug/dL",
        "normal_min": 4.5,
        "normal_max": 12.0,
        "borderline_max": 14.0,
        "category": "Thyroid",
        "synonyms": ["total t4", "t4 total", "serum t4"]
    },

    # 7. Vital Signs
    "heart_rate": {
        "label": "Resting Heart Rate",
        "unit": "bpm",
        "normal_min": 60.0,
        "normal_max": 100.0,
        "borderline_max": 110.0,
        "category": "Vitals",
        "synonyms": ["heart rate", "pulse", "resting pulse", "pulse rate", "hr"]
    },
    "spo2": {
        "label": "Oxygen Saturation (SpO2)",
        "unit": "%",
        "normal_min": 95.0,
        "normal_max": 100.0,
        "borderline_max": 94.0,
        "category": "Vitals",
        "synonyms": ["spo2", "oxygen saturation", "o2 sat", "pulse oximetry"]
    },
    "respiratory_rate": {
        "label": "Respiratory Rate",
        "unit": "breaths/min",
        "normal_min": 12.0,
        "normal_max": 20.0,
        "borderline_max": 24.0,
        "category": "Vitals",
        "synonyms": ["respiratory rate", "rr", "breathing rate"]
    },
    "temperature": {
        "label": "Body Temperature",
        "unit": "°F",
        "normal_min": 97.0,
        "normal_max": 99.0,
        "borderline_max": 100.4,
        "category": "Vitals",
        "synonyms": ["temperature", "body temperature", "temp"]
    },

    # 8. ECG Summary Parameters
    "pr_interval": {
        "label": "ECG PR Interval",
        "unit": "ms",
        "normal_min": 120.0,
        "normal_max": 200.0,
        "borderline_max": 220.0,
        "category": "Cardiovascular",
        "synonyms": ["pr interval", "pr duration"]
    },
    "qrs_duration": {
        "label": "ECG QRS Duration",
        "unit": "ms",
        "normal_min": 60.0,
        "normal_max": 100.0,
        "borderline_max": 120.0,
        "category": "Cardiovascular",
        "synonyms": ["qrs duration", "qrs width", "qrs complex"]
    },
    "qt_interval": {
        "label": "ECG QTc Interval",
        "unit": "ms",
        "normal_min": 350.0,
        "normal_max": 450.0,
        "borderline_max": 470.0,
        "category": "Cardiovascular",
        "synonyms": ["qt interval", "qtc", "qt duration"]
    },
    "st_segment_elevation": {
        "label": "ST Segment Elevation",
        "unit": "mm",
        "normal_min": 0.0,
        "normal_max": 1.0,
        "borderline_max": 2.0,
        "category": "Cardiovascular",
        "synonyms": ["st elevation", "st segment elevation", "st elevation mm"]
    },

    # 9. Urine Analysis
    "urine_protein": {
        "label": "Urinary Protein",
        "unit": "mg/dL",
        "normal_min": 0.0,
        "normal_max": 14.0,
        "borderline_max": 30.0,
        "category": "Renal",
        "synonyms": ["urine protein", "urinary protein", "proteinuria"]
    },
    "urine_glucose": {
        "label": "Urinary Glucose",
        "unit": "mg/dL",
        "normal_min": 0.0,
        "normal_max": 15.0,
        "borderline_max": 50.0,
        "category": "Renal",
        "synonyms": ["urine glucose", "urinary glucose", "glucosuria"]
    },
    "urine_specific_gravity": {
        "label": "Urine Specific Gravity",
        "unit": "index",
        "normal_min": 1.005,
        "normal_max": 1.030,
        "borderline_max": 1.035,
        "category": "Renal",
        "synonyms": ["urine specific gravity", "specific gravity", "sp gravity"]
    },

    # 10. Tumor Markers
    "psa": {
        "label": "Prostate Specific Antigen (PSA)",
        "unit": "ng/mL",
        "normal_min": 0.0,
        "normal_max": 4.0,
        "borderline_max": 10.0,
        "category": "Oncology",
        "synonyms": ["psa", "total psa", "prostate specific antigen"]
    },
    "ca125": {
        "label": "Cancer Antigen 125 (CA-125)",
        "unit": "U/mL",
        "normal_min": 0.0,
        "normal_max": 35.0,
        "borderline_max": 65.0,
        "category": "Oncology",
        "synonyms": ["ca 125", "ca-125", "cancer antigen 125"]
    },
    "cea": {
        "label": "Carcinoembryonic Antigen (CEA)",
        "unit": "ng/mL",
        "normal_min": 0.0,
        "normal_max": 3.0,
        "borderline_max": 5.0,
        "category": "Oncology",
        "synonyms": ["cea", "carcinoembryonic antigen"]
    },
    "afp": {
        "label": "Alpha-Fetoprotein (AFP)",
        "unit": "ng/mL",
        "normal_min": 0.0,
        "normal_max": 10.0,
        "borderline_max": 20.0,
        "category": "Oncology",
        "synonyms": ["afp", "alpha fetoprotein", "alpha-fetoprotein"]
    },

    # 11. Biopsy / Oncology Cellular Cytology (Wisconsin features)
    "mean_radius": {
        "label": "Mean Cell Radius",
        "unit": "um",
        "normal_min": 6.0,
        "normal_max": 14.0,
        "borderline_max": 16.5,
        "category": "Oncology",
        "synonyms": ["mean radius", "radius", "mean_radius", "cell radius"]
    },
    "mean_perimeter": {
        "label": "Mean Cell Perimeter",
        "unit": "um",
        "normal_min": 40.0,
        "normal_max": 90.0,
        "borderline_max": 105.0,
        "category": "Oncology",
        "synonyms": ["mean perimeter", "perimeter", "mean_perimeter"]
    },
    "mean_area": {
        "label": "Mean Cell Area",
        "unit": "um²",
        "normal_min": 140.0,
        "normal_max": 650.0,
        "borderline_max": 800.0,
        "category": "Oncology",
        "synonyms": ["mean area", "area", "mean_area"]
    },
    "mean_concavity": {
        "label": "Mean Cell Concavity",
        "unit": "index",
        "normal_min": 0.0,
        "normal_max": 0.08,
        "borderline_max": 0.15,
        "category": "Oncology",
        "synonyms": ["mean concavity", "concavity", "mean_concavity"]
    },
    "mean_concave_points": {
        "label": "Mean Concave Points",
        "unit": "index",
        "normal_min": 0.0,
        "normal_max": 0.05,
        "borderline_max": 0.09,
        "category": "Oncology",
        "synonyms": ["mean concave points", "concave points", "mean_concave_points"]
    },
    "mean_texture": {
        "label": "Mean Cell Texture",
        "unit": "index",
        "normal_min": 9.0,
        "normal_max": 20.0,
        "borderline_max": 24.0,
        "category": "Oncology",
        "synonyms": ["mean texture", "texture", "mean_texture"]
    },
    "mean_smoothness": {
        "label": "Mean Cell Smoothness",
        "unit": "index",
        "normal_min": 0.05,
        "normal_max": 0.10,
        "borderline_max": 0.13,
        "category": "Oncology",
        "synonyms": ["mean smoothness", "smoothness", "mean_smoothness"]
    },
    "mean_compactness": {
        "label": "Mean Cell Compactness",
        "unit": "index",
        "normal_min": 0.02,
        "normal_max": 0.15,
        "borderline_max": 0.20,
        "category": "Oncology",
        "synonyms": ["mean compactness", "compactness", "mean_compactness"]
    },
    "mean_symmetry": {
        "label": "Mean Cell Symmetry",
        "unit": "index",
        "normal_min": 0.10,
        "normal_max": 0.20,
        "borderline_max": 0.25,
        "category": "Oncology",
        "synonyms": ["mean symmetry", "symmetry", "mean_symmetry"]
    },
    "mean_fractal_dimension": {
        "label": "Mean Fractal Dimension",
        "unit": "index",
        "normal_min": 0.04,
        "normal_max": 0.07,
        "borderline_max": 0.09,
        "category": "Oncology",
        "synonyms": ["mean fractal dimension", "fractal dimension", "mean_fractal_dimension"]
    },

    # 12. Inflammatory Markers
    "crp": {
        "label": "C-Reactive Protein (CRP)",
        "unit": "mg/L",
        "normal_min": 0.0,
        "normal_max": 5.0,
        "borderline_max": 10.0,
        "category": "Inflammatory",
        "synonyms": ["crp", "c-reactive protein", "c reactive protein"]
    },
    "esr": {
        "label": "Erythrocyte Sedimentation Rate (ESR)",
        "unit": "mm/hr",
        "normal_min": 0.0,
        "normal_max": 20.0,
        "borderline_max": 30.0,
        "category": "Inflammatory",
        "synonyms": ["esr", "erythrocyte sedimentation rate", "sed rate"]
    },

    # 13. Vitamins & Minerals
    "vitamin_d": {
        "label": "25-Hydroxy Vitamin D",
        "unit": "ng/mL",
        "normal_min": 30.0,
        "normal_max": 100.0,
        "borderline_max": 29.0,
        "category": "Vitamins & Minerals",
        "synonyms": ["vitamin d", "25-hydroxy vitamin d", "vit d", "25-oh vitamin d"]
    },
    "vitamin_b12": {
        "label": "Vitamin B12 (Cobalamin)",
        "unit": "pg/mL",
        "normal_min": 200.0,
        "normal_max": 900.0,
        "borderline_max": 250.0,
        "category": "Vitamins & Minerals",
        "synonyms": ["vitamin b12", "b12", "vit b12", "cobalamin"]
    },
    "serum_iron": {
        "label": "Serum Iron",
        "unit": "ug/dL",
        "normal_min": 60.0,
        "normal_max": 170.0,
        "borderline_max": 59.0,
        "category": "Vitamins & Minerals",
        "synonyms": ["serum iron", "iron", "fe"]
    },
    "calcium": {
        "label": "Serum Calcium",
        "unit": "mg/dL",
        "normal_min": 8.5,
        "normal_max": 10.5,
        "borderline_max": 11.2,
        "category": "Vitamins & Minerals",
        "synonyms": ["calcium", "serum calcium", "ca"]
    }
}


def convert_biomarker_unit(key: str, raw_value: float, detected_unit: Optional[str]) -> Tuple[float, str, bool]:
    """
    Standardize biomarker values into reference units.
    Handles mg/dL <-> mmol/L, umol/L, g/L, Celsius <-> Fahrenheit.
    Returns: (converted_value, standard_unit, was_converted)
    """
    meta = BIOMARKER_DEFINITIONS.get(key, {})
    std_unit = meta.get("unit", "")
    if not detected_unit:
        return raw_value, std_unit, False

    u = detected_unit.strip().lower()
    val = float(raw_value)
    converted = False

    # 1. Glucose: mmol/L -> mg/dL (factor 18.0182)
    if key in ["fasting_glucose", "postprandial_glucose"]:
        if "mmol" in u:
            val = round(val * 18.0182, 1)
            converted = True

    # 2. Cholesterol / Lipids: mmol/L -> mg/dL (factor 38.67 for TC/LDL/HDL, 88.57 for TG)
    elif key in ["total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "vldl"]:
        if "mmol" in u:
            val = round(val * 38.67, 1)
            converted = True
    elif key == "triglycerides":
        if "mmol" in u:
            val = round(val * 88.57, 1)
            converted = True

    # 3. Creatinine: umol/L -> mg/dL (factor 88.4)
    elif key == "serum_creatinine":
        if "umol" in u or "µmol" in u or "micromol" in u:
            val = round(val / 88.4, 2)
            converted = True

    # 4. Bilirubin: umol/L -> mg/dL (factor 17.1)
    elif key in ["total_bilirubin", "direct_bilirubin"]:
        if "umol" in u or "µmol" in u or "micromol" in u:
            val = round(val / 17.1, 2)
            converted = True

    # 5. BUN / Urea: mmol/L -> mg/dL (factor 6.0)
    elif key == "bun":
        if "mmol" in u:
            val = round(val * 6.0, 1)
            converted = True

    # 6. Hemoglobin: g/L -> g/dL (factor 10.0)
    elif key == "hemoglobin":
        if u in ["g/l", "g/litre", "gm/l"]:
            val = round(val / 10.0, 1)
            converted = True

    # 7. Free T4: pmol/L -> ng/dL (factor 12.87)
    elif key == "free_t4":
        if "pmol" in u:
            val = round(val / 12.87, 2)
            converted = True

    # 8. Calcium: mmol/L -> mg/dL (factor 4.0)
    elif key == "calcium":
        if "mmol" in u:
            val = round(val * 4.0, 2)
            converted = True

    # 9. Temperature: Celsius -> Fahrenheit if value < 45
    elif key == "temperature":
        if "c" in u and val < 45.0:
            val = round((val * 9.0 / 5.0) + 32.0, 1)
            converted = True

    return val, std_unit, converted


# Standard Preset Reports for testing & demo
PRESET_LAB_REPORTS: List[Dict[str, Any]] = []

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded files:
    - PDF: via pdfplumber (text + tables) with fallback to pypdf
    - CSV / TSV: via pandas formatting
    - XLSX / XLS: via openpyxl / pandas
    - DOCX: via python-docx
    - Images (.jpg, .jpeg, .png): via PIL + pytesseract OCR
    - Plain text / JSON / RTF: via multi-encoding decoding
    """
    ext = Path(filename).suffix.lower()

    # 1. PDF Documents
    if ext == ".pdf":
        text_parts = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt:
                        text_parts.append(pt)
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            for row in tbl:
                                if row and any(row):
                                    text_parts.append(" | ".join(str(c or "").strip() for c in row))
            if text_parts:
                return "\n".join(text_parts)
        except Exception:
            pass

        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text_parts.append(pt)
            if text_parts:
                return "\n".join(text_parts)
        except Exception:
            pass

        # Generic stream extraction fallback
        try:
            text_matches = re.findall(rb'\(([^\)\r\n]{1,100})\)\s*Tj', file_bytes)
            if text_matches and len(text_matches) > 5:
                decoded = [m.decode("latin-1", errors="ignore") for m in text_matches]
                return " ".join(decoded)
        except Exception:
            pass

    # 2. Spreadsheet files: CSV, TSV, XLSX, XLS
    if ext in [".csv", ".tsv"]:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                lines = []
                for _, row in df.iterrows():
                    row_vals = [str(v).strip() for v in row.values if pd.notna(v)]
                    lines.append(" : ".join(row_vals))
                return "\n".join(lines)
            except Exception:
                continue

    if ext in [".xlsx", ".xls"]:
        try:
            excel = pd.ExcelFile(io.BytesIO(file_bytes))
            lines = []
            for sheet_name in excel.sheet_names:
                df = excel.parse(sheet_name)
                for _, row in df.iterrows():
                    row_vals = [str(v).strip() for v in row.values if pd.notna(v)]
                    if row_vals:
                        lines.append(" : ".join(row_vals))
            return "\n".join(lines)
        except Exception:
            pass

    # 3. Scanned Images: JPG, PNG, TIFF, BMP (OCR via pytesseract)
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(io.BytesIO(file_bytes))
            width, height = img.size
            if width < 1 or height < 1 or width * height > 40_000_000:
                raise ValueError("Image dimensions exceed the 40-megapixel processing limit.")
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text and len(ocr_text.strip()) > 10:
                return ocr_text
        except Exception as e:
            return f"Scanned image detected: {filename}. (Host OCR engine status: {e})"

    # 4. DOCX documents
    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    lines.append(" | ".join(cell.text.strip() for cell in row.cells))
            return "\n".join(lines)
        except Exception:
            pass

    # 5. Plain text, JSON, RTF, Log
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue

    # Fallback to printable ascii chunks
    printable = re.findall(rb'[\x20-\x7E\t\r\n]{4,}', file_bytes)
    if printable:
        return "\n".join(c.decode("latin-1", errors="ignore") for c in printable)

    return ""


def parse_biomarkers_from_text(text: str) -> Dict[str, float]:
    """
    Extract biomarker values and units from unstructured or structured text,
    apply unit conversions automatically, and return normalized values.
    """
    results: Dict[str, float] = {}
    lines = text.splitlines()

    for key, meta in BIOMARKER_DEFINITIONS.items():
        synonyms = sorted([re.escape(s) for s in meta["synonyms"]], key=len, reverse=True)
        # Pattern captures: SYNONYM (optional parens) (optional delimiters) NUMBER (optional unit)
        pattern_str = (
            r'(?:' + '|'.join(synonyms) + r')'
            r'(?:\s*\([^)]*\))?'
            r'(?:\s*[:=|, -]|\s+is|\s+at|\s+value)*\s*'
            r'([0-9]+(?:\.[0-9]+)?)'
            r'(?:[\s:|,]+([a-zA-Z%°/\^0-9²]+))?'
        )
        pattern = re.compile(pattern_str, re.IGNORECASE)

        matched = False
        for line in lines:
            m = pattern.search(line)
            if m:
                try:
                    raw_val = float(m.group(1))
                    unit_str = m.group(2)
                    # If unit not captured in group 2, scan line for unit tokens
                    if not unit_str:
                        l_lower = line.lower()
                        if "mmol" in l_lower:
                            unit_str = "mmol/l"
                        elif "umol" in l_lower or "µmol" in l_lower or "micromol" in l_lower:
                            unit_str = "umol/l"
                        elif "pmol" in l_lower:
                            unit_str = "pmol/l"
                        elif "g/l" in l_lower and "g/dl" not in l_lower:
                            unit_str = "g/l"

                    norm_val, _, _ = convert_biomarker_unit(key, raw_val, unit_str)
                    results[key] = norm_val
                    matched = True
                    break
                except (ValueError, TypeError):
                    pass

        if not matched:
            m = pattern.search(text)
            if m:
                try:
                    raw_val = float(m.group(1))
                    unit_str = m.group(2)
                    norm_val, _, _ = convert_biomarker_unit(key, raw_val, unit_str)
                    results[key] = norm_val
                except (ValueError, TypeError):
                    pass

    return results


def flag_biomarker(key: str, value: float) -> Dict[str, Any]:
    """Categorize biomarker value as normal, low, high, or critical."""
    meta = BIOMARKER_DEFINITIONS.get(key)
    if meta is None or not math.isfinite(float(value)):
        return {
            "key": key,
            "label": meta.get("label", key) if meta else key,
            "value": value,
            "unit": meta.get("unit", "") if meta else "",
            "reference_range": None,
            "status": "unrecognized",
            "badge": "Reference unavailable",
            "color": "#64748B",
            "category": meta.get("category", "General") if meta else "General",
        }
    unit = meta.get("unit", "")
    n_min = meta.get("normal_min", 0.0)
    n_max = meta.get("normal_max", 100.0)
    b_max = meta.get("borderline_max", n_max * 1.2)

    low_risk_markers = [
        "egfr", "albumin", "hemoglobin", "hdl_cholesterol", "platelet_count",
        "ferritin", "spo2", "vitamin_d", "vitamin_b12", "serum_iron"
    ]

    if key in low_risk_markers:
        if value < (n_min * 0.75):
            status = "critical_low"
            badge = "Critically Low"
            color = "#EF4444"
        elif value < n_min:
            status = "low"
            badge = "Low"
            color = "#3B82F6"
        else:
            status = "normal"
            badge = "Normal"
            color = "#10B981"
    else:
        if value > b_max:
            status = "critical_high"
            badge = "Critically High"
            color = "#EF4444"
        elif value > n_max:
            status = "high"
            badge = "Elevated"
            color = "#F59E0B"
        elif value < n_min:
            status = "low"
            badge = "Low"
            color = "#3B82F6"
        else:
            status = "normal"
            badge = "Normal"
            color = "#10B981"

    return {
        "key": key,
        "label": meta.get("label", key),
        "value": value,
        "unit": unit,
        "reference_range": f"{n_min} - {n_max} {unit}",
        "status": status,
        "badge": badge,
        "color": color,
        "category": meta.get("category", "General")
    }


def evaluate_disease_risks(
    biomarkers: Dict[str, float],
    patient_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return no disease predictions until a validated model is available."""
    return []


def compute_shap_explanations(
    biomarkers: Dict[str, float],
    patient_profile: Optional[Dict[str, Any]],
    diseases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """No model-based attribution exists for report review in this build."""
    return []


def analyze_report(
    raw_text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    direct_biomarkers: Optional[Dict[str, float]] = None,
    patient_profile: Optional[Dict[str, Any]] = None,
    report_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract report text, parse recognizable biomarkers, and return reference flags.
    Disease-specific scores remain unavailable until validated models are configured.
    """
    extracted_text = ""
    if file_bytes and filename:
        extracted_text = extract_text_from_file(file_bytes, filename)
    elif raw_text:
        extracted_text = raw_text

    parsed = parse_biomarkers_from_text(extracted_text) if extracted_text else {}
    if direct_biomarkers:
        parsed.update(direct_biomarkers)

    flagged_list = [flag_biomarker(k, v) for k, v in parsed.items()]
    abnormal_count = sum(1 for m in flagged_list if m["status"] in ["high", "critical_high", "low", "critical_low"])

    # Patient disease prediction is unavailable; only report review fields are returned.
    diseases = evaluate_disease_risks(parsed, patient_profile)
    evaluated_diseases = [d for d in diseases if d["status"] == "evaluated"]
    top_disease = evaluated_diseases[0] if evaluated_diseases else None

    # No patient-report model exists to explain.
    shap_explanations = compute_shap_explanations(parsed, patient_profile, diseases)

    return {
        "report_title": report_title or (filename if filename else "Clinical Laboratory Evaluation"),
        "raw_text_snippet": (extracted_text[:600] + "...") if len(extracted_text) > 600 else extracted_text,
        "biomarkers_extracted_count": len(parsed),
        "abnormal_biomarkers_count": abnormal_count,
        "parsed_biomarkers": parsed,
        "flagged_biomarkers": flagged_list,
        "diseases": diseases,
        "primary_suspected_disease": top_disease["name"] if top_disease else "Disease scoring unavailable",
        "primary_chance_percentage": top_disease["chances_percentage"] if top_disease else None,
        "primary_risk_tier": top_disease["risk_tier"] if top_disease else "insufficient_data",
        "primary_risk_category": top_disease["risk_category"] if top_disease else "Insufficient Data",
        "primary_risk_color": top_disease["risk_color"] if top_disease else "#64748B",
        "shap_explanations": shap_explanations,
        "disease_prediction_status": "unavailable_until_validated",
        "clinical_disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
    }


def generate_patient_report_pdf(
    patient_profile: Dict[str, Any],
    biomarkers: Dict[str, float],
    analysis_result: Dict[str, Any],
    output_path: Optional[str] = None
) -> bytes:
    """
    Generate a lab report review PDF with extracted values and reference flags.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    bold_style = ParagraphStyle(
        'Bold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#94A3B8'),
        alignment=1
    )

    story = []

    # Title & Header
    story.append(Paragraph("Q-DIAGNOSE LAB REPORT REVIEW", title_style))
    story.append(Paragraph("Extracted values and general reference-range flags | Research prototype", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=10))

    # Patient Demographic & Intake Table
    profile = patient_profile or {}
    bmi_val = profile.get("bmi", "N/A")
    if isinstance(bmi_val, (int, float)):
        bmi_val = f"{float(bmi_val):.1f}"

    fam_str = ", ".join(profile.get("family_history", [])) or "None reported"
    cond_str = ", ".join(profile.get("existing_conditions", [])) or "None reported"

    patient_info = [
        [
            Paragraph("<b>Subject Age:</b> " + str(profile.get("age", "N/A")), body_style),
            Paragraph("<b>Sex:</b> " + str(profile.get("sex", "N/A")).capitalize(), body_style),
            Paragraph("<b>BMI:</b> " + str(bmi_val), body_style)
        ],
        [
            Paragraph("<b>Smoking:</b> " + str(profile.get("smoking", "Never")).replace("_", " ").capitalize(), body_style),
            Paragraph("<b>Alcohol:</b> " + str(profile.get("alcohol", "None")).capitalize(), body_style),
            Paragraph("<b>Activity:</b> " + str(profile.get("activity", "Moderate")).capitalize(), body_style)
        ],
        [
            Paragraph("<b>Family History:</b> " + fam_str, body_style),
            Paragraph("<b>Existing Conditions:</b> " + cond_str, body_style),
            Paragraph("<b>Assessment Date:</b> " + pd.Timestamp.now().strftime("%Y-%m-%d"), body_style)
        ]
    ]
    t_patient = Table(patient_info, colWidths=[180, 180, 180])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 12))

    # Report prediction is unavailable in this build; never imply a risk score.
    diseases = analysis_result.get("diseases", [])
    if not diseases:
        story.append(Paragraph("Disease prediction unavailable", h2_style))
        story.append(Paragraph(
            "This build does not calculate disease risk or predict future illness from uploaded lab reports. "
            "Extracted values and reference-range flags are for review only.",
            body_style,
        ))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("Disease model results", h2_style))

    disease_headers = [
        Paragraph("<b>Disease / Organ System</b>", bold_style),
        Paragraph("<b>Risk %</b>", bold_style),
        Paragraph("<b>Risk Category</b>", bold_style),
        Paragraph("<b>Model Output</b>", bold_style),
        Paragraph("<b>Top Factors & Clinical Action</b>", bold_style)
    ]
    disease_rows = [disease_headers]

    for d in diseases:
        name_cell = Paragraph(f"<b>{d['name']}</b><br/><font color='#64748B'>{d['organ_system']}</font>", body_style)
        if d.get("status") == "insufficient_data":
            pct_cell = Paragraph("<font color='#64748B'>N/A</font>", body_style)
            tier_cell = Paragraph("<font color='#64748B'>Insufficient Data</font>", body_style)
            q_vs_c = Paragraph("Inputs Missing", body_style)
            missing_rep = d.get('missing_required_reports', [])
            missing_str = ", ".join(missing_rep) if missing_rep else "Required tests"
            factors_cell = Paragraph(f"<b>Missing:</b> {missing_str}", body_style)
        else:
            pct_cell = Paragraph(f"<b>{d['chances_percentage']}%</b>", bold_style)
            tier_cell = Paragraph(f"<font color='{d['risk_color']}'><b>{d['risk_category']}</b></font>", body_style)
            q_vs_c = Paragraph(f"Q: {round(d['quantum_probability']*100, 1)}% | C: {round(d['classical_probability']*100, 1)}%", body_style)
            factor_names = [f["name"] for f in d.get("top_factors", [])[:2]]
            action_snippet = (d.get('action_recommendation', '')[:65] + '...') if len(d.get('action_recommendation', '')) > 65 else d.get('action_recommendation', '')
            factors_cell = Paragraph(f"<b>Factors:</b> {', '.join(factor_names)}<br/><b>Action:</b> {action_snippet}", body_style)

        disease_rows.append([name_cell, pct_cell, tier_cell, q_vs_c, factors_cell])

    if diseases:
        t_diseases = Table(disease_rows, colWidths=[130, 50, 95, 105, 160])
        t_diseases.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94A3B8')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_diseases)
        story.append(Spacer(1, 12))

    # Abnormal Biomarkers Summary
    flagged = analysis_result.get("flagged_biomarkers", [])
    abnormals = [f for f in flagged if f.get("status") in ["high", "critical_high", "low", "critical_low"]]
    if abnormals:
        story.append(Paragraph(f"Identified Biomarker Deviations ({len(abnormals)} Out-of-Range Markers)", h2_style))
        ab_headers = [
            Paragraph("<b>Biomarker</b>", bold_style),
            Paragraph("<b>Observed Value</b>", bold_style),
            Paragraph("<b>Reference Target</b>", bold_style),
            Paragraph("<b>Clinical Status</b>", bold_style)
        ]
        ab_rows = [ab_headers]
        for a in abnormals:
            ab_rows.append([
                Paragraph(a["label"], body_style),
                Paragraph(f"<b>{a['value']}</b> {a['unit']}", bold_style),
                Paragraph(a["reference_range"], body_style),
                Paragraph(f"<font color='{a['color']}'><b>{a['badge']}</b></font>", body_style)
            ])
        t_ab = Table(ab_rows, colWidths=[160, 110, 150, 120])
        t_ab.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_ab)
        story.append(Spacer(1, 12))

    story.append(Paragraph("Review notes", h2_style))
    story.append(Paragraph(
        "Compare each extracted value, unit, and flag with the original report and the laboratory's own reference interval. "
        "Discuss questions with a qualified healthcare professional.",
        body_style,
    ))
    story.append(Spacer(1, 14))

    # Official Research Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph(
        "<b>RESEARCH PROTOTYPE DISCLAIMER:</b> Q-Diagnose extracts recognizable lab values and displays general reference-range flags. "
        "This report is not a diagnosis and does not predict future illness. No validated disease-specific model is connected to this workflow. "
        "Use the original laboratory report and consult a qualified healthcare professional.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
