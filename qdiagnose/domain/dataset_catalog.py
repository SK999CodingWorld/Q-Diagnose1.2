"""Curated public disease datasets with explicit label meaning and provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "heart_disease_cleveland": {
        "uci_id": 45,
        "name": "UCI Heart Disease (Cleveland)",
        "disease_area": "Cardiovascular disease",
        "description": "Clinical features with angiographic heart-disease labels; values above 0 are mapped to positive.",
        "target_definition": "Positive = angiographic disease status 1–4; negative = status 0.",
        "positive_class_label": "Angiographic disease present",
        "negative_class_label": "No angiographic disease",
        "positive_rule": "numeric_gt_zero",
        "drop_columns": ["id", "name"],
        "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "citation": "Janosi et al. (1989), UCI Heart Disease, DOI 10.24432/C52P4X",
        "license": "CC BY 4.0",
        "caveat": "Small cohort (303 records); results are sensitive to split choice and should be treated as exploratory.",
    },
    "chronic_kidney_disease": {
        "uci_id": 336,
        "name": "UCI Chronic Kidney Disease",
        "disease_area": "Kidney disease",
        "description": "Clinical and laboratory measurements with missing and categorical values.",
        "target_definition": "Positive = CKD; negative = not CKD.",
        "positive_class_label": "CKD",
        "negative_class_label": "No CKD",
        "positive_values": ["ckd"],
        "drop_columns": [],
        "source_url": "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
        "citation": "Rubini et al. (2015), UCI Chronic Kidney Disease, DOI 10.24432/C5G020",
        "license": "CC BY 4.0",
        "caveat": "Small, single-source cohort with substantial missingness; not representative of current care populations.",
    },
    "diabetes_health_indicators": {
        "uci_id": 891,
        "name": "CDC Diabetes Health Indicators",
        "disease_area": "Diabetes",
        "description": "US health survey features and self-reported diabetes indicators.",
        "target_definition": "Positive = the source dataset's diabetes-positive target; this is a survey label, not a lab-confirmed diagnosis.",
        "positive_class_label": "Survey reports diabetes",
        "negative_class_label": "Survey does not report diabetes",
        "positive_values": ["1", "1.0", "diabetes"],
        "drop_columns": ["id"],
        "source_url": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        "citation": "CDC Diabetes Health Indicators, UCI dataset 891, DOI 10.24432/C53919",
        "license": "See source dataset terms",
        "caveat": "Self-reported survey outcome; label must not be interpreted as a clinical diagnosis or prevalence estimate.",
    },
    "breast_cancer_wisconsin_original": {
        "uci_id": 15,
        "name": "UCI Breast Cancer Wisconsin (Original)",
        "disease_area": "Breast cancer",
        "description": "Cell-nuclei measurements with benign and malignant labels.",
        "target_definition": "Positive = malignant (source value 4); negative = benign (source value 2).",
        "positive_class_label": "Malignant",
        "negative_class_label": "Benign",
        "positive_values": ["4", "4.0", "malignant"],
        "drop_columns": ["sample_code_number", "id"],
        "source_url": "https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original",
        "citation": "Wolberg (1990), UCI Breast Cancer Wisconsin (Original), DOI 10.24432/C5HP4Z",
        "license": "CC BY 4.0",
        "caveat": "Historical dataset from 1989–1991; not a contemporary screening cohort.",
    },
    "gallstone": {
        "uci_id": 1150,
        "name": "UCI Gallstone Disease",
        "disease_area": "Gallbladder disease",
        "description": "Clinical and laboratory measurements from a gallstone disease cohort.",
        "target_definition": "Positive = gallstone present (source value 0); negative = no gallstone (source value 1).",
        "positive_class_label": "Gallstone present",
        "negative_class_label": "No gallstone",
        "positive_values": ["0", "0.0"],
        "drop_columns": [],
        "source_url": "https://archive.ics.uci.edu/dataset/1150/gallstone-1",
        "citation": "UCI Gallstone dataset, DOI 10.1097/MD.0000000000037258",
        "license": "CC BY 4.0",
        "caveat": "Small clinical cohort; the source uses inverse coded labels and results are exploratory.",
    },
}


def list_public_datasets() -> list[dict[str, Any]]:
    """Return safe public metadata for the currently supported importer."""
    return [{"key": key, **metadata} for key, metadata in DATASET_CATALOG.items()]


def import_public_dataset(key: str, upload_dir):
    """Fetch a curated UCI dataset, normalize it to a binary numeric CSV, and register provenance."""
    import numpy as np
    import pandas as pd

    from qdiagnose.domain import database as db

    spec = DATASET_CATALOG.get(key)
    if spec is None:
        raise KeyError(f"Unknown public dataset '{key}'.")

    dataset_id = f"uci_{key}"
    existing = db.get_dataset(dataset_id)
    if existing and existing.get("filepath") and Path(existing["filepath"]).is_file():
        return existing, False

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise RuntimeError("Public dataset imports require the 'ucimlrepo' package. Install requirements.txt and retry.") from error

    try:
        source_dataset = fetch_ucirepo(id=spec["uci_id"])
        features = source_dataset.data.features.copy()
        target_frame = source_dataset.data.targets
    except Exception as error:
        raise RuntimeError("Could not fetch the dataset from the UCI Machine Learning Repository.") from error

    if not isinstance(features, pd.DataFrame) or not isinstance(target_frame, pd.DataFrame) or target_frame.shape[1] != 1:
        raise ValueError("The selected repository entry does not expose one tabular target column.")

    target_name = str(target_frame.columns[0])
    target = target_frame.iloc[:, 0]
    valid_target = target.notna()
    features = features.loc[valid_target].reset_index(drop=True)
    target = target.loc[valid_target].reset_index(drop=True)
    positive_values = {str(value).strip().casefold() for value in spec.get("positive_values", [])}

    if spec.get("positive_rule") == "numeric_gt_zero":
        numeric_target = pd.to_numeric(target, errors="coerce")
        valid_target = numeric_target.notna()
        features = features.loc[valid_target].reset_index(drop=True)
        target_binary = (numeric_target.loc[valid_target].to_numpy() > 0).astype(int)
    else:
        normalized_target = target.astype("string").str.strip().str.casefold()
        target_binary = normalized_target.isin(positive_values).astype(int).to_numpy()

    if len(np.unique(target_binary)) != 2:
        raise ValueError("The curated label mapping did not produce both binary classes; dataset import was stopped.")

    drop_names = {str(value).strip().casefold().replace(" ", "_") for value in spec.get("drop_columns", [])}
    keep_columns = [column for column in features.columns if str(column).strip().casefold().replace(" ", "_") not in drop_names]
    features = features.loc[:, keep_columns]

    normalized_features = pd.DataFrame(index=features.index)
    for column in features.columns:
        series = features[column]
        numeric = pd.to_numeric(series, errors="coerce")
        observed = series.notna().sum()
        numeric_observed = numeric.notna().sum()
        if observed == 0 or numeric_observed / observed >= 0.85:
            normalized_features[str(column)] = numeric
        else:
            normalized_features[str(column)] = series.astype("string").fillna("__missing__").str.strip()

    encoded_features = pd.get_dummies(normalized_features, dummy_na=False, dtype=np.uint8)
    encoded_features = encoded_features.replace([np.inf, -np.inf], np.nan)
    if len(encoded_features) < 20 or encoded_features.shape[1] < 2:
        raise ValueError("The selected dataset is too small or does not have enough usable features.")

    result = encoded_features.copy()
    result["target"] = target_binary
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{dataset_id}.csv"
    result.to_csv(file_path, index=False)

    classes = {
        "Negative (0)": int((target_binary == 0).sum()),
        "Positive (1)": int((target_binary == 1).sum()),
    }
    metadata = {
        "source": "UCI Machine Learning Repository",
        "source_dataset_id": spec["uci_id"],
        "source_url": spec["source_url"],
        "citation": spec["citation"],
        "license": spec["license"],
        "disease_area": spec["disease_area"],
        "target_definition": spec["target_definition"],
        "positive_class_label": spec["positive_class_label"],
        "negative_class_label": spec["negative_class_label"],
        "source_target_column": target_name,
        "caveat": spec["caveat"],
        "preprocessing": "Categorical predictors one-hot encoded; missing numeric values retained for fold-local imputation.",
        "clinical_use": "Research only; not a clinical diagnostic dataset or validated model.",
    }
    record = {
        "id": dataset_id,
        "name": spec["name"],
        "filename": file_path.name,
        "filepath": str(file_path),
        "n_rows": int(len(result)),
        "n_cols": int(len(result.columns)),
        "target_col": "target",
        "missing_count": int(result.isna().sum().sum()),
        "duplicate_count": int(result.duplicated().sum()),
        "class_distribution": classes,
        "metadata": metadata,
    }
    db.save_dataset(record)
    return record, True
