"""Carga y preprocesamiento reproducible de datos clínicos y NIfTI."""

from __future__ import annotations

import io
import logging
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import nibabel as nib
import numpy as np
import pandas as pd
import requests
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from config import RANDOM_SEED, SETTINGS, TARGET_CANDIDATES

LOGGER = logging.getLogger(__name__)
IDENTIFIER_HINTS = ("id", "patientid", "patient_id", "doctorincharge", "nombre", "name")
POSITIVE_LABEL_HINTS = (
    "alzheimer",
    "dement",
    "positive",
    "positivo",
    "disease",
    "enfermo",
    "case",
    "yes",
    "si",
)
NEGATIVE_LABEL_HINTS = (
    "control",
    "healthy",
    "normal",
    "negative",
    "negativo",
    "sano",
    "no",
)


@dataclass
class DataBundle:
    X: pd.DataFrame
    y: pd.Series
    target: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str]
    class_labels: dict[int, Any]


def _name_of(source: Any) -> str:
    return str(getattr(source, "name", source)).lower()


def extract_nifti_features(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Extrae biomarcadores volumétricos básicos de un archivo NIfTI."""
    if hasattr(source, "read"):
        raw = source.read()
        image = nib.Nifti1Image.from_bytes(raw)
    else:
        image = nib.load(str(source))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    finite = data[np.isfinite(data)]
    nonzero = finite[np.abs(finite) > 1e-8]
    values = nonzero if nonzero.size else finite
    voxel_volume = float(np.prod(image.header.get_zooms()[:3]))
    return pd.DataFrame(
        [
            {
                "MRI_MeanIntensity": float(np.mean(values)),
                "MRI_StdIntensity": float(np.std(values)),
                "MRI_P05": float(np.percentile(values, 5)),
                "MRI_Median": float(np.median(values)),
                "MRI_P95": float(np.percentile(values, 95)),
                "MRI_NonzeroVoxels": int(nonzero.size),
                "MRI_NonzeroVolumeMM3": float(nonzero.size * voxel_volume),
            }
        ]
    )


def read_dataset(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Lee CSV, XLSX o NIfTI preservando el objeto subido por Streamlit."""
    name = _name_of(source)
    if hasattr(source, "seek"):
        source.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(source)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    if name.endswith((".nii", ".nii.gz")):
        return extract_nifti_features(source)
    raise ValueError("Formato no soportado. Use CSV, XLSX, XLS, NII o NII.GZ.")


def download_public_dataset(url: str | None = None, timeout: int = 45) -> pd.DataFrame:
    """Descarga el ZIP público de Kaggle y devuelve su primer CSV."""
    response = requests.get(url or SETTINGS.public_dataset_url, timeout=timeout)
    response.raise_for_status()
    payload = io.BytesIO(response.content)
    if zipfile.is_zipfile(payload):
        with zipfile.ZipFile(payload) as archive:
            csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("El paquete público no contiene un CSV.")
            with archive.open(csv_files[0]) as stream:
                return pd.read_csv(stream)
    payload.seek(0)
    return pd.read_csv(payload)


def infer_target(frame: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError("No se detectó la variable objetivo; selecciónela manualmente.")


def _normalized_label(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(character for character in text if not unicodedata.combining(character))


def _ordered_binary_classes(values: list[Any]) -> list[Any]:
    """Ordena clase negativa/positiva con convenciones clínicas comunes."""
    if all(isinstance(value, (int, float, np.integer, np.floating, bool)) for value in values):
        return sorted(values)
    negative = [
        value
        for value in values
        if any(hint in _normalized_label(value) for hint in NEGATIVE_LABEL_HINTS)
    ]
    positive = [
        value
        for value in values
        if value not in negative
        and any(hint in _normalized_label(value) for hint in POSITIVE_LABEL_HINTS)
    ]
    if len(positive) == 1:
        return [value for value in values if value != positive[0]] + positive
    if len(negative) == 1:
        return negative + [value for value in values if value != negative[0]]
    return sorted(values, key=str)


def prepare_data(frame: pd.DataFrame, target: str) -> DataBundle:
    if target not in frame.columns:
        raise KeyError(f"La columna objetivo '{target}' no existe.")
    working = frame.copy()
    working = working.dropna(subset=[target])
    original_y = working.pop(target)
    classes = _ordered_binary_classes(original_y.unique().tolist())
    if len(classes) != 2:
        raise ValueError("La variable objetivo debe contener exactamente dos clases.")
    label_to_code = {label: index for index, label in enumerate(classes)}
    y = original_y.map(label_to_code).astype(int)
    class_labels = {index: label for label, index in label_to_code.items()}

    dropped = [
        column
        for column in working.columns
        if column.lower().replace(" ", "") in IDENTIFIER_HINTS
        or working[column].nunique(dropna=True) == len(working)
    ]
    X = working.drop(columns=dropped, errors="ignore")
    if X.empty:
        raise ValueError("No quedan predictores después de retirar identificadores.")
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numerical]
    return DataBundle(X, y, target, numerical, categorical, dropped, class_labels)


def build_preprocessor(
    bundle: DataBundle, scaling: str = "standard"
) -> ColumnTransformer:
    scaler = RobustScaler() if scaling == "robust" else StandardScaler()
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if bundle.numerical_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
                ),
                bundle.numerical_columns,
            )
        )
    if bundle.categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                bundle.categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_sampler(name: str):
    normalized = name.upper()
    if normalized == "SMOTE":
        return SMOTE(random_state=RANDOM_SEED)
    if normalized == "ADASYN":
        return ADASYN(random_state=RANDOM_SEED)
    return "passthrough"


def dataset_summary(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "filas": int(len(frame)),
        "columnas": int(len(frame.columns)),
        "faltantes": int(frame.isna().sum().sum()),
        "duplicados": int(frame.duplicated().sum()),
    }
    if target and target in frame:
        result["distribucion_objetivo"] = {
            str(key): int(value) for key, value in frame[target].value_counts().items()
        }
    return result
