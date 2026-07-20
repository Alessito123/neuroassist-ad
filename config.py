"""Configuración central de la aplicación NeuroAssist AD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
TARGET_CANDIDATES = ("Diagnosis", "diagnosis", "Class", "class", "target", "Target")
PUBLIC_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rabieelkharoua/alzheimers-disease-dataset"
)
PUBLIC_DATASET_NAME = "Alzheimer's Disease Dataset — Rabie El Kharoua (Kaggle, 2024)"


def normalize_database_url(url: str) -> str:
    """Normaliza URLs antiguas de PostgreSQL para SQLAlchemy 2."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


@dataclass(frozen=True)
class Settings:
    """Valores configurables por variables de entorno."""

    database_url: str = normalize_database_url(
        os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'neuroassist.db'}")
    )
    public_dataset_url: str = os.getenv("PUBLIC_DATASET_URL", PUBLIC_DATASET_URL)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    model_dir: Path = BASE_DIR / "artifacts"


SETTINGS = Settings()
SETTINGS.model_dir.mkdir(parents=True, exist_ok=True)

