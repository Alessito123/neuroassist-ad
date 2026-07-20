"""Inicializa PostgreSQL con el dataset público predeterminado de forma idempotente."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import PUBLIC_DATASET_NAME, SETTINGS
from database import (
    DatasetProcessed,
    DatasetRaw,
    init_database,
    save_processed_summary,
    save_raw_dataset,
)
from preprocessing import dataset_summary, download_public_dataset, prepare_data

LOGGER = logging.getLogger(__name__)


def seed_public_dataset(
    database_url: str | None = None, frame: pd.DataFrame | None = None
) -> dict[str, int | bool]:
    """Persiste el dataset público una sola vez y devuelve IDs/conteos verificables."""
    url = database_url or SETTINGS.database_url
    engine = init_database(url)
    with Session(engine) as session:
        existing_id = session.scalar(
            select(DatasetRaw.id)
            .where(DatasetRaw.nombre == PUBLIC_DATASET_NAME)
            .order_by(DatasetRaw.id)
            .limit(1)
        )
        if existing_id is not None:
            raw_count = session.scalar(select(func.count()).select_from(DatasetRaw)) or 0
            processed_count = (
                session.scalar(select(func.count()).select_from(DatasetProcessed)) or 0
            )
            return {
                "created": False,
                "raw_id": int(existing_id),
                "processed_id": 0,
                "raw_count": int(raw_count),
                "processed_count": int(processed_count),
            }

    data = frame if frame is not None else download_public_dataset()
    bundle = prepare_data(data, "Diagnosis")
    raw_id = save_raw_dataset(data, PUBLIC_DATASET_NAME, SETTINGS.public_dataset_url, url)
    processed_id = save_processed_summary(
        raw_id,
        {
            "target": bundle.target,
            "numericas": bundle.numerical_columns,
            "categoricas": bundle.categorical_columns,
            "eliminadas": bundle.dropped_columns,
            "mapeo_clases": bundle.class_labels,
            "imputacion": "mediana/moda",
            "escalado": "StandardScaler",
        },
        dataset_summary(data, bundle.target),
        url,
    )
    with Session(engine) as session:
        raw_count = session.scalar(select(func.count()).select_from(DatasetRaw)) or 0
        processed_count = (
            session.scalar(select(func.count()).select_from(DatasetProcessed)) or 0
        )
    return {
        "created": True,
        "raw_id": raw_id,
        "processed_id": processed_id,
        "raw_count": int(raw_count),
        "processed_count": int(processed_count),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = seed_public_dataset()
    print(
        "SEED_CREATED={created} RAW_ID={raw_id} PROCESSED_ID={processed_id} "
        "RAW_COUNT={raw_count} PROCESSED_COUNT={processed_count}".format(**result)
    )

