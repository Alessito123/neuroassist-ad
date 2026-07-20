"""Persistencia SQLAlchemy para datasets, modelos y diagnósticos."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import DateTime, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import SETTINGS

LOGGER = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DatasetRaw(Base):
    __tablename__ = "datasets_raw"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(255), nullable=False)
    fuente: Mapped[str] = mapped_column(String(500), nullable=False)
    filas: Mapped[int] = mapped_column(Integer, nullable=False)
    columnas: Mapped[int] = mapped_column(Integer, nullable=False)
    datos_json: Mapped[str] = mapped_column(Text, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class DatasetProcessed(Base):
    __tablename__ = "datasets_processed"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_raw_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    configuracion_json: Mapped[str] = mapped_column(Text, nullable=False)
    resumen_json: Mapped[str] = mapped_column(Text, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ModeloEntrenado(Base):
    __tablename__ = "modelos_entrenados"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), default="1.0")
    parametros_json: Mapped[str] = mapped_column(Text, nullable=False)
    metricas_json: Mapped[str] = mapped_column(Text, nullable=False)
    modelo_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ResultadoDiagnostico(Base):
    __tablename__ = "resultados_diagnostico"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    modelo_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paciente_codigo: Mapped[str] = mapped_column(String(100), nullable=False)
    entrada_json: Mapped[str] = mapped_column(Text, nullable=False)
    clase_predicha: Mapped[str] = mapped_column(String(100), nullable=False)
    probabilidad: Mapped[str] = mapped_column(String(50), nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def get_engine(database_url: str | None = None):
    """Crea un engine con comprobación de conexión previa a la operación."""
    url = database_url or SETTINGS.database_url
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_database(database_url: str | None = None):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def _session(database_url: str | None = None):
    return sessionmaker(bind=init_database(database_url), expire_on_commit=False)()


def save_raw_dataset(
    frame: pd.DataFrame, nombre: str, fuente: str, database_url: str | None = None
) -> int:
    """Guarda los registros crudos como JSON portable entre SQLite/PostgreSQL."""
    record = DatasetRaw(
        nombre=nombre,
        fuente=fuente,
        filas=len(frame),
        columnas=len(frame.columns),
        datos_json=frame.to_json(orient="records", date_format="iso"),
    )
    with _session(database_url) as session:
        session.add(record)
        session.commit()
        return record.id


def save_processed_summary(
    dataset_raw_id: int | None,
    configuration: dict[str, Any],
    summary: dict[str, Any],
    database_url: str | None = None,
) -> int:
    record = DatasetProcessed(
        dataset_raw_id=dataset_raw_id,
        configuracion_json=json.dumps(configuration, ensure_ascii=False, default=str),
        resumen_json=json.dumps(summary, ensure_ascii=False, default=str),
    )
    with _session(database_url) as session:
        session.add(record)
        session.commit()
        return record.id


def save_model(
    name: str,
    model: Any,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    database_url: str | None = None,
) -> int:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    record = ModeloEntrenado(
        nombre=name,
        parametros_json=json.dumps(parameters, ensure_ascii=False, default=str),
        metricas_json=json.dumps(metrics, ensure_ascii=False, default=str),
        modelo_blob=buffer.getvalue(),
    )
    with _session(database_url) as session:
        session.add(record)
        session.commit()
        return record.id


def load_model(model_id: int, database_url: str | None = None) -> Any:
    with _session(database_url) as session:
        record = session.get(ModeloEntrenado, model_id)
        if record is None:
            raise LookupError(f"No existe el modelo {model_id}")
        return joblib.load(io.BytesIO(record.modelo_blob))


def save_diagnosis(
    patient_code: str,
    features: dict[str, Any],
    predicted_class: Any,
    probability: float,
    model_id: int | None = None,
    database_url: str | None = None,
) -> int:
    record = ResultadoDiagnostico(
        modelo_id=model_id,
        paciente_codigo=patient_code,
        entrada_json=json.dumps(features, ensure_ascii=False, default=str),
        clase_predicha=str(predicted_class),
        probabilidad=f"{probability:.8f}",
    )
    with _session(database_url) as session:
        session.add(record)
        session.commit()
        return record.id


def database_healthcheck(database_url: str | None = None) -> tuple[bool, str]:
    try:
        engine = init_database(database_url)
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return True, "Conexión y esquema verificados"
    except Exception as exc:  # pragma: no cover - depende de infraestructura
        LOGGER.exception("Fallo al verificar la base de datos")
        return False, str(exc)

