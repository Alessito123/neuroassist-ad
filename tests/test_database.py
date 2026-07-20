import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import (
    DatasetProcessed,
    DatasetRaw,
    database_healthcheck,
    get_engine,
    save_diagnosis,
    save_raw_dataset,
)
from scripts.seed_database import seed_public_dataset


def test_sqlite_persistence(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    ok, _ = database_healthcheck(url)
    assert ok
    dataset_id = save_raw_dataset(
        pd.DataFrame({"Age": [70], "Diagnosis": [0]}), "demo", "test", url
    )
    diagnosis_id = save_diagnosis("PAC-1", {"Age": 70}, 0, 0.91, database_url=url)
    assert dataset_id > 0
    assert diagnosis_id > 0


def test_public_seed_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'seed.db'}"
    frame = pd.DataFrame(
        {
            "PatientID": [1, 2, 3, 4],
            "Age": [70, 70, 80, 80],
            "Diagnosis": [0, 0, 1, 1],
        }
    )
    first = seed_public_dataset(url, frame)
    second = seed_public_dataset(url, frame)
    assert first["created"] is True
    assert second["created"] is False
    with Session(get_engine(url)) as session:
        assert session.scalar(select(func.count()).select_from(DatasetRaw)) == 1
        assert session.scalar(select(func.count()).select_from(DatasetProcessed)) == 1
