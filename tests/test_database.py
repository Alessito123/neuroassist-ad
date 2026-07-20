import pandas as pd

from database import database_healthcheck, save_diagnosis, save_raw_dataset


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

