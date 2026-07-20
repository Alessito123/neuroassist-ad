import numpy as np
import pandas as pd

from preprocessing import build_preprocessor, dataset_summary, prepare_data


def sample_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    diagnosis = np.tile([0, 1], rows // 2)
    return pd.DataFrame(
        {
            "PatientID": np.arange(1000, 1000 + rows),
            "Age": rng.integers(60, 91, rows),
            "MMSE": np.where(diagnosis == 1, rng.normal(18, 3, rows), rng.normal(27, 2, rows)),
            "Gender": rng.choice(["F", "M"], rows),
            "BMI": np.r_[np.nan, rng.normal(25, 3, rows - 1)],
            "Diagnosis": diagnosis,
        }
    )


def test_prepare_data_drops_identifiers_and_builds_transformer():
    frame = sample_frame()
    bundle = prepare_data(frame, "Diagnosis")
    assert "PatientID" in bundle.dropped_columns
    assert "PatientID" not in bundle.X
    transformed = build_preprocessor(bundle).fit_transform(bundle.X)
    assert transformed.shape[0] == len(frame)
    assert np.isfinite(transformed).all()


def test_dataset_summary():
    summary = dataset_summary(sample_frame(), "Diagnosis")
    assert summary["filas"] == 80
    assert summary["faltantes"] == 1
    assert summary["distribucion_objetivo"] == {"0": 40, "1": 40}

