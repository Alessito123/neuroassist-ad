import numpy as np
import pandas as pd

from preprocessing import build_preprocessor, dataset_summary, prepare_data
from visualization import (
    interactive_boxplot,
    interactive_correlation_heatmap,
    interactive_numeric_distribution,
    interactive_pca,
    interactive_target_distribution,
)


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


def test_interactive_eda_figures_have_renderable_traces():
    frame = sample_frame()
    bundle = prepare_data(frame, "Diagnosis")
    figures = [
        interactive_target_distribution(frame, "Diagnosis"),
        interactive_numeric_distribution(frame, "MMSE", "Diagnosis"),
        interactive_boxplot(frame, "MMSE", "Diagnosis"),
        interactive_correlation_heatmap(frame),
        interactive_pca(bundle),
    ]
    assert all(len(figure.data) > 0 for figure in figures)


def test_string_diagnosis_labels_are_encoded_and_preserved():
    frame = sample_frame()
    frame["Diagnosis"] = frame["Diagnosis"].map({0: "Control", 1: "Alzheimer"})
    bundle = prepare_data(frame, "Diagnosis")
    assert set(bundle.y.unique()) == {0, 1}
    assert set(bundle.class_labels.values()) == {"Control", "Alzheimer"}
    assert bundle.class_labels[0] == "Control"
    assert bundle.class_labels[1] == "Alzheimer"
