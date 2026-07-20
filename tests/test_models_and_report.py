import numpy as np
import pandas as pd
from pypdf import PdfReader
from io import BytesIO

from models import evaluate_models
from preprocessing import prepare_data
from report_generator import generate_report
from visualization import confusion_matrix_figure, model_comparison_figure


def test_evaluation_and_pdf_generation():
    rng = np.random.default_rng(42)
    rows = 90
    y = np.tile([0, 1], rows // 2)
    frame = pd.DataFrame(
        {
            "Age": rng.integers(60, 91, rows),
            "MMSE": np.where(y == 1, rng.normal(17, 2, rows), rng.normal(27, 2, rows)),
            "ADL": np.where(y == 1, rng.normal(3, 1, rows), rng.normal(8, 1, rows)),
            "Diagnosis": y,
        }
    )
    bundle = prepare_data(frame, "Diagnosis")
    suite = evaluate_models(
        bundle,
        folds=3,
        sampler="Ninguno",
        model_names=["Random Forest", "XGBoost", "SVM RBF"],
    )
    assert suite.best_model_name in suite.models
    assert suite.comparison["AUC-ROC"].between(0, 1).all()
    figures = [model_comparison_figure(suite), confusion_matrix_figure(suite)]
    pdf = generate_report(frame, "Diagnosis", suite, figures)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 10_000
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) >= 2
    assert "NeuroAssist AD" in (reader.pages[0].extract_text() or "")


def test_pdf_escapes_dynamic_patient_text():
    frame = pd.DataFrame({"Age": [70, 75], "Diagnosis": [0, 1]})
    patient = {
        "patient_code": "PAC-01 <control> & revisión",
        "predicted_class": "Alzheimer <probable>",
        "probability": 0.82,
    }
    pdf = generate_report(frame, "Diagnosis", patient_result=patient)
    assert pdf.startswith(b"%PDF")
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)
    assert "PAC-01 <control> & revisión" in extracted
