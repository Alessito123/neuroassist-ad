import pandas as pd
from streamlit.testing.v1 import AppTest


def test_report_flow_keeps_navigation_and_builds_pdf():
    frame = pd.DataFrame(
        {
            "Age": [64, 67, 69, 71, 75, 78, 82, 84],
            "MMSE": [29, 28, 27, 26, 21, 19, 16, 14],
            "Diagnosis": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    app = AppTest.from_file("app.py")
    app.session_state["auto_load_attempted"] = True
    app.session_state["dataset"] = frame
    app.session_state["dataset_name"] = "Dataset sintético de prueba"
    app.session_state["dataset_source"] = "Prueba local"
    app.session_state["target"] = "Diagnosis"
    app.session_state["selected_module"] = "8. Reportes PDF"

    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["selected_module"] == "8. Reportes PDF"

    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["selected_module"] == "8. Reportes PDF"
    assert bytes(app.session_state["generated_pdf"]).startswith(b"%PDF")
