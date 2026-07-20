"""Punto de entrada Streamlit para NeuroAssist AD."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import PUBLIC_DATASET_NAME, RANDOM_SEED, SETTINGS
from database import (
    database_healthcheck,
    save_diagnosis,
    save_model,
    save_processed_summary,
    save_raw_dataset,
)
from models import EvaluationSuite, evaluate_models, predict_patient, tune_model
from preprocessing import (
    DataBundle,
    dataset_summary,
    download_public_dataset,
    infer_target,
    prepare_data,
    read_dataset,
)
from report_generator import generate_report
from visualization import (
    boxplot_figure,
    confusion_matrix_figure,
    correlation_heatmap,
    descriptive_by_group,
    feature_importance_figure,
    model_comparison_figure,
    numeric_distribution_figure,
    pca_figure,
    roc_pr_figure,
    target_distribution_figure,
)

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

st.set_page_config(
    page_title="NeuroAssist AD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_public_dataset() -> pd.DataFrame:
    return download_public_dataset()


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "dataset": None,
        "dataset_name": None,
        "dataset_source": None,
        "target": None,
        "raw_dataset_id": None,
        "suite": None,
        "bundle": None,
        "model_db_id": None,
        "patient_result": None,
        "database_url": SETTINGS.database_url,
        "auto_load_attempted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def assign_dataset(frame: pd.DataFrame, name: str, source: str) -> None:
    st.session_state.dataset = frame
    st.session_state.dataset_name = name
    st.session_state.dataset_source = source
    st.session_state.suite = None
    st.session_state.bundle = None
    st.session_state.model_db_id = None
    st.session_state.patient_result = None
    try:
        st.session_state.target = infer_target(frame)
    except ValueError:
        st.session_state.target = frame.columns[-1] if len(frame.columns) else None


def require_dataset() -> pd.DataFrame | None:
    frame = st.session_state.dataset
    if frame is None:
        st.info("Cargue un dataset en **Gestión de datos** para habilitar este módulo.")
        return None
    return frame


def current_bundle(frame: pd.DataFrame | None = None) -> DataBundle:
    data = frame if frame is not None else st.session_state.dataset
    return prepare_data(data, st.session_state.target)


def sidebar() -> str:
    st.sidebar.markdown("# 🧠 NeuroAssist AD")
    st.sidebar.caption("Analítica y ML para apoyo educativo en Alzheimer")
    sections = [
        "Inicio",
        "1. Gestión de datos",
        "2. EDA",
        "3. Modelado predictivo",
        "4. Evaluación y validación",
        "5. Optimización",
        "6. Selección del mejor modelo",
        "7. Diagnóstico asistido",
        "8. Reportes PDF",
        "Arquitectura y metodología",
    ]
    selected = st.sidebar.radio("Módulo", sections)
    st.sidebar.divider()
    st.sidebar.warning(
        "Uso educativo e investigativo. No constituye ni reemplaza un diagnóstico médico."
    )
    if st.session_state.dataset is not None:
        frame = st.session_state.dataset
        st.sidebar.success(f"Dataset activo: {len(frame):,} × {len(frame.columns)}")
    return selected


def page_home() -> None:
    st.title("NeuroAssist AD")
    st.subheader("Diagnóstico asistido de Alzheimer con aprendizaje automático reproducible")
    st.markdown(
        """
        Esta plataforma reúne gestión de datos, exploración, entrenamiento, validación
        estadística, selección de modelos, inferencia individual y reportes clínico-analíticos.

        El flujo por defecto usa el **Alzheimer's Disease Dataset** de Kaggle (2,149 casos,
        datos sintéticos para educación). También acepta datos tabulares desidentificados y
        extracción básica de biomarcadores desde NIfTI. Ningún resultado debe utilizarse para
        tomar decisiones clínicas sin validación externa, supervisión profesional y revisión de sesgos.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modelos base", "3", "RF · XGBoost · SVM")
    col2.metric("Ensambles", "2", "Stacking · Voting")
    col3.metric("Validación", "5–10 folds", "Estratificada")
    col4.metric("Persistencia", "PostgreSQL", "SQLAlchemy")
    st.info(
        "Comience en **Gestión de datos** para revisar la carga automática y configurar PostgreSQL/Neon."
    )


def page_data_management() -> None:
    st.header("1. Gestión de datos")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Carga")
        uploaded = st.file_uploader(
            "Dataset personalizado desidentificado",
            type=["csv", "xlsx", "xls", "nii", "gz"],
            help="Para modelar se requiere una columna objetivo binaria. NIfTI genera biomarcadores básicos.",
        )
        if uploaded is not None and st.button("Usar archivo cargado", type="primary"):
            try:
                assign_dataset(read_dataset(uploaded), uploaded.name, "Carga local")
                st.success("Archivo leído correctamente.")
            except Exception as exc:
                LOGGER.exception("No se pudo leer el archivo")
                st.error(f"No se pudo leer el archivo: {exc}")
        if st.button("Recargar dataset público de Kaggle"):
            try:
                with st.spinner("Descargando dataset público…"):
                    assign_dataset(
                        cached_public_dataset(), PUBLIC_DATASET_NAME, SETTINGS.public_dataset_url
                    )
                st.success("Dataset público cargado.")
            except Exception as exc:
                st.error(f"La descarga pública falló: {exc}")

    with right:
        st.subheader("PostgreSQL / Neon")
        default_is_sqlite = st.session_state.database_url.startswith("sqlite")
        st.caption(
            "Configure DATABASE_URL en el entorno de producción. La URL no se muestra ni se registra."
        )
        custom_url = st.text_input(
            "URL alternativa para esta sesión",
            type="password",
            placeholder="postgresql://usuario:clave@host/db?sslmode=require",
        )
        if custom_url:
            st.session_state.database_url = custom_url
        if default_is_sqlite:
            st.warning("Sin DATABASE_URL: se usa SQLite local como modo de desarrollo.")
        if st.button("Verificar conexión y crear tablas"):
            ok, message = database_healthcheck(st.session_state.database_url)
            (st.success if ok else st.error)(message)

    frame = st.session_state.dataset
    if frame is None:
        st.info("No hay datos activos.")
        return

    st.divider()
    st.subheader(f"Dataset activo: {st.session_state.dataset_name}")
    st.caption(f"Fuente: {st.session_state.dataset_source}")
    target_index = (
        list(frame.columns).index(st.session_state.target)
        if st.session_state.target in frame.columns
        else len(frame.columns) - 1
    )
    st.session_state.target = st.selectbox(
        "Variable objetivo", frame.columns.tolist(), index=max(target_index, 0)
    )
    summary = dataset_summary(frame, st.session_state.target)
    cols = st.columns(4)
    for column, (label, value) in zip(
        cols,
        [
            ("Filas", summary["filas"]),
            ("Columnas", summary["columnas"]),
            ("Faltantes", summary["faltantes"]),
            ("Duplicados", summary["duplicados"]),
        ],
    ):
        column.metric(label, value)
    st.dataframe(frame.head(100), use_container_width=True)

    action1, action2, action3 = st.columns(3)
    with action1:
        st.download_button(
            "Descargar CSV activo",
            frame.to_csv(index=False).encode("utf-8"),
            "dataset_neuroassist.csv",
            "text/csv",
        )
    with action2:
        if st.button("Persistir datos crudos"):
            try:
                record_id = save_raw_dataset(
                    frame,
                    st.session_state.dataset_name,
                    st.session_state.dataset_source,
                    st.session_state.database_url,
                )
                st.session_state.raw_dataset_id = record_id
                st.success(f"Dataset guardado con ID {record_id}.")
            except Exception as exc:
                LOGGER.exception("Error guardando dataset")
                st.error(f"No fue posible persistirlo: {exc}")
    with action3:
        if st.button("Validar pipeline y guardar resumen"):
            try:
                bundle = current_bundle(frame)
                record_id = save_processed_summary(
                    st.session_state.raw_dataset_id,
                    {
                        "target": bundle.target,
                        "numericas": bundle.numerical_columns,
                        "categoricas": bundle.categorical_columns,
                        "eliminadas": bundle.dropped_columns,
                        "imputacion": "mediana/moda",
                        "escalado": "StandardScaler",
                    },
                    summary,
                    st.session_state.database_url,
                )
                st.success(f"Pipeline válido; resumen procesado ID {record_id}.")
            except Exception as exc:
                st.error(str(exc))


def page_eda() -> None:
    st.header("2. Análisis exploratorio de datos")
    frame = require_dataset()
    if frame is None:
        return
    target = st.session_state.target
    try:
        bundle = current_bundle(frame)
    except Exception as exc:
        st.error(str(exc))
        return
    numeric = [column for column in bundle.numerical_columns if column != target]
    if not numeric:
        st.warning("No se detectaron variables numéricas para los gráficos.")
        return
    selected = st.selectbox("Variable numérica", numeric)
    first, second = st.columns(2)
    first.pyplot(target_distribution_figure(frame, target), clear_figure=True)
    second.pyplot(numeric_distribution_figure(frame, selected, target), clear_figure=True)
    third, fourth = st.columns(2)
    third.pyplot(boxplot_figure(frame, selected, target), clear_figure=True)
    fourth.pyplot(pca_figure(bundle), clear_figure=True)
    st.pyplot(correlation_heatmap(frame), clear_figure=True)
    with st.expander("Estadísticas descriptivas comparativas"):
        st.dataframe(descriptive_by_group(frame, target), use_container_width=True)


def page_modeling() -> None:
    st.header("3. Modelado predictivo")
    frame = require_dataset()
    if frame is None:
        return
    st.markdown(
        "Se evalúan **Random Forest**, **XGBoost**, **SVM RBF**, **Stacking** y "
        "**Voting suave** sobre exactamente los mismos folds estratificados. El preprocesamiento "
        "y el sobremuestreo ocurren dentro de cada fold para evitar fuga de información."
    )
    col1, col2, col3 = st.columns(3)
    folds = col1.select_slider("Folds", options=[3, 5, 10], value=5)
    sampler = col2.selectbox("Balanceo", ["SMOTE", "ADASYN", "Ninguno"])
    scaling = col3.selectbox("Escalado", ["standard", "robust"])
    if st.button("Entrenar y validar los cinco modelos", type="primary"):
        try:
            bundle = current_bundle(frame)
            with st.spinner("Entrenando modelos y generando predicciones out-of-fold…"):
                suite = evaluate_models(bundle, folds, sampler, scaling)
            st.session_state.bundle = bundle
            st.session_state.suite = suite
            st.success(f"Evaluación completada. Mejor modelo: {suite.best_model_name}.")
        except Exception as exc:
            LOGGER.exception("Falló el entrenamiento")
            st.error(f"No se pudo completar el entrenamiento: {exc}")

    suite: EvaluationSuite | None = st.session_state.suite
    if suite:
        st.dataframe(
            suite.comparison.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall / Sensibilidad": "{:.3f}",
                    "F1-Score": "{:.3f}",
                    "AUC-ROC": "{:.3f}",
                    "AUC-PR": "{:.3f}",
                    "Puntaje clínico": "{:.3f}",
                }
            ),
            use_container_width=True,
        )


def page_evaluation() -> None:
    st.header("4. Evaluación y validación")
    suite: EvaluationSuite | None = st.session_state.suite
    bundle: DataBundle | None = st.session_state.bundle
    if suite is None or bundle is None:
        st.info("Entrene primero los modelos en el módulo 3.")
        return
    st.pyplot(roc_pr_figure(bundle, suite), clear_figure=True)
    col1, col2 = st.columns(2)
    col1.pyplot(confusion_matrix_figure(suite), clear_figure=True)
    col2.pyplot(model_comparison_figure(suite), clear_figure=True)
    st.subheader("Prueba global de Friedman sobre AUC-ROC por fold")
    st.json(suite.friedman_test)
    st.subheader("Comparaciones pareadas de McNemar")
    st.dataframe(suite.mcnemar_tests, use_container_width=True)
    st.caption(
        "Los intervalos se estiman con la distribución t sobre los folds. McNemar usa las "
        "predicciones out-of-fold pareadas; p < 0.05 indica diferencia estadísticamente significativa."
    )


def page_optimization() -> None:
    st.header("5. Optimización de hiperparámetros")
    frame = require_dataset()
    if frame is None:
        return
    col1, col2, col3 = st.columns(3)
    name = col1.selectbox("Modelo base", ["Random Forest", "XGBoost", "SVM RBF"])
    iterations = col2.slider("Combinaciones aleatorias", 5, 40, 15, 5)
    folds = col3.select_slider("Folds", [3, 5, 10], value=5)
    if st.button("Optimizar", type="primary"):
        try:
            bundle = current_bundle(frame)
            with st.spinner("Ejecutando RandomizedSearchCV…"):
                search = tune_model(name, bundle, folds=folds, n_iter=iterations)
            st.session_state.optimized_search = search
            st.session_state.optimized_name = name
            st.success(f"Mejor AUC-ROC CV: {search.best_score_:.3f}")
        except Exception as exc:
            LOGGER.exception("Falló la optimización")
            st.error(str(exc))
    search = st.session_state.get("optimized_search")
    if search is not None:
        st.code(str(search.best_params_), language="python")
        results = pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc")
        visible = [
            "rank_test_roc_auc",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_f1",
            "params",
        ]
        st.dataframe(results[visible].head(20), use_container_width=True)
        if st.button("Guardar configuración y modelo optimizado en PostgreSQL"):
            try:
                model_id = save_model(
                    st.session_state.optimized_name,
                    search.best_estimator_,
                    search.best_params_,
                    {"best_cv_auc_roc": search.best_score_},
                    st.session_state.database_url,
                )
                st.session_state.model_db_id = model_id
                st.success(f"Modelo optimizado guardado con ID {model_id}.")
            except Exception as exc:
                st.error(str(exc))


def page_selection() -> None:
    st.header("6. Selección del mejor modelo")
    suite: EvaluationSuite | None = st.session_state.suite
    bundle: DataBundle | None = st.session_state.bundle
    if suite is None or bundle is None:
        st.info("Entrene primero los modelos en el módulo 3.")
        return
    best = suite.best_model_name
    row = suite.comparison.iloc[0]
    st.success(
        f"Modelo seleccionado: **{best}** — puntaje clínico {row['Puntaje clínico']:.3f} "
        f"(promedio de AUC-ROC {row['AUC-ROC']:.3f} y F1 {row['F1-Score']:.3f})."
    )
    st.pyplot(model_comparison_figure(suite), clear_figure=True)
    try:
        st.pyplot(
            feature_importance_figure(bundle, suite.models[best].fitted_model),
            clear_figure=True,
        )
    except Exception as exc:
        st.warning(f"No fue posible calcular la importancia: {exc}")
    friedman = suite.friedman_test
    st.markdown(
        f"**Justificación estadística:** {friedman['interpretacion']} "
        f"(p = {friedman['p_valor']:.4f}). La elección operacional prioriza discriminación "
        "global y equilibrio entre precisión y sensibilidad; debe confirmarse en una cohorte externa."
    )
    if st.button("Persistir el mejor modelo"):
        try:
            result = suite.models[best]
            metrics = {key: value["mean"] for key, value in result.summary.items()}
            model_id = save_model(
                best,
                result.fitted_model,
                result.fitted_model.get_params(deep=False),
                metrics,
                st.session_state.database_url,
            )
            st.session_state.model_db_id = model_id
            st.success(f"Modelo guardado con ID {model_id}.")
        except Exception as exc:
            st.error(str(exc))


def _patient_form(bundle: DataBundle) -> pd.DataFrame | None:
    values: dict[str, Any] = {}
    with st.form("patient_form"):
        patient_code = st.text_input("Código desidentificado", value=f"PAC-{uuid.uuid4().hex[:8].upper()}")
        columns = st.columns(2)
        for index, feature in enumerate(bundle.X.columns):
            series = bundle.X[feature].dropna()
            container = columns[index % 2]
            unique = series.unique().tolist()
            if feature in bundle.categorical_columns or len(unique) <= 8:
                ordered = sorted(unique, key=str)
                values[feature] = container.selectbox(feature, ordered, key=f"patient_{feature}")
            else:
                median = float(pd.to_numeric(series, errors="coerce").median())
                values[feature] = container.number_input(
                    feature, value=median, key=f"patient_{feature}"
                )
        submitted = st.form_submit_button("Estimar probabilidad", type="primary")
    if submitted:
        st.session_state.pending_patient_code = patient_code
        return pd.DataFrame([values], columns=bundle.X.columns)
    return None


def page_diagnosis() -> None:
    st.header("7. Diagnóstico asistido")
    suite: EvaluationSuite | None = st.session_state.suite
    bundle: DataBundle | None = st.session_state.bundle
    if suite is None or bundle is None:
        st.info("Entrene y seleccione modelos antes de usar inferencia individual.")
        return
    best = suite.best_model_name
    model = suite.models[best].fitted_model
    st.caption(f"Modelo activo: {best}")
    method = st.radio("Entrada", ["Formulario", "Archivo de un paciente"], horizontal=True)
    patient_frame: pd.DataFrame | None = None
    patient_code = f"PAC-{uuid.uuid4().hex[:8].upper()}"
    if method == "Formulario":
        patient_frame = _patient_form(bundle)
        patient_code = st.session_state.get("pending_patient_code", patient_code)
    else:
        uploaded = st.file_uploader(
            "Registro CSV/XLSX o NIfTI", type=["csv", "xlsx", "xls", "nii", "gz"], key="patient_upload"
        )
        if uploaded is not None and st.button("Procesar registro"):
            try:
                candidate = read_dataset(uploaded)
                missing = [column for column in bundle.X.columns if column not in candidate]
                if missing:
                    raise ValueError(f"Faltan variables requeridas: {missing}")
                patient_frame = candidate[bundle.X.columns].head(1)
                patient_code = uploaded.name
            except Exception as exc:
                st.error(str(exc))

    if patient_frame is not None:
        try:
            predicted, probability = predict_patient(model, patient_frame)
            positive = suite.models[best].positive_label
            positive_probability = float(
                model.predict_proba(patient_frame)[0][list(model.classes_).index(positive)]
            )
            result = {
                "patient_code": patient_code,
                "predicted_class": predicted,
                "probability": probability,
                "positive_probability": positive_probability,
                "model": best,
                "features": patient_frame.iloc[0].to_dict(),
            }
            st.session_state.patient_result = result
            if predicted == positive:
                st.error(f"Resultado del modelo: clase positiva — confianza {probability:.1%}")
            else:
                st.success(f"Resultado del modelo: clase negativa — confianza {probability:.1%}")
            st.metric("Probabilidad estimada de clase positiva", f"{positive_probability:.1%}")
            st.warning(
                "Esto no es un diagnóstico. Un profesional debe integrar evaluación cognitiva, "
                "historia clínica, neuroimagen, biomarcadores y diagnóstico diferencial."
            )
            try:
                diagnosis_id = save_diagnosis(
                    patient_code,
                    result["features"],
                    predicted,
                    probability,
                    st.session_state.model_db_id,
                    st.session_state.database_url,
                )
                st.success(f"Resultado persistido con ID {diagnosis_id}.")
            except Exception as exc:
                st.warning(f"Predicción calculada, pero no se persistió: {exc}")
        except Exception as exc:
            LOGGER.exception("Falló la inferencia")
            st.error(str(exc))


def page_reports() -> None:
    st.header("8. Reportes PDF")
    frame = require_dataset()
    if frame is None:
        return
    suite: EvaluationSuite | None = st.session_state.suite
    bundle: DataBundle | None = st.session_state.bundle
    st.write(
        "El reporte incluye resumen del dataset, métricas e IC 95 %, selección, matriz de "
        "confusión, curvas ROC/PR, importancia de características y el último resultado individual."
    )
    if st.button("Generar reporte"):
        figures = [target_distribution_figure(frame, st.session_state.target)]
        if suite is not None and bundle is not None:
            figures.extend(
                [
                    model_comparison_figure(suite),
                    confusion_matrix_figure(suite),
                    roc_pr_figure(bundle, suite),
                ]
            )
            try:
                figures.append(
                    feature_importance_figure(
                        bundle, suite.models[suite.best_model_name].fitted_model
                    )
                )
            except Exception:
                LOGGER.exception("No se incluyó importancia en el PDF")
        try:
            pdf = generate_report(
                frame,
                st.session_state.target,
                suite,
                figures,
                st.session_state.patient_result,
            )
            st.session_state.generated_pdf = pdf
        finally:
            for figure in figures:
                plt.close(figure)
    if st.session_state.get("generated_pdf"):
        st.download_button(
            "Descargar reporte NeuroAssist AD",
            st.session_state.generated_pdf,
            "reporte_neuroassist_ad.pdf",
            "application/pdf",
            type="primary",
        )


def page_methodology() -> None:
    st.header("Arquitectura y metodología")
    st.markdown(
        """
        **Random Forest** combina árboles sobre muestras y subconjuntos de variables. Es robusto ante
        relaciones no lineales, permite ponderar clases y ofrece una primera interpretación por importancia.

        **XGBoost** construye árboles secuenciales que corrigen errores previos. Capta interacciones complejas
        y suele rendir bien en tablas clínicas, aunque requiere regularización y vigilancia de sobreajuste.

        **SVM RBF** busca una frontera de máximo margen en un espacio no lineal. El escalado dentro del fold
        es esencial; `C` y `gamma` controlan el compromiso entre margen y complejidad.

        **Stacking** entrega las probabilidades out-of-fold de RF, XGBoost y SVM a una regresión logística.
        El meta-modelo aprende cuándo confiar en cada algoritmo sin entrenarse sobre predicciones in-sample.

        **Voting suave** promedia probabilidades ponderadas de los tres modelos. Reduce varianza y evita que
        una sola familia domine; exige probabilidades razonablemente calibradas.

        La interpretación clínica prioriza sensibilidad, AUC-PR y falsos negativos además de accuracy. La
        selección combina AUC-ROC y F1, se contrasta con Friedman/McNemar y siempre requiere validación externa,
        calibración, análisis de subgrupos, trazabilidad, consentimiento y supervisión humana antes de cualquier uso real.
        """
    )


def auto_load_public_data() -> None:
    if st.session_state.auto_load_attempted or st.session_state.dataset is not None:
        return
    st.session_state.auto_load_attempted = True
    try:
        assign_dataset(cached_public_dataset(), PUBLIC_DATASET_NAME, SETTINGS.public_dataset_url)
    except Exception as exc:
        LOGGER.warning("Carga automática no disponible: %s", exc)
        st.session_state.auto_load_error = str(exc)


def main() -> None:
    initialize_state()
    auto_load_public_data()
    selected = sidebar()
    pages = {
        "Inicio": page_home,
        "1. Gestión de datos": page_data_management,
        "2. EDA": page_eda,
        "3. Modelado predictivo": page_modeling,
        "4. Evaluación y validación": page_evaluation,
        "5. Optimización": page_optimization,
        "6. Selección del mejor modelo": page_selection,
        "7. Diagnóstico asistido": page_diagnosis,
        "8. Reportes PDF": page_reports,
        "Arquitectura y metodología": page_methodology,
    }
    pages[selected]()


if __name__ == "__main__":
    main()
