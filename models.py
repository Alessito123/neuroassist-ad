"""Modelos base, ensambles, validación clínica y pruebas estadísticas."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import friedmanchisquare, t
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

from config import RANDOM_SEED
from preprocessing import DataBundle, build_preprocessor, make_sampler

LOGGER = logging.getLogger(__name__)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall / Sensibilidad",
    "f1": "F1-Score",
    "roc_auc": "AUC-ROC",
    "pr_auc": "AUC-PR",
}


@dataclass
class ModelEvaluation:
    name: str
    summary: dict[str, dict[str, float]]
    fold_metrics: pd.DataFrame
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray
    confusion: np.ndarray
    fitted_model: Any
    positive_label: Any


@dataclass
class EvaluationSuite:
    models: dict[str, ModelEvaluation]
    comparison: pd.DataFrame
    mcnemar_tests: pd.DataFrame
    friedman_test: dict[str, float | str]
    best_model_name: str


def base_estimators() -> dict[str, Any]:
    """Instancia los tres algoritmos base con probabilidades calibrables."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=2,
            tree_method="hist",
        ),
        "SVM RBF": SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def all_estimators() -> dict[str, Any]:
    base = base_estimators()
    stack_members = [
        ("rf", clone(base["Random Forest"])),
        ("xgb", clone(base["XGBoost"])),
        ("svm", clone(base["SVM RBF"])),
    ]
    vote_members = [(name, clone(model)) for name, model in stack_members]
    return {
        **base,
        "Stacking": StackingClassifier(
            estimators=stack_members,
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=1500, random_state=RANDOM_SEED
            ),
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1,
        ),
        "Voting suave": VotingClassifier(
            estimators=vote_members,
            voting="soft",
            weights=[2, 2, 1],
            n_jobs=-1,
        ),
    }


def make_model_pipeline(
    estimator: Any,
    bundle: DataBundle,
    sampler: str = "SMOTE",
    scaling: str = "standard",
) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(bundle, scaling=scaling)),
            ("sampler", make_sampler(sampler)),
            ("model", estimator),
        ]
    )


def _positive_probability(model: Any, X: pd.DataFrame, positive_label: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(positive_label)]


def _metrics(y_binary: np.ndarray, predicted_binary: np.ndarray, probability: np.ndarray):
    return {
        "accuracy": accuracy_score(y_binary, predicted_binary),
        "precision": precision_score(y_binary, predicted_binary, zero_division=0),
        "recall": recall_score(y_binary, predicted_binary, zero_division=0),
        "f1": f1_score(y_binary, predicted_binary, zero_division=0),
        "roc_auc": roc_auc_score(y_binary, probability),
        "pr_auc": average_precision_score(y_binary, probability),
    }


def _confidence_interval(values: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sem)
    return {
        "mean": mean,
        "ci_low": max(0.0, mean - margin),
        "ci_high": min(1.0, mean + margin),
    }


def evaluate_models(
    bundle: DataBundle,
    folds: int = 5,
    sampler: str = "SMOTE",
    scaling: str = "standard",
    model_names: list[str] | None = None,
) -> EvaluationSuite:
    """Evalúa modelos con predicciones out-of-fold y luego ajusta cada modelo completo."""
    estimators = all_estimators()
    selected = model_names or list(estimators)
    unknown = set(selected) - set(estimators)
    if unknown:
        raise KeyError(f"Modelos desconocidos: {sorted(unknown)}")

    class_counts = bundle.y.value_counts()
    effective_folds = min(folds, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError("Cada clase necesita al menos dos observaciones.")
    cv = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED
    )
    classes = sorted(bundle.y.unique().tolist(), key=str)
    if len(classes) != 2:
        raise ValueError("Esta versión clínica admite clasificación binaria.")
    positive_label = classes[-1]
    y_binary_all = (bundle.y.to_numpy() == positive_label).astype(int)
    evaluations: dict[str, ModelEvaluation] = {}

    for name in selected:
        LOGGER.info("Evaluando %s con %s folds", name, effective_folds)
        pipeline = make_model_pipeline(estimators[name], bundle, sampler, scaling)
        oof_pred = np.empty(len(bundle.y), dtype=object)
        oof_prob = np.zeros(len(bundle.y), dtype=float)
        fold_rows: list[dict[str, float]] = []

        for fold_index, (train_idx, test_idx) in enumerate(
            cv.split(bundle.X, bundle.y), start=1
        ):
            fold_model = clone(pipeline)
            X_train, X_test = bundle.X.iloc[train_idx], bundle.X.iloc[test_idx]
            y_train, y_test = bundle.y.iloc[train_idx], bundle.y.iloc[test_idx]
            fold_model.fit(X_train, y_train)
            predictions = fold_model.predict(X_test)
            probabilities = _positive_probability(fold_model, X_test, positive_label)
            oof_pred[test_idx] = predictions
            oof_prob[test_idx] = probabilities
            values = _metrics(
                (y_test.to_numpy() == positive_label).astype(int),
                (predictions == positive_label).astype(int),
                probabilities,
            )
            fold_rows.append({"fold": fold_index, **values})

        fold_frame = pd.DataFrame(fold_rows)
        summary = {
            metric: _confidence_interval(fold_frame[metric].to_numpy())
            for metric in METRIC_LABELS
        }
        fitted = clone(pipeline).fit(bundle.X, bundle.y)
        predicted_binary = (oof_pred == positive_label).astype(int)
        evaluations[name] = ModelEvaluation(
            name=name,
            summary=summary,
            fold_metrics=fold_frame,
            oof_predictions=oof_pred,
            oof_probabilities=oof_prob,
            confusion=confusion_matrix(y_binary_all, predicted_binary, labels=[0, 1]),
            fitted_model=fitted,
            positive_label=positive_label,
        )

    comparison = comparison_table(evaluations)
    mcnemar_frame = pairwise_mcnemar(bundle.y, evaluations, positive_label)
    friedman = global_friedman(evaluations, metric="roc_auc")
    return EvaluationSuite(
        models=evaluations,
        comparison=comparison,
        mcnemar_tests=mcnemar_frame,
        friedman_test=friedman,
        best_model_name=str(comparison.iloc[0]["Modelo"]),
    )


def comparison_table(evaluations: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for name, result in evaluations.items():
        row: dict[str, Any] = {"Modelo": name}
        for metric, label in METRIC_LABELS.items():
            values = result.summary[metric]
            row[label] = values["mean"]
            row[f"IC95% {label}"] = f"[{values['ci_low']:.3f}, {values['ci_high']:.3f}]"
        row["Puntaje clínico"] = (
            result.summary["roc_auc"]["mean"] + result.summary["f1"]["mean"]
        ) / 2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["Puntaje clínico", "AUC-ROC"], ascending=False, ignore_index=True
    )


def pairwise_mcnemar(
    y: pd.Series,
    evaluations: dict[str, ModelEvaluation],
    positive_label: Any,
) -> pd.DataFrame:
    truth = (y.to_numpy() == positive_label).astype(int)
    rows = []
    for first, second in combinations(evaluations, 2):
        pred_a = (evaluations[first].oof_predictions == positive_label).astype(int)
        pred_b = (evaluations[second].oof_predictions == positive_label).astype(int)
        correct_a, correct_b = pred_a == truth, pred_b == truth
        table = [
            [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
            [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
        ]
        test = mcnemar(table, exact=(table[0][1] + table[1][0] < 25), correction=True)
        rows.append(
            {
                "Modelo A": first,
                "Modelo B": second,
                "Estadístico": float(test.statistic),
                "p-valor": float(test.pvalue),
                "Diferencia significativa (α=0.05)": bool(test.pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def global_friedman(
    evaluations: dict[str, ModelEvaluation], metric: str = "roc_auc"
) -> dict[str, float | str]:
    if len(evaluations) < 3:
        return {
            "estadistico": float("nan"),
            "p_valor": float("nan"),
            "interpretacion": "Se requieren al menos tres modelos.",
        }
    arrays = [result.fold_metrics[metric].to_numpy() for result in evaluations.values()]
    statistic, p_value = friedmanchisquare(*arrays)
    return {
        "estadistico": float(statistic),
        "p_valor": float(p_value),
        "interpretacion": (
            "Hay evidencia de diferencias globales entre modelos."
            if p_value < 0.05
            else "No se detectaron diferencias globales significativas."
        ),
    }


def tune_model(
    model_name: str,
    bundle: DataBundle,
    folds: int = 5,
    n_iter: int = 15,
    sampler: str = "SMOTE",
) -> RandomizedSearchCV:
    estimators = base_estimators()
    if model_name not in estimators:
        raise ValueError("La optimización está disponible para los tres modelos base.")
    distributions = {
        "Random Forest": {
            "model__n_estimators": [120, 200, 300, 450],
            "model__max_depth": [None, 5, 10, 16],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.7],
        },
        "XGBoost": {
            "model__n_estimators": [100, 180, 260, 350],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
        "SVM RBF": {
            "model__C": [0.1, 0.5, 1, 2, 5, 10, 30],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    }
    effective_folds = min(folds, int(bundle.y.value_counts().min()))
    cv = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED
    )
    search = RandomizedSearchCV(
        estimator=make_model_pipeline(estimators[model_name], bundle, sampler),
        param_distributions=distributions[model_name],
        n_iter=n_iter,
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        return_train_score=False,
    )
    return search.fit(bundle.X, bundle.y)


def predict_patient(model: Any, patient: pd.DataFrame) -> tuple[Any, float]:
    predicted = model.predict(patient)[0]
    probabilities = model.predict_proba(patient)[0]
    classes = list(model.classes_)
    probability = float(probabilities[classes.index(predicted)])
    return predicted, probability

