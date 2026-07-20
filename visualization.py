"""Gráficos EDA, evaluación e interpretación de modelos."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from config import RANDOM_SEED
from models import EvaluationSuite
from preprocessing import DataBundle, build_preprocessor

sns.set_theme(style="whitegrid", palette="colorblind")


def interactive_target_distribution(frame: pd.DataFrame, target: str):
    counts = (
        frame[target]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Grupo")
        .reset_index(name="Pacientes")
    )
    figure = px.bar(
        counts,
        x="Grupo",
        y="Pacientes",
        color="Grupo",
        title="Distribución de la variable objetivo",
        text_auto=True,
    )
    figure.update_layout(showlegend=False)
    return figure


def interactive_numeric_distribution(frame: pd.DataFrame, column: str, target: str):
    plot_data = frame[[column, target]].copy()
    plot_data["Grupo"] = plot_data[target].astype(str)
    return px.histogram(
        plot_data,
        x=column,
        color="Grupo",
        marginal="box",
        barmode="overlay",
        opacity=0.65,
        histnorm="probability density",
        title=f"Distribución interactiva de {column} por grupo",
    )


def interactive_correlation_heatmap(frame: pd.DataFrame, max_columns: int = 20):
    numeric = frame.select_dtypes(include=np.number)
    if numeric.shape[1] > max_columns:
        variances = numeric.var().sort_values(ascending=False)
        numeric = numeric[variances.head(max_columns).index]
    correlation = numeric.corr()
    figure = px.imshow(
        correlation,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Mapa interactivo de correlaciones",
    )
    figure.update_layout(height=650)
    return figure


def interactive_boxplot(frame: pd.DataFrame, column: str, target: str):
    plot_data = frame[[column, target]].copy()
    plot_data["Grupo"] = plot_data[target].astype(str)
    return px.box(
        plot_data,
        x="Grupo",
        y=column,
        color="Grupo",
        points="outliers",
        title=f"{column}: comparación interactiva entre grupos",
    )


def interactive_pca(bundle: DataBundle):
    transformed = build_preprocessor(bundle).fit_transform(bundle.X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(transformed)
    plot_data = pd.DataFrame(
        {
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "Grupo": bundle.y.map(bundle.class_labels).astype(str).to_numpy(),
        }
    )
    figure = px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color="Grupo",
        opacity=0.72,
        title=(
            "PCA interactivo — varianza explicada "
            f"{pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}"
        ),
    )
    return figure


def target_distribution_figure(frame: pd.DataFrame, target: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=frame, x=target, hue=target, legend=False, ax=ax)
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel("Grupo diagnóstico")
    ax.set_ylabel("Pacientes")
    fig.tight_layout()
    return fig


def numeric_distribution_figure(frame: pd.DataFrame, column: str, target: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=frame,
        x=column,
        hue=target,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title(f"Distribución de {column} por grupo")
    fig.tight_layout()
    return fig


def correlation_heatmap(frame: pd.DataFrame, max_columns: int = 20):
    numeric = frame.select_dtypes(include=np.number)
    if numeric.shape[1] > max_columns:
        variances = numeric.var().sort_values(ascending=False)
        numeric = numeric[variances.head(max_columns).index]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        numeric.corr(), cmap="vlag", center=0, square=False, linewidths=0.25, ax=ax
    )
    ax.set_title("Mapa de correlaciones (variables numéricas)")
    fig.tight_layout()
    return fig


def boxplot_figure(frame: pd.DataFrame, column: str, target: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=frame, x=target, y=column, hue=target, legend=False, ax=ax)
    ax.set_title(f"{column}: comparación entre grupos")
    fig.tight_layout()
    return fig


def pca_figure(bundle: DataBundle):
    transformed = build_preprocessor(bundle).fit_transform(bundle.X)
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(transformed)
    plot_data = pd.DataFrame(
        {
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "Grupo": bundle.y.map(bundle.class_labels).astype(str),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=plot_data, x="PC1", y="PC2", hue="Grupo", alpha=0.72, s=45, ax=ax
    )
    ax.set_title("PCA de los predictores preprocesados")
    fig.tight_layout()
    return fig


def model_comparison_figure(suite: EvaluationSuite):
    metrics = ["AUC-ROC", "AUC-PR", "F1-Score", "Recall / Sensibilidad"]
    melted = suite.comparison[["Modelo", *metrics]].melt(
        id_vars="Modelo", var_name="Métrica", value_name="Valor"
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=melted, x="Modelo", y="Valor", hue="Métrica", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Comparación clínica de modelos (media en validación cruzada)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def roc_pr_figure(bundle: DataBundle, suite: EvaluationSuite):
    positive = next(iter(suite.models.values())).positive_label
    y_binary = (bundle.y.to_numpy() == positive).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in suite.models.items():
        fpr, tpr, _ = roc_curve(y_binary, result.oof_probabilities)
        precision, recall, _ = precision_recall_curve(y_binary, result.oof_probabilities)
        roc_auc = result.summary["roc_auc"]["mean"]
        pr_auc = result.summary["pr_auc"]["mean"]
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({pr_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(title="Curvas ROC out-of-fold", xlabel="1 - Especificidad", ylabel="Sensibilidad")
    axes[1].set(title="Curvas Precision–Recall out-of-fold", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(suite: EvaluationSuite, model_name: str | None = None):
    name = model_name or suite.best_model_name
    matrix = suite.models[name].confusion
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negativo", "Positivo"],
        yticklabels=["Negativo", "Positivo"],
        ax=ax,
    )
    ax.set(title=f"Matriz de confusión — {name}", xlabel="Predicción", ylabel="Real")
    fig.tight_layout()
    return fig


def feature_importance_figure(
    bundle: DataBundle, fitted_pipeline: Any, top_n: int = 15
):
    """Usa importancia nativa o, para ensambles/SVM, permutación sobre columnas originales."""
    estimator = fitted_pipeline.named_steps["model"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        names = preprocessor.get_feature_names_out()
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
        names = preprocessor.get_feature_names_out()
    else:
        sample_n = min(500, len(bundle.X))
        sample = bundle.X.sample(sample_n, random_state=RANDOM_SEED)
        y_sample = bundle.y.loc[sample.index]
        permutation = permutation_importance(
            fitted_pipeline,
            sample,
            y_sample,
            scoring="roc_auc",
            n_repeats=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        values = permutation.importances_mean
        names = np.asarray(bundle.X.columns, dtype=str)
    order = np.argsort(values)[-top_n:]
    clean_names = [str(name).replace("num__", "").replace("cat__", "") for name in names[order]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(clean_names, values[order], color=sns.color_palette("viridis", len(order)))
    ax.set_title("Importancia de características")
    ax.set_xlabel("Contribución relativa / importancia por permutación")
    fig.tight_layout()
    return fig


def descriptive_by_group(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = [column for column in frame.select_dtypes(include=np.number) if column != target]
    if not numeric:
        return pd.DataFrame()
    return frame.groupby(target)[numeric].agg(["mean", "std", "median"]).round(3)
